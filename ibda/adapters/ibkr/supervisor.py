"""ibda.adapters.ibkr.supervisor — IBKR session supervisor.

VENDOR-CONFINED: this module imports deephaven_ib.  It MUST NOT import
deephaven (engine).  For table reads it delegates to
``ibda.adapters.deephaven.views.snapshot_rows`` (engine helper).

Responsibilities:
- connect to TWS via IbSessionTws (one session; reconnect not implemented).
- discover the managed account from accounts_managed.
- wait for accounts_positions to settle (large books take ~75 s).
- expose raw deephaven-ib tables via ``raw_table(name)``.
- report a health dict: connected flag, account, position count, last_error.
- pivot accounts_overview (key→value) into one canonical account row dict.

The account pivot (``canonical_account_snapshot``) is v1 snapshot-grade:
it calls snapshot_rows at invocation time and returns a static dict.  Unlike
the live-view tables produced by apply_canonical_view, the account table does
not update on its own; callers must call this method again to refresh.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapping from IBKR accounts_overview Key values → canonical output column names.
# IBKR emits "MaintMarginReq" but the canonical schema column is "MaintMargin".
# ---------------------------------------------------------------------------
_ACCOUNT_METRIC_KEYS: dict[str, str] = {
    "NetLiquidation": "NetLiquidation",
    "BuyingPower": "BuyingPower",
    "MaintMarginReq": "MaintMargin",        # IBKR emits MaintMarginReq; canonical col is MaintMargin
    "GrossPositionValue": "GrossPositionValue",
}


class IbkrSupervisor:
    """Lifecycle manager for an IBKR deephaven-ib session.

    Parameters
    ----------
    snapshot_rows_fn:
        Injectable snapshot helper — defaults to
        ``ibda.adapters.deephaven.views.snapshot_rows``.  Override in
        tests with a pure stub that returns pre-built row lists.
    """

    def __init__(
        self,
        snapshot_rows_fn: Callable[[Any], list[dict[str, Any]]] | None = None,
        snapshot_rows_where_fn: Callable[[Any, str], list[dict[str, Any]]] | None = None,
    ) -> None:
        self._session: Any = None
        self._account: str = ""
        self._last_error: str = ""
        if snapshot_rows_fn is None:
            from ibda.adapters.deephaven.views import snapshot_rows
            self._snapshot_rows: Callable[[Any], list[dict[str, Any]]] = snapshot_rows
        else:
            self._snapshot_rows = snapshot_rows_fn
        # snapshot_rows_where_fn is injected by tests to avoid a JVM dependency;
        # None means the production views.snapshot_rows_where path is used at call time.
        self._snapshot_rows_where_fn: Callable[[Any, str], list[dict[str, Any]]] | None = (
            snapshot_rows_where_fn
        )

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    @classmethod
    def from_session(
        cls,
        session: Any,
        snapshot_rows_fn: Callable[[Any], list[dict[str, Any]]] | None = None,
        snapshot_rows_where_fn: Callable[[Any, str], list[dict[str, Any]]] | None = None,
    ) -> IbkrSupervisor:
        """Wrap an already-connected, externally-owned session (no connect/settle).

        Use this to reuse a session that someone else owns and has already
        connected — e.g. the strategy engine's single TWS session — so the
        whole process keeps exactly **one** TWS connection (no second
        connect(), no client-id collision, no second JVM start, no position
        settle).  This supervisor does not own the session lifecycle: it never
        calls ``connect()`` and never disconnects.

        Parameters
        ----------
        session:
            An already-connected deephaven-ib ``IbSessionTws`` (or compatible).
            Assigned directly to ``_session`` as-is.
        snapshot_rows_fn:
            Same injection point as ``__init__`` — optional snapshot helper,
            defaulting to ``ibda.adapters.deephaven.views.snapshot_rows``.

        Returns
        -------
        IbkrSupervisor
            A supervisor whose ``_session`` is the supplied session, with
            ``_last_error=""`` and ``_account=""`` (call ``discover_account``
            if the managed account is needed).
        """
        sup = cls(
            snapshot_rows_fn=snapshot_rows_fn,
            snapshot_rows_where_fn=snapshot_rows_where_fn,
        )
        sup._session = session
        sup._last_error = ""
        return sup

    def connect(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 77,
        read_only: bool = True,
        settle_secs: int = 5,
    ) -> None:
        """Connect to TWS and wait for the session to stabilise.

        Forces a fresh connection: if a prior session is still live (this
        supervisor was already connected and ``connect()`` is called again
        without an intervening ``is_connected()`` check — e.g. a reconnect
        attempt after a farm blip that did not actually drop the old
        session), the prior session is cleanly disconnected first. This
        keeps every call's parameters (``host``/``port``/``client_id``/
        ``read_only``) authoritative rather than silently no-op'ing (and
        ignoring possibly-different parameters) when a session already
        exists.

        Parameters
        ----------
        host:
            TWS/Gateway host.
        port:
            7497 = paper TWS; 7496 = live TWS.
        client_id:
            IBKR API client ID (must be unique per concurrent session).
        read_only:
            True for analytics-only sessions (no order placement).
        settle_secs:
            Seconds to sleep after connect() before verifying connectivity.
        """
        # Guard against a double-connect leaking the prior JNI-backed session
        # (open socket + native threads): if self._session is already live,
        # disconnect it before building a new one. is_connected() itself is
        # guarded — a session object that exists but is unreadable is treated
        # as still-live (disconnect() below is idempotent/exception-safe
        # either way, so defaulting to "disconnect it" is always safe).
        if self._session is not None:
            try:
                still_connected = self._session.is_connected()
            except Exception as exc:  # noqa: BLE001 — untyped deephaven-ib; treat as still-live and disconnect defensively
                logger.debug(
                    "connect: prior session.is_connected() raised; disconnecting "
                    "defensively: %s", exc
                )
                still_connected = True
            if still_connected:
                logger.warning(
                    "connect: called while a session is already connected — "
                    "disconnecting the prior session before establishing a new one"
                )
                self.disconnect()

        import deephaven_ib as dhib

        session = dhib.IbSessionTws(
            host=host,
            port=port,
            client_id=client_id,
            read_only=read_only,
            download_short_rates=False,
        )
        # Assign immediately (construction succeeded) so that ANY failure below
        # — session.connect() raising, or the connectivity check failing — can
        # be cleaned up via the existing idempotent self.disconnect(), instead
        # of leaking a JNI-backed session that was created but never attached
        # to self. connected=False until the check passes; the finally block
        # disconnects on every path that leaves it False. The real error (a
        # RuntimeError we raise, or whatever session.connect() raises) is
        # never swallowed — it propagates normally through the try/finally.
        self._session = session
        connected = False
        try:
            session.connect()
            time.sleep(settle_secs)

            if not session.is_connected():
                self._last_error = "disconnected immediately after connect"
                raise RuntimeError(
                    f"{self._last_error} — check TWS API settings / client_id={client_id}"
                )
            connected = True
        finally:
            if not connected:
                self.disconnect()

        self._last_error = ""

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def disconnect(self) -> None:
        """Disconnect the underlying TWS session.

        Idempotent: safe to call when already disconnected or when no session
        has been established.  Any error from the underlying ``disconnect()``
        call is caught and logged at DEBUG — double-disconnect is benign and
        must never propagate.  After this call ``is_connected()`` returns
        ``False``.
        """
        if self._session is None:
            logger.debug("disconnect: no session to disconnect")
            return
        try:
            self._session.disconnect()
            logger.debug("disconnect: session disconnected")
        except Exception as exc:  # noqa: BLE001 — deephaven-ib is untyped; double-disconnect is benign
            logger.debug(
                "disconnect: session.disconnect() raised (already disconnected?): %s", exc
            )
        finally:
            self._session = None

    # ------------------------------------------------------------------
    # Account discovery
    # ------------------------------------------------------------------

    def discover_account(self, fallback: str | None = None) -> str:
        """Read the live managed account from accounts_managed.

        Prefers the live table; uses *fallback* (and warns) if the table is
        empty or unreadable.  Result is stored as ``self._account``.

        Parameters
        ----------
        fallback:
            Fallback account id string when discovery fails.

        Returns
        -------
        str
            The discovered (or fallback) account id.
        """
        self._account = ""
        try:
            rows = self._snapshot_rows(self._session.tables["accounts_managed"])
            accounts = [str(r.get("Account", "")) for r in rows if r.get("Account")]
            if accounts:
                self._account = accounts[0]
                return self._account
            msg = "accounts_managed is empty"
        except Exception as exc:  # noqa: BLE001 — surfaced as a warning, not fatal
            msg = f"could not read accounts_managed: {exc}"

        # Discovery did not yield an account. Surface why via the health dict
        # (last_error) rather than dropping it — the caller may have passed a
        # fallback, but the reason discovery failed is still worth reporting.
        self._last_error = msg

        if fallback is not None:
            self._account = fallback
            return self._account

        return ""

    # ------------------------------------------------------------------
    # Position settlement
    # ------------------------------------------------------------------

    def wait_for_positions_settled(
        self,
        stable_secs: int = 10,
        max_secs: int = 120,
    ) -> int:
        """Poll accounts_positions until the row count stabilises.

        Large books (2000+ positions) take ~75 s to stream in after connect.

        Parameters
        ----------
        stable_secs:
            Consecutive stable-count seconds required before returning.
        max_secs:
            Hard timeout.

        Returns
        -------
        int
            The final settled (or timed-out) position row count.
        """
        last_count = -1
        stable_for = 0

        for _ in range(max_secs):
            try:
                rows = self._snapshot_rows(self._session.tables["accounts_positions"])
                n = len(rows)
                if n == 0:
                    # Read-only-login fallback: reqPositions withheld, updatePortfolio served.
                    # accounts_portfolio is optional (absent in older deephaven-ib versions
                    # and non-portfolio login modes).  Use 'in' + [] to stay consistent with
                    # the subscript interface used everywhere else in this module; .get() would
                    # silently return None if tables is a custom object without that method.
                    if "accounts_portfolio" in self._session.tables:
                        portfolio = self._session.tables["accounts_portfolio"]
                        n = int(portfolio.size)
            except Exception as exc:  # noqa: BLE001 — transient read error; keep waiting
                logger.warning(
                    "wait_for_positions_settled: snapshot error, treating as not settled: %s", exc
                )
                time.sleep(1)
                continue
            if n != last_count:
                stable_for = 0
                last_count = n
            else:
                stable_for += 1
                if stable_for >= stable_secs and n > 0:
                    return n
            time.sleep(1)

        return last_count

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def is_connected(self) -> bool:
        """Return True if the session is alive and connected."""
        if self._session is None:
            return False
        result: bool = self._session.is_connected()
        return result

    def health(self) -> dict[str, Any]:
        """Return a health snapshot dict.

        Returns
        -------
        Dict with keys: connected (bool), account (str), position_count (int),
        last_error (str).
        """
        if not self.is_connected():
            return {
                "connected": False,
                "account": self._account,
                "position_count": 0,
                "last_error": self._last_error,
            }
        try:
            rows = self._snapshot_rows(self._session.tables["accounts_positions"])
            pos_count = len(rows)
        except Exception as exc:  # noqa: BLE001 — health should never raise
            pos_count = 0
            self._last_error = str(exc)
        return {
            "connected": True,
            "account": self._account,
            "position_count": pos_count,
            "last_error": self._last_error,
        }

    # ------------------------------------------------------------------
    # Raw table access
    # ------------------------------------------------------------------

    def raw_table(self, name: str) -> Any:
        """Return a raw deephaven-ib table by name.

        Parameters
        ----------
        name:
            The deephaven-ib table name (e.g. ``"accounts_positions"``).

        Returns
        -------
        The live deephaven Table (untyped; Any).
        """
        return self._session.tables[name]

    def snapshot_raw_rows(self, name: str) -> list[dict[str, Any]]:
        """Snapshot a raw deephaven-ib table by name to a list of row dicts.

        Reuses the injected ``snapshot_rows`` helper so this stays engine-free
        (the deephaven import lives inside that helper). Convenience for
        vendor-confined modules (e.g. marketdata) that need to read a raw table
        without importing the engine themselves.

        Parameters
        ----------
        name:
            The deephaven-ib table name (e.g. ``"ticks_price"``).
        """
        return self._snapshot_rows(self.raw_table(name))

    def snapshot_raw_rows_where(self, name: str, predicate: str) -> list[dict[str, Any]]:
        """Snapshot a raw table filtered in the engine before materialization.

        Production path: calls Deephaven's ``Table.where(predicate)`` before
        ``to_pandas``, so only matching rows cross into Python — O(matching)
        vs O(total) per call.  Critical for append-only tick tables that grow
        unbounded over a hot session.

        Test path: if a ``snapshot_rows_where_fn`` was injected at construction
        time, that callable is used instead (JVM-free).

        Parameters
        ----------
        name:
            The deephaven-ib table name (e.g. ``"ticks_price"``).
        predicate:
            A Deephaven filter expression, e.g. ``"ContractId == 12345"`` or
            ``"RequestId == 77"``.

        Returns
        -------
        Filtered list of row dicts — only rows matching *predicate*.
        """
        raw = self.raw_table(name)
        if self._snapshot_rows_where_fn is not None:
            return self._snapshot_rows_where_fn(raw, predicate)
        from ibda.adapters.deephaven.views import snapshot_rows_where  # noqa: PLC0415
        return snapshot_rows_where(raw, predicate)

    # ------------------------------------------------------------------
    # Account pivot
    # ------------------------------------------------------------------

    def canonical_account_snapshot(self) -> list[dict[str, Any]]:
        """Pivot accounts_overview key-value rows into one canonical account row.

        Reads accounts_overview at call time (snapshot-grade: v1 returns a
        static snapshot; callers must invoke again to refresh).

        The canonical account schema has these metric columns:
        NetLiquidation, BuyingPower, MaintMargin, GrossPositionValue.
        Rows for the current account and Currency = "USD" are used; the
        first Currency found is used when no USD row is present.

        Returns
        -------
        list[dict[str, Any]]
            A one-element list containing the canonical account row dict
            with keys: Account, Timestamp, NetLiquidation, BuyingPower,
            MaintMargin, GrossPositionValue, Currency.  Returns an empty
            list if the accounts_overview table is unreadable.
        """
        from datetime import datetime, timezone

        try:
            rows = self._snapshot_rows(
                self._session.tables["accounts_overview"]
            )
        except Exception as exc:  # noqa: BLE001 — return empty on any read failure
            logger.warning(
                "canonical_account_snapshot: error reading accounts_overview: %s", exc
            )
            self._last_error = str(exc)
            return []

        # Filter to the managed account (if known) and to metric keys we care about.
        # accounts_overview has columns: Account, Key, Value, Currency, ...
        # (verified against a live session: Key, DoubleValue + Tag/Currency)
        # deephaven-ib may expose DoubleValue (numeric) and Value (string) columns.
        # We attempt DoubleValue first, then Value (parsed as float).
        #
        # Two-pass: collect metrics by currency first, then prefer USD (FIX 5).
        # The previous single-pass code overwrote `currency`/`metrics` on every
        # valid row, so the last-seen currency won on a multi-currency account.
        per_currency: dict[str, dict[str, float]] = {}
        first_currency: str | None = None

        for row in rows:
            account_col = row.get("Account", "")
            if self._account and account_col != self._account:
                continue
            key = str(row.get("Key", ""))
            if key not in _ACCOUNT_METRIC_KEYS:
                continue
            canonical = _ACCOUNT_METRIC_KEYS[key]
            # Try DoubleValue first (numeric column), then Value (string).
            raw_val = row.get("DoubleValue")
            if raw_val is None:
                raw_val = row.get("Value")
            try:
                fval = float(raw_val)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            row_currency = str(row.get("Currency") or "USD")
            if first_currency is None:
                first_currency = row_currency
            if row_currency not in per_currency:
                per_currency[row_currency] = {}
            per_currency[row_currency][canonical] = fval

        # Prefer USD rows; fall back to the first currency seen when no USD rows exist.
        if "USD" in per_currency:
            currency: str = "USD"
            metrics: dict[str, float] = per_currency["USD"]
        elif first_currency is not None:
            currency = first_currency
            metrics = per_currency[first_currency]
        else:
            return []

        if not metrics:
            return []

        return [
            {
                "Account": self._account,
                "Timestamp": datetime.now(tz=timezone.utc),
                "NetLiquidation": metrics.get("NetLiquidation"),
                "BuyingPower": metrics.get("BuyingPower"),
                "MaintMargin": metrics.get("MaintMargin"),
                "GrossPositionValue": metrics.get("GrossPositionValue"),
                "Currency": currency,
            }
        ]
