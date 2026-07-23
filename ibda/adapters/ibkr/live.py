"""ibda.adapters.ibkr.live — top-level IBKR live wiring.

Orchestrates the startup sequence for a live IBKR connection:
  1. Start the Deephaven server (engine).
  2. Connect IbkrSupervisor to TWS.
  3. Discover account and wait for positions to settle.
  4. Build canonical live views for each spec (apply_canonical_view) — this
     now includes account_pnl, price_tick, and news_provider alongside the
     original five.
  5. Build the account (LIVE pivot) and cash_balance (LIVE per-currency
     ledger) views from accounts_overview (build_account_view /
     build_cash_balance_view) — both re-tick as TWS pushes updates, unlike
     the earlier table_from_rows() snapshot which froze at connect time.
  6. Optionally subscribe quotes/bars for specified symbols.
  7. Wrap in ibda.connect() → return (supervisor, DataPort).

This module is ORCHESTRATION ONLY — it imports neither deephaven (engine) nor
deephaven_ib/ibapi (vendor) directly.  It calls the engine and vendor helpers
through the adapter modules so the boundary guard stays clean.

Usage::

    from ibda.adapters.ibkr.live import connect_live

    supervisor, port = connect_live(
        client_id=77,
        read_only=True,
        subscribe_symbols=["AAPL", "MSFT"],
    )
    arrow = port.table("position").snapshot()
"""
from __future__ import annotations

import logging
from typing import Any

import ibda
from ibda.adapters.deephaven.server import start_server
from ibda.adapters.deephaven.views import (
    apply_canonical_view,
    build_account_view,
    build_cash_balance_view,
    enrich_position_with_marks,
    snapshot_rows,
)
from ibda.adapters.ibkr.marketdata import subscribe_bars, subscribe_quotes
from ibda.adapters.ibkr.pacing import PacingGovernor
from ibda.adapters.ibkr.specs import CANONICAL_SPECS, POSITION_PORTFOLIO_SPEC
from ibda.adapters.ibkr.supervisor import IbkrSupervisor
from ibda.port import DataPort
from ibda.schema import ALL as ALL_SCHEMAS

logger = logging.getLogger(__name__)


def _resolve_position_spec(supervisor: Any, default_spec: Any) -> Any:
    """Choose the position raw source by availability.

    Returns the default (accounts_positions) spec when that table has rows;
    falls back to POSITION_PORTFOLIO_SPEC (accounts_portfolio) when positions are
    empty but the portfolio stream is populated (read-only TWS login).  Any error
    (e.g. older deephaven-ib without accounts_portfolio) -> default.
    """
    try:
        positions = supervisor.raw_table("accounts_positions")
        if positions is not None and positions.size > 0:
            return default_spec
        portfolio = supervisor.raw_table("accounts_portfolio")
        if portfolio is not None and portfolio.size > 0:
            logger.info(
                "connect_live: accounts_positions empty; using accounts_portfolio "
                "(updatePortfolio) as the position source"
            )
            return POSITION_PORTFOLIO_SPEC
    except Exception:  # noqa: BLE001 — source resolution must never break connect
        logger.warning("connect_live: position-source resolution failed; using default", exc_info=True)
    return default_spec


def _enrich_position_marks(supervisor: Any, tables: dict[str, Any]) -> None:
    """Try to join live marks from accounts_portfolio onto the position view.

    Called only when the position view was built from ``accounts_positions``
    (which lacks live valuation columns).  The join propagates MarketPrice,
    MarketValue, and UnrealizedPnl from ``accounts_portfolio`` into the
    canonical position view so downstream analytics (portfolio_metrics, VaR)
    see real market values rather than null-filled placeholders.

    Graceful degradation: any failure (absent table, empty table, deephaven
    error) is caught and logged at WARNING.  The position view is left
    unchanged — connect_live must never fail due to enrichment.

    Parameters
    ----------
    supervisor:
        The connected IbkrSupervisor.
    tables:
        The in-progress canonical tables dict; ``tables["position"]`` is
        updated in-place when enrichment succeeds.
    """
    try:
        portfolio_raw = supervisor.raw_table("accounts_portfolio")
        if portfolio_raw is None or portfolio_raw.size == 0:
            logger.info(
                "connect_live: accounts_portfolio absent or empty — marks not merged"
            )
            return
        tables["position"] = enrich_position_with_marks(
            tables["position"], portfolio_raw
        )
        logger.info(
            "connect_live: merged live marks from accounts_portfolio (%d rows) "
            "into position view",
            portfolio_raw.size,
        )
    except Exception:  # noqa: BLE001 — enrichment is best-effort; never break connect_live
        logger.warning(
            "connect_live: marks enrichment from accounts_portfolio failed; skipping",
            exc_info=True,
        )


def connect_live(
    *,
    host: str = "127.0.0.1",
    port: int = 7497,
    client_id: int = 77,
    read_only: bool = True,
    settle_secs: int = 5,
    stable_secs: int = 10,
    max_settle_secs: int = 120,
    subscribe_symbols: list[str] | None = None,
    max_hz: float = 5.0,
    engine_port: int = 10000,
) -> tuple[IbkrSupervisor, DataPort]:
    """Connect to TWS, build canonical tables, and return a live DataPort.

    Steps
    -----
    1. Start the Deephaven in-process server on *engine_port*.
    2. Connect IbkrSupervisor to TWS at *host*:*port*.
    3. Discover the managed account; wait for positions to settle.
    4. For each spec in CANONICAL_SPECS, build a canonical live view
       (includes account_pnl, price_tick, news_provider).
    5. Build the account + cash_balance LIVE pivot views from
       accounts_overview (build_account_view / build_cash_balance_view) —
       ticking, not a frozen snapshot.
    6. If *subscribe_symbols* is given, resolve contract IDs from the
       position/definition tables and issue paced quote + bar subscriptions.
    7. Call ``ibda.connect(tables)`` and return ``(supervisor, port)``.

    Parameters
    ----------
    host:
        TWS/Gateway host (default ``127.0.0.1``).
    port:
        TWS API port (default 7497 = paper TWS; use 7496 for live TWS).
    client_id:
        IBKR API client ID.
    read_only:
        True for analytics-only sessions.
    settle_secs:
        Seconds to sleep after the TWS handshake before verifying connectivity.
    stable_secs:
        Consecutive stable-count seconds for positions settlement.
    max_settle_secs:
        Hard timeout on position settlement.
    subscribe_symbols:
        Optional list of instrument symbols to subscribe to quotes and bars.
        If None, no market-data subscriptions are issued.
        **Held-only restriction:** contract IDs are resolved exclusively from
        ``accounts_positions`` (see ``_subscribe_for_symbols``) — a symbol you do
        not currently hold (e.g. a benchmark/watchlist name like ``"QQQ"``) will
        NOT resolve, and no subscription is issued for it; only a WARNING is
        logged (connect_live never fails because of this). A future option is a
        ``reqContractDetails`` fallback to resolve arbitrary symbols regardless
        of holdings — a separate, larger change, not implemented here.
    max_hz:
        Maximum market-data subscription rate (requests/s).
    engine_port:
        Port for the Deephaven in-process server (default 10000).

    Returns
    -------
    tuple[IbkrSupervisor, DataPort]
        The live supervisor (for health checks, account refresh, etc.) and
        the fully wired DataPort.
    """
    # --- Step 1: start Deephaven server ---
    # start_server() is idempotent (see ibda/adapters/deephaven/server.py): it checks
    # deephaven_server.Server.instance FIRST and, if a Server already exists in this
    # process, returns it as-is without attempting a second construction — *port* and
    # *jvm_args* are ignored on that reuse path. This makes connect_live safe to call
    # repeatedly from a background retry loop (e.g. a long-lived feed consumer's
    # backoff-retry worker calls connect_tws -> connect_live -> start_server on every
    # reconnect attempt) without ever risking a second Server() construction in-process.
    #
    # Verified: even a genuine second `Server(...)` construction does not abort the
    # JVM in the installed deephaven_server version — Server.__init__ checks
    # `Server.instance is not None` and raises a catchable `deephaven.DHError` BEFORE
    # touching the JVM (start_jvm() is only called after that guard). start_server()
    # also wraps its own Server(...) call in try/except and re-checks Server.instance
    # on failure as a belt-and-suspenders recovery. A double-Server-construction
    # hypothesis for a silent consumer-process death was tested against this code path
    # and does not hold — the reuse-first check above prevents it.
    logger.info("connect_live: starting Deephaven server on port %d", engine_port)
    start_server(port=engine_port)

    # --- Step 2: connect supervisor to TWS ---
    supervisor = IbkrSupervisor()
    logger.info("connect_live: connecting to TWS %s:%d client_id=%d", host, port, client_id)
    supervisor.connect(
        host=host,
        port=port,
        client_id=client_id,
        read_only=read_only,
        settle_secs=settle_secs,
    )

    # --- Step 3: discover account + wait for positions ---
    account = supervisor.discover_account()
    logger.info("connect_live: account = %r", account)
    pos_count = supervisor.wait_for_positions_settled(
        stable_secs=stable_secs, max_secs=max_settle_secs
    )
    logger.info("connect_live: %d positions settled", pos_count)

    # --- Steps 4-7: build canonical views, wire subscriptions, wire the DataPort ---
    # supervisor.connect() (step 2) is leak-safe on its own (it tears itself down on
    # failure — see IbkrSupervisor.connect()'s try/finally). Everything below it is
    # NOT: if any of it raises, the supervisor would otherwise be left CONNECTED
    # (holding client_id) with no tables/DataPort ever returned to the caller. A
    # caller that retries connect_live() after such a failure then hits IB error 326
    # ("client id already in use") because the leaked session never released it.
    # Wrap the whole post-connect sequence and disconnect on any failure — mirroring
    # the same idempotent/exception-safe supervisor.disconnect() connect() itself
    # uses in its finally block — before re-raising the original exception.
    try:
        # --- Step 4: build canonical live views ---
        tables: dict[str, Any] = {}
        _position_spec_used: Any = None  # track which spec was chosen for enrichment guard
        for canonical_name, spec in CANONICAL_SPECS.items():
            if canonical_name == "position":
                spec = _resolve_position_spec(supervisor, spec)
                _position_spec_used = spec
            schema = ALL_SCHEMAS[spec.schema_name]
            raw = supervisor.raw_table(spec.raw_table)
            join_raw = supervisor.raw_table(spec.join_table) if spec.join_table else None
            tables[canonical_name] = apply_canonical_view(raw, spec, schema, join_raw=join_raw)
            logger.info("connect_live: built canonical view for %r", canonical_name)

        # --- Step 4b: enrich position marks from accounts_portfolio ---
        # Only when the position source is accounts_positions (no live marks).
        # When POSITION_PORTFOLIO_SPEC was selected, accounts_portfolio IS the source
        # and marks are already present natively — no join needed.
        if (
            _position_spec_used is not None
            and _position_spec_used.raw_table == "accounts_positions"
        ):
            _enrich_position_marks(supervisor, tables)

        # --- Step 5: account (LIVE pivot) + cash_balance (LIVE per-currency ledger) ---
        # account_pnl is built above in the Step 4 loop (it is now a plain
        # CANONICAL_SPECS entry, since accounts_pnl is columnar — no pivot needed).
        # account/cash_balance still need a hand-written pivot because
        # accounts_overview is key-value, not columnar.
        overview_raw = supervisor.raw_table("accounts_overview")
        tables["account"] = build_account_view(overview_raw, account=account)
        tables["cash_balance"] = build_cash_balance_view(overview_raw, account=account)
        logger.info("connect_live: built live account + cash_balance pivot views")

        # --- Step 6: optional market-data subscriptions ---
        if subscribe_symbols:
            _subscribe_for_symbols(supervisor, subscribe_symbols, max_hz=max_hz)

        # --- Step 7: wire the DataPort ---
        data_port: DataPort = ibda.connect(tables)
        logger.info("connect_live: DataPort ready with tables: %s", sorted(tables))
    except Exception:
        logger.warning(
            "connect_live: post-connect setup failed after a successful TWS "
            "connect — disconnecting the supervisor to release client_id=%d "
            "before propagating the error (otherwise a retry would hit IB "
            "error 326, client id already in use)",
            client_id,
            exc_info=True,
        )
        supervisor.disconnect()
        raise

    return supervisor, data_port


def _subscribe_for_symbols(
    supervisor: IbkrSupervisor,
    symbols: list[str],
    max_hz: float,
) -> None:
    """Resolve contract IDs for *symbols* and issue paced quote+bar subscriptions.

    Held-only: resolution is scoped to ``accounts_positions`` only, so a symbol not
    currently held resolves to nothing (see ``connect_live``'s ``subscribe_symbols``
    docstring for the full caveat).
    """
    governor = PacingGovernor(max_hz=max_hz)

    # Resolve contract IDs from accounts_positions (symbols held).
    try:
        pos_rows = snapshot_rows(supervisor.raw_table("accounts_positions"))
        sym_set = set(symbols)
        contract_ids: list[int] = [
            int(r["ContractId"])
            for r in pos_rows
            if r.get("Symbol") in sym_set and r.get("ContractId") is not None
        ]
    except Exception as exc:  # noqa: BLE001 — best-effort
        logger.warning("connect_live: could not resolve contract IDs: %s", exc)
        return

    if not contract_ids:
        logger.warning(
            "connect_live: none of %s found in current positions; no subscriptions issued",
            symbols,
        )
        return

    logger.info(
        "connect_live: subscribing quotes+bars for %d contract(s)", len(contract_ids)
    )
    subscribe_quotes(supervisor, contract_ids, governor)
    subscribe_bars(supervisor, contract_ids, PacingGovernor(max_hz=max_hz))
