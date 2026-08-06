"""ibda.adapters.ibkr.pnl — PnL snapshot helpers for IBKR.

VENDOR-CONFINED: this module may access deephaven-ib objects only through the
supervisor (``supervisor._session``).  It MUST NOT import ``deephaven`` (engine)
directly; engine work lives in ``ibda.adapters.deephaven``.

Public API
----------
``snapshot_account_pnl(supervisor) -> pa.Table``
    Build a canonical ``account_pnl`` Arrow table from the live
    ``accounts_pnl`` deephaven-ib table (auto-populated on connect).

``snapshot_position_pnl(supervisor, conid_to_sym) -> pa.Table``
    Build a canonical ``position_pnl`` Arrow table from the live
    ``accounts_pnl_single`` deephaven-ib table.  ``conid_to_sym`` maps
    ConId (int) → Sym (str) for the symbol-join; rows whose ConId is not in
    the map get ``Sym = None``.

``request_single_pnl_for_conids(supervisor, account, conids) -> None``
    Issue ``reqPnLSingle`` for each conid in the bounded list.  Resolves each
    conid to a ``RegisteredContract`` (one ``reqContractDetails`` round-trip)
    and calls ``session.request_single_pnl``.  Best-effort: individual failures
    are logged at WARNING, not raised.

Design notes
------------
- ``accounts_pnl`` canonical view columns:
    RequestId, ReceiveTime, Account, ModelCode, DailyPnl, UnrealizedPnl, RealizedPnl
  ``DailyPnl`` uses a lowercase 'l' (confirmed from deephaven-ib ``__init__.py``
  line 604 where the canonical view is constructed).

- ``accounts_pnl_single`` canonical view columns:
    RequestId, ReceiveTime, Account, ModelCode, ConId, Position,
    DailyPnL, UnrealizedPnL, RealizedPnL, Value
  ``ConId`` arrives as a **String** (parsed from request Note); this adapter
  casts it to int.  ``DailyPnL`` uses capital 'L' (different from account-level).

- ``request_single_pnl`` on IbSessionTws takes a ``RegisteredContract`` (not a
  bare conid), so we must call ``get_registered_contract(Contract(conId=...))``
  for each conid before subscribing.  Same pattern as ``subscribe_bars`` in
  ``ibda.adapters.ibkr.marketdata``.
"""
from __future__ import annotations

import logging
import math
from typing import Any, cast

import pyarrow as pa

from ibda.adapters.ibkr.contracts import build_contract_from_conid
from ibda.adapters.ibkr.supervisor import IbkrSupervisor
from ibda.schema.account_pnl import ACCOUNT_PNL
from ibda.schema.position_pnl import POSITION_PNL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal row → Arrow helpers (pure; no engine, no vendor)
# ---------------------------------------------------------------------------


def _account_pnl_arrow_from_rows(
    rows: list[dict[str, Any]],
) -> pa.Table:
    """Project raw ``accounts_pnl`` snapshot rows to a canonical ACCOUNT_PNL Arrow table.

    Args:
        rows: Raw snapshot rows from the ``accounts_pnl`` deephaven-ib table.
            Expected keys: Account, ReceiveTime, DailyPnl, UnrealizedPnl, RealizedPnl.

    Returns:
        A canonical ``ACCOUNT_PNL`` Arrow table (all ACCOUNT_PNL columns, schema order).
    """
    canonical: list[dict[str, Any]] = [
        {
            "Account": str(r.get("Account") or ""),
            "Timestamp": r.get("ReceiveTime"),
            "DailyPnl": _f(r.get("DailyPnl")),
            "UnrealizedPnl": _f(r.get("UnrealizedPnl")),
            "RealizedPnl": _f(r.get("RealizedPnl")),
        }
        for r in rows
    ]
    return _rows_to_arrow(canonical, ACCOUNT_PNL)


def _position_pnl_arrow_from_rows(
    rows: list[dict[str, Any]],
    conid_to_sym: dict[int, str],
) -> pa.Table:
    """Project raw ``accounts_pnl_single`` snapshot rows to a canonical POSITION_PNL Arrow table.

    ``ConId`` arrives as a String in the raw rows and is cast to int here.  ``Sym``
    is resolved from ``conid_to_sym``; unresolved ConIds produce ``Sym = None``.

    Args:
        rows: Raw snapshot rows from the ``accounts_pnl_single`` deephaven-ib table.
            Expected keys: Account, ConId (str), DailyPnL, UnrealizedPnL, RealizedPnL, Value.
        conid_to_sym: Mapping from ConId (int) to Sym (str) for symbol resolution.

    Returns:
        A canonical ``POSITION_PNL`` Arrow table (all POSITION_PNL columns, schema order).
    """
    canonical: list[dict[str, Any]] = []
    for r in rows:
        raw_conid = r.get("ConId")
        try:
            conid = int(raw_conid) if raw_conid is not None else None
        except (TypeError, ValueError):
            logger.warning("pnl: could not parse ConId %r as int — skipping row", raw_conid)
            conid = None

        canonical.append(
            {
                "Account": str(r.get("Account") or ""),
                "ConId": conid,
                "Sym": conid_to_sym.get(conid) if conid is not None else None,
                "DailyPnl": _f(r.get("DailyPnL")),   # capital L in accounts_pnl_single
                "UnrealizedPnl": _f(r.get("UnrealizedPnL")),
                "RealizedPnl": _f(r.get("RealizedPnL")),
                "MarketValue": _f(r.get("Value")),
            }
        )
    return _rows_to_arrow(canonical, POSITION_PNL)


def _f(val: Any) -> float | None:
    """Cast a raw value to float, returning ``None`` for every "no value" encoding.

    Three encodings reach this function, and each of them must become ``None`` rather
    than a number, because every consumer of these columns sums them:

    - ``None`` — the column was absent from the row.
    - ``NaN`` — kept as a real branch, but **not** for the reason this docstring used to
      give. It claimed NaN is "what ``deephaven.pandas.to_pandas`` produces for a
      Deephaven null, which is how these rows arrive". Measured against a live JVM, it
      is not: a ``(double)null`` reaches Python as plain ``None`` through
      both paths — ``snapshot_rows`` yields ``None`` directly, and ``to_pandas`` yields a
      nullable ``Float64`` whose ``<NA>`` collapses to ``None`` (not ``pandas.NA``) under
      ``to_dict``. So on this path the ``None`` branch above is what catches a Deephaven
      null, and the NaN branch is belt-and-braces against the other ways a NaN can enter
      a float column. The underlying fix was still real — ``nan >= 1.797e308`` is
      ``False``, so before it existed a NaN passed straight through, and one NaN poisons
      an entire ``sum()``; only the account of where the NaN came from was wrong.
    - ``±Double.MAX_VALUE`` — IBKR's "not yet computed" sentinel. The magnitude test is
      ``abs()`` because **IB emits it with either sign**; see ``map_null_value`` in the
      vendored ``deephaven-ib`` fork (``_tws/ib_type_logger.py``), which states both have
      been observed live and is the canonical rule this mirrors. The previous one-sided
      ``>=`` let ``-Double.MAX_VALUE`` through as a real number roughly
      ``-1.8e308`` — arithmetically catastrophic in a sum, and indistinguishable from a
      genuine loss in a column of dollars.

    Not shared with ``ibda/adapters/deephaven/views.py``'s DQL scrub, which correctly
    uses a one-sided ``>=``: there the value stays inside the engine, where a null *is*
    ``-Double.MAX_VALUE`` and must pass through untouched. The two tests differ because
    the two representations differ — that is stated in both places so neither is
    "corrected" to match the other.
    """
    if val is None:
        return None
    try:
        result = float(val)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    if abs(result) >= 1.7976931348623157e+308:
        return None
    return result


def _rows_to_arrow(rows: list[dict[str, Any]], schema_obj: Any) -> pa.Table:
    """Project a list of canonical row dicts into an Arrow table matching *schema_obj*.

    Args:
        rows: Canonical row dicts keyed by schema column names.
        schema_obj: A ``ibda.schema._core.Schema`` instance.

    Returns:
        A ``pa.Table`` with the schema's columns in order.
    """
    columns: list[pa.Array] = []
    for col in schema_obj.columns:
        columns.append(
            pa.array([r.get(col.name) for r in rows], type=col.dtype.to_arrow())
        )
    return cast(
        pa.Table,
        pa.table(
            dict(zip(schema_obj.column_names, columns, strict=True)),
            schema=schema_obj.to_arrow_schema(),
        ),
    )


# ---------------------------------------------------------------------------
# Public snapshot functions
# ---------------------------------------------------------------------------


def snapshot_account_pnl(supervisor: IbkrSupervisor) -> pa.Table:
    """Snapshot the live ``accounts_pnl`` table to a canonical ``ACCOUNT_PNL`` Arrow table.

    ``accounts_pnl`` is auto-populated on connect (no explicit subscription needed).

    Args:
        supervisor: A connected IbkrSupervisor.

    Returns:
        A canonical ``ACCOUNT_PNL`` Arrow table.
    """
    rows = supervisor.snapshot_raw_rows("accounts_pnl")
    return _account_pnl_arrow_from_rows(rows)


def snapshot_position_pnl(
    supervisor: IbkrSupervisor,
    conid_to_sym: dict[int, str],
) -> pa.Table:
    """Snapshot the live ``accounts_pnl_single`` table to a canonical ``POSITION_PNL`` Arrow table.

    ``accounts_pnl_single`` is empty until ``request_single_pnl_for_conids`` has been
    called for at least one contract.

    Args:
        supervisor: A connected IbkrSupervisor.
        conid_to_sym: Mapping from ConId (int) to Sym (str) for symbol resolution.

    Returns:
        A canonical ``POSITION_PNL`` Arrow table (zero rows if no subscriptions have been made).
    """
    rows = supervisor.snapshot_raw_rows("accounts_pnl_single")
    return _position_pnl_arrow_from_rows(rows, conid_to_sym)


# ---------------------------------------------------------------------------
# Bounded subscription entry point
# ---------------------------------------------------------------------------


def request_single_pnl_for_conids(
    supervisor: IbkrSupervisor,
    account: str,
    conids: list[int],
    model_code: str = "",
) -> None:
    """Issue ``reqPnLSingle`` for each conid in the bounded list.

    The caller is responsible for keeping *conids* small (e.g. top-N by
    |MarketValue|).  Subscribing every position of a large book (2,000+ positions)
    at once is a known failure mode (280k log lines, stalls the decode thread).

    Resolves each conid to a ``RegisteredContract`` via ``reqContractDetails``
    (one blocking round-trip per conid), then calls ``session.request_single_pnl``.
    Individual failures are logged at WARNING and skipped; the function does not
    raise.

    Args:
        supervisor: A connected IbkrSupervisor.
        account: The IBKR account id (e.g. ``"U0000000"``).
        conids: Bounded list of contract IDs to subscribe.
        model_code: Optional model portfolio code (default ``""``).
    """
    session = supervisor._session
    succeeded = 0
    failed = 0

    for con_id in conids:
        try:
            c = build_contract_from_conid(con_id)
            rc = session.get_registered_contract(c)
            session.request_single_pnl(rc, account, model_code)
            succeeded += 1
            logger.debug("pnl: subscribed reqPnLSingle for conId=%d", con_id)
        except Exception as exc:  # noqa: BLE001 — best-effort; log, don't crash
            failed += 1
            logger.warning(
                "pnl: request_single_pnl failed for conId=%d: %s", con_id, exc
            )

    logger.info(
        "pnl: request_single_pnl_for_conids: %d succeeded, %d failed (of %d requested)",
        succeeded,
        failed,
        len(conids),
    )
