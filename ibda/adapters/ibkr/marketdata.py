"""ibda.adapters.ibkr.marketdata — paced market-data subscription helpers.

VENDOR-CONFINED: this module calls into deephaven-ib's raw TWS client and
imports ibapi.contract.Contract.  It MUST NOT import deephaven (engine).

Three public subscription functions:

``subscribe_quotes(supervisor, contract_ids, governor)``
    Request L1 bid/ask ticks (populates ticks_bid_ask) for each contract_id,
    paced by the PacingGovernor.  Uses DELAYED_FROZEN (type 4) — paper-safe.

``subscribe_bars(supervisor, contract_ids, governor)``
    Request real-time 5-second OHLCV bars (populates bars_realtime) for each
    contract_id via session.request_bars_realtime(), paced by the governor.

``subscribe_news(supervisor, contract_ids, governor, ...)``
    Request historical news headlines (populates news_historical, and via it
    the canonical ``news`` table) for each contract_id via
    session.request_news_historical(), paced by the governor.

All three are best-effort: entitlement errors (code 10089) and "Max
tickers reached" (code 100) are logged at WARNING, not raised.  They return
once all paced requests have been issued (not when data arrives).

Notes
-----
- reqMarketDataType(4) = DELAYED_FROZEN.  The deephaven-ib MarketDataType enum
  only defines REAL_TIME/FROZEN/DELAYED (1-3); we pass int 4 directly to the
  raw client — the same approach used consistently by every other market-data
  subscriber in this project.
- session._client.reqMktData(reqId, contract, genericTicks, snapshot, regulatory,
  mktDataOptions) — called directly because deephaven-ib's high-level API does
  not expose a pacing-friendly per-contract call.
- session.request_bars_realtime(registered_contract, bar_type, bar_size=5) is
  the high-level deephaven-ib API for bars.
"""
from __future__ import annotations

import logging
import time as _time
from typing import Any, cast

import pyarrow as pa
from ibapi.contract import Contract

from ibda.adapters.ibkr.contracts import (
    build_contract_from_conid,
    build_stock_contract,
    _conid_of,
)
from ibda.adapters.ibkr.pacing import PacingGovernor
from ibda.adapters.ibkr.supervisor import IbkrSupervisor
from ibda.schema import BAR

logger = logging.getLogger(__name__)

# IBKR TWS market-data type 4 = DELAYED_FROZEN.
_DELAYED_FROZEN_TYPE: int = 4
# Generic tick list: empty string = price/size only (sufficient for bid/ask).
_GENERIC_TICKS: str = ""
# reqId base for our market-data requests (avoids collision with other users).
_QUOTE_REQ_ID_BASE: int = 300_000


def _set_delayed_frozen(supervisor: IbkrSupervisor) -> None:
    """Set TWS market data type to DELAYED_FROZEN (type 4) on the raw client."""
    try:
        supervisor.raw_table  # noqa: B018 — just verifying session is up
        session = supervisor._session
        session._client.reqMarketDataType(_DELAYED_FROZEN_TYPE)
        logger.info("marketdata: set market data type to DELAYED_FROZEN (%d)", _DELAYED_FROZEN_TYPE)
    except Exception as exc:  # noqa: BLE001 — best-effort
        logger.warning("marketdata: failed to set market data type: %s", exc)


def subscribe_quotes(
    supervisor: IbkrSupervisor,
    contract_ids: list[int],
    governor: PacingGovernor,
) -> None:
    """Request L1 bid/ask tick subscriptions (ticks_bid_ask) for each contract.

    Parameters
    ----------
    supervisor:
        A connected IbkrSupervisor.
    contract_ids:
        List of IBKR contract IDs to subscribe.
    governor:
        PacingGovernor controlling the request rate.
    """
    _set_delayed_frozen(supervisor)
    session = supervisor._session

    for i, con_id in enumerate(contract_ids):
        req_id = _QUOTE_REQ_ID_BASE + i
        c = build_contract_from_conid(con_id)

        wait = governor.wait_for(_time.monotonic())
        if wait > 0.0:
            _time.sleep(wait)

        try:
            session._client.reqMktData(req_id, c, _GENERIC_TICKS, False, False, [])
            logger.debug(
                "marketdata: subscribed quotes reqId=%d conId=%d", req_id, con_id
            )
        except Exception as exc:  # noqa: BLE001 — best-effort; log, don't crash
            logger.warning(
                "marketdata: reqMktData failed for conId=%d: %s", con_id, exc
            )


def subscribe_bars(
    supervisor: IbkrSupervisor,
    contract_ids: list[int],
    governor: PacingGovernor,
) -> None:
    """Request real-time 5-second bar subscriptions (bars_realtime) for each contract.

    Uses deephaven-ib's high-level ``request_bars_realtime`` which registers the
    contract and issues reqRealTimeBars via the rate-limiter.

    Parameters
    ----------
    supervisor:
        A connected IbkrSupervisor.
    contract_ids:
        List of IBKR contract IDs to subscribe.
    governor:
        PacingGovernor controlling the request rate.
    """
    import deephaven_ib as dhib  # noqa: PLC0415 — vendor import, lazy

    session = supervisor._session

    for con_id in contract_ids:
        wait = governor.wait_for(_time.monotonic())
        if wait > 0.0:
            _time.sleep(wait)

        try:
            c = build_contract_from_conid(con_id)
            rc = session.get_registered_contract(c)
            session.request_bars_realtime(
                rc,
                bar_type=dhib.BarDataType.TRADES,
                bar_size=5,
                market_data_type=dhib.MarketDataType.FROZEN,
            )
            logger.debug("marketdata: subscribed bars conId=%d", con_id)
        except Exception as exc:  # noqa: BLE001 — best-effort; log, don't crash
            logger.warning(
                "marketdata: request_bars_realtime failed for conId=%d: %s",
                con_id, exc,
            )


# ---------------------------------------------------------------------------
# On-demand snapshot quote (one-shot, by symbol)
# ---------------------------------------------------------------------------

# Accept both real-time and DELAYED_* tick-type strings (DELAYED_FROZEN emits
# the DELAYED_* variants).
_LAST_TYPES: frozenset[str] = frozenset({"LAST", "DELAYED_LAST"})
_BID_TYPES: frozenset[str] = frozenset({"BID", "DELAYED_BID"})
_ASK_TYPES: frozenset[str] = frozenset({"ASK", "DELAYED_ASK"})
_BID_SIZE_TYPES: frozenset[str] = frozenset({"BID_SIZE", "DELAYED_BID_SIZE"})
_ASK_SIZE_TYPES: frozenset[str] = frozenset({"ASK_SIZE", "DELAYED_ASK_SIZE"})
_LAST_SIZE_TYPES: frozenset[str] = frozenset({"LAST_SIZE", "DELAYED_LAST_SIZE"})


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _pick_price(rows: list[dict[str, Any]], types: frozenset[str]) -> float | None:
    """Return the latest ``Price`` among *rows* whose ``TickType`` is in *types*."""
    clean = [
        r["Price"] for r in rows
        if str(r.get("TickType", "")) in types and r.get("Price") is not None
    ]
    return float(clean[-1]) if clean else None


def _pick_size(rows: list[dict[str, Any]], types: frozenset[str]) -> float | None:
    """Return the latest ``Size`` among *rows* whose ``TickType`` is in *types*."""
    clean = [
        r["Size"] for r in rows
        if str(r.get("TickType", "")) in types and r.get("Size") is not None
    ]
    return float(clean[-1]) if clean else None


def _latest_ts(rows: list[dict[str, Any]]) -> str | None:
    ts = rows[-1].get("ReceiveTime") if rows else None
    return str(ts) if ts is not None else None


def subscribe_news(
    supervisor: IbkrSupervisor,
    contract_ids: list[int],
    governor: PacingGovernor,
    *,
    start: str | None = None,
    end: str | None = None,
    total_results: int = 100,
) -> None:
    """Request historical news headlines (news_historical) for each contract.

    Uses deephaven-ib's high-level ``request_news_historical``, which registers
    the contract and issues ``reqHistoricalNews`` via the rate-limiter — the
    same registration path ``subscribe_bars`` uses.  ``news_historical`` (and
    therefore the canonical ``news`` table) is empty until this is called for
    at least one contract.

    Parameters
    ----------
    supervisor:
        A connected IbkrSupervisor.
    contract_ids:
        List of IBKR contract IDs to request headlines for.
    governor:
        PacingGovernor controlling the request rate.
    start, end:
        Optional date-range bounds passed through to ``request_news_historical``
        (``None`` for each means "no bound" — provider defaults apply).
    total_results:
        Maximum headlines to fetch per contract (IB caps this at 300).
    """
    session = supervisor._session

    for con_id in contract_ids:
        wait = governor.wait_for(_time.monotonic())
        if wait > 0.0:
            _time.sleep(wait)

        try:
            c = build_contract_from_conid(con_id)
            rc = session.get_registered_contract(c)
            session.request_news_historical(
                rc, start=start, end=end, total_results=total_results
            )
            logger.debug("marketdata: subscribed news conId=%d", con_id)
        except Exception as exc:  # noqa: BLE001 — best-effort; log, don't crash
            logger.warning(
                "marketdata: request_news_historical failed for conId=%d: %s",
                con_id, exc,
            )


def snapshot_quote(
    supervisor: IbkrSupervisor,
    symbol: str,
    *,
    exchange: str | None = None,
    timeout_s: float = 6.0,
    poll_interval_s: float = 0.25,
    governor: PacingGovernor | None = None,
    conid_hint: int | None = None,
) -> dict[str, Any]:
    """Resolve *symbol* and return a one-shot market-data quote via TWS.

    Resolves the symbol as a US stock on SMART (blocking ``get_registered_contract``;
    raises → ``symbol_not_found``), issues a single DELAYED_FROZEN snapshot
    (paper-safe, auto-cancels), then polls ``ticks_price``/``ticks_size``
    (filtered by contract id) until a tick arrives or *timeout_s* elapses.

    Parameters
    ----------
    supervisor:
        A connected IbkrSupervisor.
    symbol:
        Ticker symbol (e.g. ``"AAPL"``).
    exchange:
        Optional primary exchange to disambiguate (e.g. ``"NASDAQ"``); routing
        stays SMART.
    timeout_s, poll_interval_s:
        Bounded-wait controls.
    governor:
        Optional PacingGovernor (defaults to 5 Hz) — a snapshot is one request.
    conid_hint:
        Optional IBKR contract id for *symbol*.  When provided, the contract is
        built directly from the conId (``exchange="SMART"``), mirroring the
        pattern ``subscribe_quotes`` uses, and ``get_registered_contract`` is
        **skipped entirely** — eliminating the up-to-120 s blocking resolution
        call on large books.  The market data request goes directly via the raw
        TWS client (``reqMktData``), same as the persistent-subscription path.
        This is a latency mitigation for symbols already held in the portfolio
        (where the conId is known from canonical data); it does NOT change the
        120 s ceiling for genuinely unresolved symbols — only those that need
        name→conId resolution still hit that limit.  When omitted, the original
        full-resolution path is used unchanged.

    Returns
    -------
    dict
        On success: ``{"symbol", "conId", "last", "bid", "ask", "bid_size",
        "ask_size", "last_size", "timestamp"}`` (prices/sizes may be ``None``).
        On failure (never raises for these): ``{"error": "symbol_not_found"|
        "request_failed"|"quote_timeout", "symbol", ...}``.
    """
    sym = symbol.strip().upper()
    if not sym:
        return {"error": "symbol_not_found", "symbol": symbol, "message": "empty symbol"}

    session = supervisor._session

    # 1. Resolve the contract and issue the market-data request.
    #
    #    Fast path (conid_hint is set):
    #      Build the contract from the known conId — identical to subscribe_quotes.
    #      get_registered_contract is NOT called.  The snapshot request goes directly
    #      via session._client.reqMktData, bypassing the RegisteredContract machinery
    #      that requires a resolved contract.
    #
    #    Slow path (conid_hint is None):
    #      Symbol-based resolution via get_registered_contract (blocks up to 120 s
    #      on a busy large-book session).  Unchanged from the original behaviour.
    _set_delayed_frozen(supervisor)
    gov = governor if governor is not None else PacingGovernor(max_hz=5.0)
    wait = gov.wait_for(_time.monotonic())
    if wait > 0.0:
        _time.sleep(wait)

    if conid_hint is not None:
        c = build_contract_from_conid(conid_hint)
        conid: int | None = conid_hint
        # Issue directly via raw client — same shortcut subscribe_quotes uses.
        try:
            session._client.reqMktData(
                _QUOTE_REQ_ID_BASE, c, _GENERIC_TICKS, True, False, []
            )
        except Exception as exc:  # noqa: BLE001 — best-effort; return structured
            return {"error": "request_failed", "symbol": sym, "conId": conid, "message": str(exc)}
    else:
        c = build_stock_contract(sym, primary_exchange=exchange)
        try:
            rc = session.get_registered_contract(c)
        except Exception as exc:  # noqa: BLE001 — return structured, never raise into tools
            return {"error": "symbol_not_found", "symbol": sym, "message": str(exc)}
        conid = _conid_of(rc)
        if conid is None:
            return {"error": "symbol_not_found", "symbol": sym,
                    "message": "contract resolved but no contract id available"}
        try:
            session.request_market_data(rc, snapshot=True)
        except Exception as exc:  # noqa: BLE001 — best-effort; return structured
            return {"error": "request_failed", "symbol": sym, "conId": conid, "message": str(exc)}

    # 3. Bounded poll of the tick tables, filtered to our contract id.
    #    snapshot_raw_rows_where filters in the engine before materialization
    #    (O(matching) vs O(total) — critical for large append-only tables).
    deadline = _time.monotonic() + timeout_s
    price_rows: list[dict[str, Any]] = []
    while _time.monotonic() < deadline:
        price_rows = supervisor.snapshot_raw_rows_where(
            "ticks_price", f"ContractId == {conid}"
        )
        if (
            _pick_price(price_rows, _LAST_TYPES) is not None
            or _pick_price(price_rows, _BID_TYPES) is not None
            or _pick_price(price_rows, _ASK_TYPES) is not None
        ):
            break
        _time.sleep(poll_interval_s)

    if not price_rows:
        return {
            "error": "quote_timeout", "symbol": sym, "conId": conid,
            "message": f"no market data within {timeout_s:.0f}s (market closed or not subscribed)",
        }

    size_rows = supervisor.snapshot_raw_rows_where(
        "ticks_size", f"ContractId == {conid}"
    )
    return {
        "symbol": sym,
        "conId": conid,
        "last": _pick_price(price_rows, _LAST_TYPES),
        "bid": _pick_price(price_rows, _BID_TYPES),
        "ask": _pick_price(price_rows, _ASK_TYPES),
        "bid_size": _pick_size(size_rows, _BID_SIZE_TYPES),
        "ask_size": _pick_size(size_rows, _ASK_SIZE_TYPES),
        "last_size": _pick_size(size_rows, _LAST_SIZE_TYPES),
        "timestamp": _latest_ts(price_rows),
    }


def _trade_row(r: dict[str, Any]) -> dict[str, Any]:
    price = r.get("Price")
    size = r.get("Size")
    ts = r.get("Timestamp")
    return {
        "timestamp": str(ts) if ts is not None else None,
        "price": float(price) if price is not None else None,
        "size": float(size) if size is not None else None,
    }


def snapshot_trades(
    supervisor: IbkrSupervisor,
    symbol: str,
    *,
    count: int = 30,
    exchange: str | None = None,
    timeout_s: float = 6.0,
    poll_interval_s: float = 0.25,
    governor: PacingGovernor | None = None,
    conid_hint: int | None = None,
) -> dict[str, Any]:
    """Resolve *symbol* and return the most recent *count* last-trade prints via TWS.

    Resolves the symbol as a US stock on SMART (blocking ``get_registered_contract``;
    raises → ``symbol_not_found``), requests historical tick-by-tick LAST data
    (``request_tick_data_historical``, populates ``ticks_trade``), then polls
    ``ticks_trade`` (filtered by contract id) until ticks arrive or *timeout_s*
    elapses.

    Parameters
    ----------
    supervisor:
        A connected IbkrSupervisor.
    symbol:
        Ticker symbol (e.g. ``"AAPL"``).
    count:
        Number of most-recent trade prints to return.
    exchange:
        Optional primary exchange to disambiguate.
    timeout_s, poll_interval_s:
        Bounded-wait controls.
    governor:
        Optional PacingGovernor (defaults to 5 Hz).
    conid_hint:
        Optional IBKR contract id for *symbol*.  When provided,
        ``get_registered_contract`` is **skipped entirely** and the historical
        tick request goes via the raw TWS client (``reqHistoricalTicks``),
        using the conId directly — eliminating the up-to-120 s blocking
        resolution call on large books.  This is a latency mitigation for
        symbols already held in the portfolio; it does NOT change the 120 s
        ceiling for genuinely unresolved symbols.  When omitted, the original
        full-resolution path is used unchanged.

    Returns
    -------
    dict
        On success: ``{"symbol", "conId", "count", "trades": [{"timestamp",
        "price", "size"}, ...]}``.  On failure (never raises for these):
        ``{"error": "symbol_not_found"|"request_failed"|"trade_timeout", ...}``.
    """
    sym = symbol.strip().upper()
    if not sym:
        return {"error": "symbol_not_found", "symbol": symbol, "message": "empty symbol"}
    n = count if count > 0 else 30

    session = supervisor._session

    _set_delayed_frozen(supervisor)
    gov = governor if governor is not None else PacingGovernor(max_hz=5.0)
    wait = gov.wait_for(_time.monotonic())
    if wait > 0.0:
        _time.sleep(wait)

    if conid_hint is not None:
        c = build_contract_from_conid(conid_hint)
        conid: int | None = conid_hint
        # Issue directly via raw client — skip get_registered_contract entirely.
        # reqHistoricalTicks signature: (reqId, contract, startDateTime, endDateTime,
        #   numberOfTicks, whatToShow, useRth, ignoreSize, miscOptions)
        try:
            session._client.reqHistoricalTicks(
                _QUOTE_REQ_ID_BASE, c, "", "", n, "TRADES", 0, True, []
            )
        except Exception as exc:  # noqa: BLE001 — best-effort; return structured
            return {"error": "request_failed", "symbol": sym, "conId": conid, "message": str(exc)}
    else:
        import deephaven_ib as dhib  # noqa: PLC0415 — vendor import, lazy

        c = build_stock_contract(sym, primary_exchange=exchange)
        try:
            rc = session.get_registered_contract(c)
        except Exception as exc:  # noqa: BLE001 — return structured, never raise into tools
            return {"error": "symbol_not_found", "symbol": sym, "message": str(exc)}
        conid = _conid_of(rc)
        if conid is None:
            return {"error": "symbol_not_found", "symbol": sym,
                    "message": "contract resolved but no contract id available"}
        try:
            session.request_tick_data_historical(rc, dhib.TickDataType.LAST, number_of_ticks=n)
        except Exception as exc:  # noqa: BLE001 — best-effort; return structured
            return {"error": "request_failed", "symbol": sym, "conId": conid, "message": str(exc)}

    deadline = _time.monotonic() + timeout_s
    rows: list[dict[str, Any]] = []
    while _time.monotonic() < deadline:
        rows = supervisor.snapshot_raw_rows_where(
            "ticks_trade", f"ContractId == {conid}"
        )
        if rows:
            break
        _time.sleep(poll_interval_s)

    if not rows:
        return {
            "error": "trade_timeout", "symbol": sym, "conId": conid,
            "message": f"no trade ticks within {timeout_s:.0f}s (market closed or not subscribed)",
        }

    trades = [_trade_row(r) for r in rows][-n:]
    return {"symbol": sym, "conId": conid, "count": len(trades), "trades": trades}


# ---------------------------------------------------------------------------
# On-demand historical daily bars (the sole benchmark source — IB-only, no
# offline/public fallback)
# ---------------------------------------------------------------------------


def historical_bars(
    supervisor: IbkrSupervisor,
    symbol: str,
    *,
    duration: str = "1 Y",
    bar_size: str = "1 day",
    exchange: str | None = None,
    contract: Contract | None = None,
    timeout_s: float = 15.0,
    poll_interval_s: float = 0.25,
    keep_up_to_date: bool = True,
    bar_type: Any | None = None,
) -> pa.Table:
    """Resolve *symbol* and return its historical daily bars as a canonical ``bar`` table.

    The sole benchmark source for :func:`ibda.analytics.benchmark.relative_summary` /
    ``rolling_relative`` — ``ibda`` has no offline/public benchmark fallback (no Yahoo, no
    other public feed); a symbol benchmark always resolves through this call. Wraps
    deephaven-ib's ``request_bars_historical`` (TRADES), then polls ``bars_historical``
    filtered by THIS call's own ``RequestId`` (see the RequestId-scoping note below).
    **Entitlement-gated:** on a non-entitled account (e.g. a paper account without the
    relevant market-data subscription) IBKR returns no data (errors 10167/10091); this
    returns an EMPTY ``bar`` table on timeout rather than raising, mirroring the
    best-effort tone of this module. Returns whatever bars arrived.

    Parameters
    ----------
    duration:
        IB duration string, e.g. ``"1 Y"``, ``"6 M"``.  Passed verbatim to
        ``dhib.Duration(duration)``.
    bar_size:
        IB bar-size string.  Must be a value accepted by ``dhib.BarSize``, e.g.
        ``"1 day"``, ``"1 hour"``, ``"1 min"``.  An unrecognised string causes a
        ``ValueError`` from the enum reverse-lookup; it is caught by the best-effort
        handler and an empty bar table is returned.
    bar_type:
        Optional ``deephaven_ib.BarDataType``.  Defaults to ``TRADES``, which is correct
        for equities but **not** for every instrument: IB serves no trade prints for an FX
        ``CASH`` contract, so a TRADES request there returns nothing at all.  Callers
        passing a non-equity ``contract`` should pass a matching type (``MIDPOINT`` for FX).
        Kept as ``None``-defaulted rather than a literal default so the enum is not imported
        at signature-evaluation time.
    contract:
        Optional pre-built ``Contract`` to resolve instead of a US-stock lookup built
        from *symbol* (e.g. an FX ``CASH``/IDEALPRO contract from
        :func:`ibda.adapters.ibkr.contracts.build_fx_contract`, used by callers that
        need an FX prior-close reference rate). *symbol* still labels the ``Sym``
        column of the returned bar table. Defaults to ``build_stock_contract(symbol)``.

    RequestId scoping (a prior regression, fixed here)
    ----------------------------------------------------
    The poll previously filtered ``bars_historical`` by ``ContractId`` ONLY. That table is
    shared by every ``request_bars_historical`` call for the life of the session — so
    requesting, say, ``"5 mins"`` bars for a contract whose ``"1 day"`` bars were already
    pulled earlier in the session returned the DAILY and 5-MINUTE candles UNIONED into one
    series (every daily bar duplicated, betas computed off a garbage mixed-granularity
    series — the exact live defect that motivated this fix).
    ``bars_historical`` carries a ``RequestId``
    column precisely so concurrent/successive requests on the same contract stay
    distinguishable; this function now captures the ``Request`` returned by
    ``request_bars_historical`` and polls filtered to ``RequestId == <this call's id>``
    (additionally ANDed with ``ContractId`` when the conId is known, as a belt-and-suspenders
    cross-check — never the sole filter). If ``request_bars_historical`` itself failed (the
    existing best-effort ``except`` below) or returned no ``Request`` objects, this falls
    back to the pre-fix ``ContractId``-only filter (or an unfiltered full snapshot if even
    the conId is unknown) — the same-symbol, same-bar-size, single-outstanding-request case
    this module was originally written for is unaffected either way.
    keep_up_to_date:
        Forwarded verbatim to ``session.request_bars_historical``. **Default ``True``,
        preserving this function's long-standing (previously implicit-by-omission)
        behavior** — every call opens a persistent, continuously-updating IB historical
        subscription against the ~50-simultaneous-historical cap. Callers that only want a
        one-shot snapshot (e.g. a memoized once-per-process fetch) should pass ``False``
        explicitly.

        **Measured consequence of ``False`` (paper account, market open — this
        docstring is the full trial record):** ``False`` still
        populates ``bars_historical`` with the SAME rows as ``True``, including
        TODAY'S still-forming bar (both delivered the identical last row — the same
        local trading date, arriving as ``03:59:59+00:00`` the following day on the
        wire, across 3 symbols). The forming bar is NOT absent under ``False`` — it is present but
        FROZEN at whatever value IB returned in that one initial delivery: ``True``
        additionally keeps that row refreshing as the session progresses (per
        deephaven-ib's own docstring, ``keep_up_to_date: True to continuously update
        bars``); ``False`` delivers it once and the subscription then closes, so a
        caller that fetches once per process (the intended, memoized usage — see
        ``historical_bars_fetch_batch``) will see today's bar/price STALE (frozen at
        fetch time) for the rest of the process's life, not missing.
    """
    import deephaven_ib as dhib  # noqa: PLC0415 — vendor import, lazy

    sym = symbol.strip().upper()
    session = supervisor._session

    _set_delayed_frozen(supervisor)

    c = contract if contract is not None else build_stock_contract(sym, primary_exchange=exchange)
    rc: object = session.get_registered_contract(c)
    conid: int | None = _conid_of(rc)
    request_id: int | None = None
    try:
        dur_obj = dhib.Duration(duration)
        bar_size_obj = dhib.BarSize(bar_size)
        requests = session.request_bars_historical(
            rc, duration=dur_obj, bar_size=bar_size_obj,
            bar_type=bar_type if bar_type is not None else dhib.BarDataType.TRADES,
            market_data_type=dhib.MarketDataType.FROZEN,
            keep_up_to_date=keep_up_to_date,
        )
        if requests:
            request_id = int(requests[0].request_id)
    except Exception as exc:  # noqa: BLE001 — best-effort; return empty on failure
        logger.warning("marketdata: request_bars_historical failed for %s: %s", sym, exc)

    deadline = _time.monotonic() + timeout_s
    rows: list[dict[str, Any]] = []
    while _time.monotonic() < deadline:
        # Prefer RequestId scoping (see docstring — the fix for the regression above): a request only ever
        # reads back its own rows, immune to other bar-size/duration requests already
        # outstanding on the same ContractId. ContractId is ANDed in too when known, as
        # a cheap cross-check, never the sole filter once a request_id is available.
        if request_id is not None:
            predicate = f"RequestId == {request_id}"
            if conid is not None:
                predicate += f" && ContractId == {conid}"
            rows = supervisor.snapshot_raw_rows_where("bars_historical", predicate)
        elif conid is not None:
            # request_bars_historical failed/returned no Request — fall back to the
            # pre-fix ContractId-only filter (rare: only on a failed request).
            rows = supervisor.snapshot_raw_rows_where(
                "bars_historical", f"ContractId == {conid}"
            )
        else:
            rows = supervisor.snapshot_raw_rows("bars_historical")
        if rows:
            break
        _time.sleep(poll_interval_s)

    return _rows_to_bar_table(sym, rows)


def _opt_f(row: dict[str, Any], key: str) -> float | None:
    v = row.get(key)
    return float(v) if v is not None else None


def _rows_to_bar_table(sym: str, rows: list[dict[str, Any]]) -> pa.Table:
    """Project raw ``bars_historical`` *rows* into a canonical ``BAR`` Arrow table.

    Shared by ``historical_bars`` (daily, filtered by ContractId) and
    ``read_intraday_bars`` (intraday, filtered by RequestId).  Reads
    Timestamp + OHLCV per row, tagging every row with *sym*.  An empty *rows*
    yields a zero-row table with the canonical ``BAR`` schema.
    """
    recs = [{
        "Sym": sym, "Timestamp": r.get("Timestamp"),
        "Open": _opt_f(r, "Open"), "High": _opt_f(r, "High"),
        "Low": _opt_f(r, "Low"), "Close": _opt_f(r, "Close"),
        "Volume": _opt_f(r, "Volume"),
    } for r in rows]
    columns: list[pa.Array] = []
    for col in BAR.columns:
        columns.append(pa.array([rec.get(col.name) for rec in recs], type=col.dtype.to_arrow()))
    return cast(pa.Table, pa.table(dict(zip(BAR.column_names, columns, strict=True)),
                                   schema=BAR.to_arrow_schema()))


# ---------------------------------------------------------------------------
# Intraday per-cycle bars (strategy-engine bar feed surface)
# ---------------------------------------------------------------------------

# Map a human bar-size string → the deephaven-ib BarSize enum member name.
# The enum itself is resolved lazily inside request_intraday_bars (vendor
# import); we keep only the attribute name here so this dict needs no JVM.
_BAR_SIZE_ATTRS: dict[str, str] = {
    "1 min": "MIN_1",
}


def request_intraday_bars(
    supervisor: IbkrSupervisor,
    symbols: list[str],
    *,
    duration_secs: int,
    bar_size: str = "1 min",
    reg_cache: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Register (cached) + issue request_bars_historical per symbol; return {symbol: request_id}.

    The intraday per-cycle counterpart to ``historical_bars`` (which defaults
    ``keep_up_to_date=True`` — a persistent, continuously-updating subscription,
    not a one-shot fetch — and polls filtered by ``RequestId``, ANDed with
    ``ContractId`` only as a belt-and-suspenders cross-check; see that
    function's own docstring for the todo #31 fix).  This issues a
    ``request_bars_historical`` per symbol covering the last *duration_secs*
    seconds and returns each symbol's integer ``request_id`` so the caller can
    later read the shared ``bars_historical`` table filtered **by RequestId**
    (matching the strategy engine's bar contract).

    For each symbol: build an STK/SMART/USD contract, register it via
    ``get_registered_contract``; a multi-contract (``rc.is_multi()``) is
    skipped (no entry in the result).  If *reg_cache* is supplied, registered
    contracts are reused/populated keyed by symbol so contracts are not
    re-registered across polling cycles.

    Best-effort per symbol: on any exception for a symbol, a warning is logged
    and the symbol is skipped (no entry), consistent with this module's tone.

    Parameters
    ----------
    supervisor:
        A connected IbkrSupervisor (e.g. wrapped via ``from_session``).
    symbols:
        Ticker symbols to request (e.g. ``["AAPL", "MSFT"]``).
    duration_secs:
        Lookback window in seconds (``Duration.seconds(duration_secs)``).
    bar_size:
        Bar-size string; currently only ``"1 min"`` is supported.
    reg_cache:
        Optional per-symbol registered-contract cache, reused across cycles.

    Returns
    -------
    dict[str, int]
        ``{symbol: request_id}`` for the symbols that issued cleanly.
    """
    import deephaven_ib as dhib  # noqa: PLC0415 — vendor import, lazy

    attr = _BAR_SIZE_ATTRS.get(bar_size)
    if attr is None:
        raise ValueError(
            f"unsupported bar_size {bar_size!r}; supported: {sorted(_BAR_SIZE_ATTRS)}"
        )
    dh_bar_size: Any = getattr(dhib.BarSize, attr)

    session = supervisor._session
    duration: Any = dhib.Duration.seconds(duration_secs)

    req_ids: dict[str, int] = {}
    for symbol in symbols:
        sym = symbol.strip().upper()
        try:
            rc: Any = reg_cache.get(sym) if reg_cache is not None else None
            if rc is None:
                c = build_stock_contract(sym)
                rc = session.get_registered_contract(c)
                if rc.is_multi():
                    logger.warning("marketdata: skipping multi-contract symbol %s", sym)
                    continue
                if reg_cache is not None:
                    reg_cache[sym] = rc

            requests: Any = session.request_bars_historical(
                rc,
                duration=duration,
                bar_size=dh_bar_size,
                bar_type=dhib.BarDataType.TRADES,
                end=None,
                keep_up_to_date=False,
            )
            req_ids[sym] = int(requests[0].request_id)
        except Exception as exc:  # noqa: BLE001 — best-effort per symbol; log + skip
            logger.warning(
                "marketdata: request_intraday_bars failed for %s: %s", sym, exc
            )

    return req_ids


def read_intraday_bars(
    supervisor: IbkrSupervisor,
    request_ids: dict[str, int],
) -> dict[str, pa.Table]:
    """Read bars_historical filtered to tracked RequestIds → canonical BAR Arrow tables.

    Issues a **single** engine-side ``where`` filter covering all tracked
    ``RequestId`` values so only matching rows are materialised — O(matching)
    rather than O(total) per call.  This is critical because ``bars_historical``
    is append-only and grows unboundedly over a hot session; the previous
    full-snapshot path re-materialised every accumulated row on every poll cycle.

    Returns an empty dict immediately when *request_ids* is empty (no snapshot
    is issued).

    Parameters
    ----------
    supervisor:
        A connected IbkrSupervisor.
    request_ids:
        ``{symbol: request_id}`` as returned by ``request_intraday_bars``.

    Returns
    -------
    dict[str, pa.Table]
        ``{symbol: bar_table}`` for every symbol in *request_ids*.  A symbol
        with no matching rows yields an empty (zero-row) ``BAR`` table.
    """
    out: dict[str, pa.Table] = {}
    if not request_ids:
        return out

    # Build one engine-side predicate covering ALL tracked RequestIds before
    # materialisation.  Single-id fast path avoids the (…) || (…) wrapper for
    # the common case; multi-id path OR-chains each clause.  The || format
    # matches the existing == predicates used throughout this module and has a
    # verified execution path through snapshot_raw_rows_where.
    req_id_values = list(request_ids.values())
    if len(req_id_values) == 1:
        predicate = f"RequestId == {req_id_values[0]}"
    else:
        predicate = " || ".join(f"(RequestId == {r})" for r in req_id_values)

    try:
        rows = supervisor.snapshot_raw_rows_where("bars_historical", predicate)
    except Exception as exc:  # noqa: BLE001 — best-effort; empty tables on read failure
        logger.warning("marketdata: cannot read bars_historical: %s", exc)
        rows = []

    for sym, req_id in request_ids.items():
        matched = [r for r in rows if _as_int(r.get("RequestId")) == req_id]
        out[sym] = _rows_to_bar_table(sym, matched)
    return out


# ---------------------------------------------------------------------------
# Live-ticking intraday bars (chart surface for a downstream consumer — RequestId only, no poll)
# ---------------------------------------------------------------------------


def request_intraday_bars_ticking(
    supervisor: IbkrSupervisor,
    symbol: str,
    *,
    duration: str = "1 D",
    bar_size: str = "5 mins",
    exchange: str | None = None,
) -> int | None:
    """Issue ONE ``request_bars_historical(..., keep_up_to_date=True)`` for *symbol*
    and return its ``RequestId`` — the scoping key a caller needs to read the LIVE,
    continuously-ticking ``bars_historical`` rows for exactly this request.

    Unlike ``historical_bars`` (which polls ``bars_historical`` and returns a static
    pyarrow snapshot), this function does not poll and returns no table: with
    ``keep_up_to_date=True`` (the deephaven-ib default), IBKR backfills today's bars
    and then keeps appending/re-emitting the forming bar for the rest of the session —
    a caller that wants a chart which updates as the day goes on needs the native,
    ticking Deephaven ``Table`` (``supervisor.raw_table("bars_historical")``), not a
    point-in-time pyarrow copy. Building that filtered/deduped/sorted ticking Table
    requires the Deephaven engine, which this module — VENDOR-CONFINED, see the module
    docstring — must not import; the caller does that step, filtering to
    ``RequestId == <this return value>``.

    RequestId scoping is not optional here: without it, this request's 5-minute bars
    would union with any daily/other-bar-size request already outstanding for the same
    contract (the exact defect fixed in ``historical_bars`` — see its docstring above).
    ``RequestId`` is the only column that disambiguates concurrent requests
    on one ContractId; two different ``bar_size``s for the same contract get two
    different ``RequestId``s deterministically (verified via the deephaven-ib
    ``RequestIdManager``, which hands out a strictly increasing id per call — see
    ``request_bars_historical``'s per-``contract_details`` loop in
    ``deephaven_ib/__init__.py``).

    Returns
    -------
    int | None
        The new request's ``RequestId``, or ``None`` if contract resolution or the
        ``reqHistoricalData`` call itself failed (best-effort — logged at WARNING,
        never raised, matching this module's other on-demand helpers) or the call
        returned no ``Request`` objects.
    """
    import deephaven_ib as dhib  # noqa: PLC0415 — vendor import, lazy

    sym = symbol.strip().upper()
    session = supervisor._session

    _set_delayed_frozen(supervisor)

    try:
        c = build_stock_contract(sym, primary_exchange=exchange)
        rc = session.get_registered_contract(c)
        dur_obj = dhib.Duration(duration)
        bar_size_obj = dhib.BarSize(bar_size)
        requests = session.request_bars_historical(
            rc, duration=dur_obj, bar_size=bar_size_obj,
            bar_type=dhib.BarDataType.TRADES,
            market_data_type=dhib.MarketDataType.FROZEN,
            keep_up_to_date=True,
        )
    except Exception as exc:  # noqa: BLE001 — best-effort; return None on failure
        logger.warning(
            "marketdata: request_intraday_bars_ticking failed for %s: %s", sym, exc
        )
        return None

    if not requests:
        return None
    return int(requests[0].request_id)


def cancel_historical_bars(supervisor: IbkrSupervisor, request_id: int) -> None:
    """Best-effort cancel of a ``keep_up_to_date`` historical-data line by *request_id*.

    ``request_intraday_bars_ticking`` opens its live-ticking line via deephaven-ib's
    ``session.request_bars_historical(..., keep_up_to_date=True)``, which internally
    builds its ``Request`` with ``cancel_func=None`` (unlike ``request_bars_realtime``,
    which does supply one) — so the returned ``Request.cancel()`` is a no-op that
    raises "not cancellable", and ``request_intraday_bars_ticking`` discards the
    ``Request`` object entirely, surfacing only the integer ``request_id``. There is
    therefore no deephaven-ib-level way to cancel this line.

    This function reaches past that gap to the raw ibapi client — deliberately, the
    same way this module already does elsewhere (e.g. ``_set_delayed_frozen``,
    ``subscribe_quotes``'s ``session._client.reqMktData``): ``supervisor._session._client``
    is deephaven-ib's ``IbTwsClient``, which extends ibapi's ``EClient`` and exposes
    ``cancelHistoricalData(reqId)`` directly. *request_id* must be the value returned by
    ``request_intraday_bars_ticking`` — that value **is** the ibapi ``reqId`` used in the
    underlying ``reqHistoricalData`` call (deephaven-ib's ``request_id_manager.next_id()``),
    so ``cancelHistoricalData(request_id)`` cancels exactly that line.

    Best-effort: never raises. Any failure (missing/``None`` session or client, not
    connected, ibapi itself raising) is caught and logged at WARNING; the function
    always returns ``None``.

    Upstream-proper alternative (future PR): give deephaven-ib's
    ``request_bars_historical`` a real ``cancel_func`` (mirroring
    ``request_bars_realtime``) so the ``Request`` it returns is genuinely
    cancellable via ``Request.cancel()`` — at which point this raw-client reach
    could be removed in favor of holding onto the ``Request`` object.

    Parameters
    ----------
    supervisor:
        A connected IbkrSupervisor.
    request_id:
        The ``RequestId`` returned by ``request_intraday_bars_ticking``.
    """
    try:
        session = supervisor._session
        client = session._client
        client.cancelHistoricalData(request_id)
        logger.debug(
            "marketdata: cancelled historical bars requestId=%d", request_id
        )
    except Exception as exc:  # noqa: BLE001 — best-effort; log, don't crash
        logger.warning(
            "marketdata: cancelHistoricalData failed for requestId=%d: %s",
            request_id, exc,
        )
