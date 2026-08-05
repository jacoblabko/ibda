"""Option-chain discovery + bounded per-contract Greeks over the IBKR data layer.

``option_chain`` is a ONE-SHOT, on-demand reference call: it issues a single
reqSecDefOptParams and polls the ``securities_def_option_params`` table (filtered by
its own reqId) with a timeout, mirroring marketdata.snapshot_quote. Nothing is
retained or subscribed. Greeks are subscribed one contract at a time. No deephaven
import here — the supervisor owns the engine.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import logging
import time as _time
from collections.abc import Sequence
from typing import Any

from ibda.adapters.ibkr.contracts import (
    _conid_of,
    build_option_contract,
    build_stock_contract,
    parse_expiry,
)

logger = logging.getLogger(__name__)

_CHAIN_TABLE = "securities_def_option_params"
_GREEKS_TABLE = "ticks_option_computation"


@dataclasses.dataclass(frozen=True)
class OptionParams:
    """Chain parameters for one exchange / trading class."""

    exchange: str
    trading_class: str
    multiplier: str
    expirations: list[_dt.date]
    strikes: list[float]


@dataclasses.dataclass(frozen=True)
class OptionChain:
    """Option chain for an underlying: a SMART convenience view + full per-exchange detail."""

    symbol: str
    underlying_con_id: int
    smart: OptionParams | None
    by_exchange: list[OptionParams]


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_expirations(raw: str) -> list[_dt.date]:
    out: list[_dt.date] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(parse_expiry(token))
        except ValueError:
            logger.debug("option_chain: skipping malformed expiration token %r", token)
            continue
    return sorted(out)


def _parse_strikes(raw: str) -> list[float]:
    out: list[float] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(float(token))
        except ValueError:
            logger.debug("option_chain: skipping malformed strike token %r", token)
            continue
    return sorted(out)


def option_chain(
    supervisor: Any,
    symbol: str,
    *,
    timeout_s: float = 8.0,
    poll_interval_s: float = 0.25,
) -> OptionChain | dict[str, Any]:
    """Discover an underlying's option chain (expirations/strikes per exchange).

    One-shot: resolve the underlying conId, issue a single reqSecDefOptParams, poll the
    chain table by reqId until the row set stabilises or ``timeout_s`` elapses, parse.

    Returns
    -------
    ``OptionChain``
        On success: fully parsed chain with ``by_exchange`` and ``smart`` populated.
    ``dict``
        On the three guarded failure modes: ``{"error": "symbol_not_found"|"request_failed"|
        "chain_timeout", "symbol": ..., "message": ...}``.  This mirrors the
        structured-error contract that ``snapshot_quote`` uses in marketdata.py —
        callers can distinguish unresolved symbols, request failures, and timeouts.
        Only ``get_registered_contract`` and ``request_sec_def_opt_params`` are
        wrapped in try/except; the poll loop's ``snapshot_raw_rows_where`` call is
        unguarded, so a non-``is_failed`` exception from the engine still
        propagates to the caller — this function does not, in fact, never raise.

    Parameters
    ----------
    supervisor:
        A connected IbkrSupervisor.
    symbol:
        Underlying ticker (e.g. "AAPL").
    timeout_s:
        Maximum seconds to wait for the chain table to stabilise.
    poll_interval_s:
        Polling interval in seconds.
    """
    sym = symbol.strip().upper()
    session = supervisor._session

    contract = build_stock_contract(sym)
    try:
        rc = session.get_registered_contract(contract)
    except Exception as exc:  # noqa: BLE001 — never raise into callers
        logger.info("option_chain: symbol %s not resolved: %s", sym, exc)
        return {
            "error": "symbol_not_found",
            "symbol": sym,
            "message": str(exc),
        }

    conid = _conid_of(rc)
    if conid is None:
        return {
            "error": "symbol_not_found",
            "symbol": sym,
            "message": "contract resolved but no contract id available",
        }

    try:
        req_id = session._client.request_sec_def_opt_params(
            underlying_symbol=sym, underlying_con_id=conid
        )
    except Exception as exc:  # noqa: BLE001
        logger.info("option_chain: request failed for %s: %s", sym, exc)
        return {
            "error": "request_failed",
            "symbol": sym,
            "underlying_con_id": conid,
            "message": str(exc),
        }

    # Poll until the row set is non-empty AND its count has not changed across
    # one full poll_interval_s — this prevents early exit on a partially-delivered
    # first batch.  snapshot_raw_rows_where filters in the engine before
    # materialization, keeping per-poll cost O(matching) vs O(total).
    deadline = _time.monotonic() + timeout_s
    prev_count = -1
    rows: list[dict[str, Any]] = []
    while _time.monotonic() < deadline:
        rows = supervisor.snapshot_raw_rows_where(_CHAIN_TABLE, f"RequestId == {req_id}")
        if rows and len(rows) == prev_count:
            break
        prev_count = len(rows)
        _time.sleep(poll_interval_s)

    if not rows:
        return {
            "error": "chain_timeout",
            "symbol": sym,
            "underlying_con_id": conid,
            "message": (
                f"no chain data for {sym!r} within {timeout_s:.0f}s — "
                "symbol may not have listed options, or the request timed out"
            ),
        }

    params = [
        OptionParams(
            exchange=str(r.get("Exchange")),
            trading_class=str(r.get("TradingClass")),
            multiplier=str(r.get("Multiplier")),
            expirations=_parse_expirations(str(r.get("Expirations", ""))),
            strikes=_parse_strikes(str(r.get("Strikes", ""))),
        )
        for r in rows
    ]
    smart = next((p for p in params if p.exchange == "SMART"), None)
    return OptionChain(
        symbol=sym, underlying_con_id=conid, smart=smart, by_exchange=params
    )


def subscribe_option_greeks(
    supervisor: Any,
    symbol: str,
    expiry: str | _dt.date,
    strike: float,
    right: str,
    exchange: str = "SMART",
    trading_class: str | None = None,
) -> int | None:
    """Subscribe Greeks for ONE option contract (bounded, on-demand).

    Builds the OPT contract, resolves it, and issues a single request_market_data so
    tickOptionComputation flows into the existing ``ticks_option_computation`` table.
    Returns the contract id (read it back via latest_option_greeks), or None if the
    contract cannot be resolved (or is ambiguous — see ``trading_class`` below). NO
    bulk "subscribe whole chain" path exists — that would be the bloat trap. Live
    verification is entitlement-gated (paper → 10089).

    Parameters
    ----------
    supervisor:
        A connected IbkrSupervisor.
    symbol:
        Underlying ticker (e.g. "AAPL").
    expiry:
        Expiry as a "YYYYMMDD" string or a ``datetime.date`` object.
    strike:
        Option strike price.
    right:
        "C" for call, "P" for put.
    exchange:
        Routing exchange (default "SMART").
    trading_class:
        Optional trading-class disambiguator (typically an already-parsed
        ``OptionParams.trading_class`` from ``option_chain``). Some underlyings
        register to MORE THAN ONE ``ContractDetails`` when this is omitted — e.g.
        SPY's SMART chain resolves to both ``'2SPY'`` and ``'SPY'`` — and
        deephaven-ib's ``request_market_data`` issues one ``reqMktData`` PER
        ``ContractDetails`` on the registered contract (its own docstring:
        "Registered contracts that are associated with multiple contract details
        produce multiple requests"), so an under-specified contract silently spends
        more than one market-data line for what looks like one subscription. Passing
        ``trading_class`` lets IBKR collapse the resolution to a single
        ``ContractDetails``.
    """
    session = supervisor._session
    contract = build_option_contract(
        symbol, expiry, strike, right, exchange=exchange, trading_class=trading_class
    )
    try:
        rc = session.get_registered_contract(contract)
    except Exception as exc:  # noqa: BLE001
        logger.info("subscribe_option_greeks: %s not resolved: %s", symbol, exc)
        return None
    if rc.is_multi():
        # Mirrors marketdata.py's request_intraday_bars guard (`if rc.is_multi():
        # ... continue`): even with trading_class supplied (or when the caller had
        # none to give), the contract can still be ambiguous. Skip rather than let
        # request_market_data fan out into N reqMktData calls / N market-data lines
        # for what the caller budgeted as one subscription.
        logger.warning(
            "subscribe_option_greeks: skipping multi-contract option %s "
            "(expiry=%s strike=%s right=%s trading_class=%r) — ambiguous ContractDetails",
            symbol, expiry, strike, right, trading_class,
        )
        return None
    conid = _conid_of(rc)
    if conid is None:
        return None
    try:
        session.request_market_data(rc, snapshot=False)
    except Exception as exc:  # noqa: BLE001
        logger.info("subscribe_option_greeks: request failed for %s: %s", symbol, exc)
        return None
    return conid


def latest_option_greeks(supervisor: Any, con_id: int) -> dict[str, Any] | None:
    """Return the most recent ticks_option_computation row for ``con_id``, or None.

    'Most recent' = the last matching row in the append-only table (arrival order).
    Filters in the engine before materialization (O(matching) vs O(total)).

    Parameters
    ----------
    supervisor:
        A connected IbkrSupervisor.
    con_id:
        IBKR contract ID to look up.
    """
    matching = supervisor.snapshot_raw_rows_where(_GREEKS_TABLE, f"ContractId == {con_id}")
    return matching[-1] if matching else None


# ---------------------------------------------------------------------------
# ATM-option subscribe helper — reusable across any caller needing "the nearest
# qualifying, at-the-money option for this underlying" without hand-rolling
# expiry/strike selection each time (a general-purpose adapter-layer gap a prior
# caller needing per-name implied-vol overlay identified as UN-RUN/needs-market-hours).
# ---------------------------------------------------------------------------


def _third_friday(year: int, month: int) -> _dt.date:
    """The standard-monthly-option expiration date: the 3rd Friday of ``(year, month)``."""
    first = _dt.date(year, month, 1)
    first_friday = first + _dt.timedelta(days=(4 - first.weekday()) % 7)
    return first_friday + _dt.timedelta(weeks=2)


def nearest_monthly_expiry(
    expirations: Sequence[str], today: _dt.date, *, min_dte: int = 20
) -> str | None:
    """Earliest STANDARD MONTHLY expiry (3rd Friday of its month) at least ``min_dte``
    days out, or ``None`` if no monthly qualifies.

    Standard monthlies carry the full strike ladder ``reqSecDefOptParams`` lists;
    weeklies/short-dated series frequently only support a narrow strike subset. This
    matters because ``reqSecDefOptParams`` returns ``expirations``/``strikes`` as
    separate UNION lists across the WHOLE chain (see ``OptionParams``'s docstring and
    ``subscribe_atm_option``'s) — not every ``(expiry, strike)`` pair drawn independently
    from those two lists is a contract that actually exists. Preferring a monthly
    expiry reduces (does not eliminate) the odds of picking a nonexistent combination.

    ``expirations`` are IBKR's "YYYYMMDD" strings, any order. Unparseable entries are
    skipped (never raise), matching :func:`nearest_expiry`.
    """
    candidates: list[tuple[int, str]] = []
    for token in expirations:
        try:
            d = parse_expiry(token)
        except ValueError:
            logger.debug("nearest_monthly_expiry: skipping malformed expiry token %r", token)
            continue
        if d != _third_friday(d.year, d.month):
            continue
        dte = (d - today).days
        if dte >= min_dte:
            candidates.append((dte, token))
    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0])
    return candidates[0][1]


def nearest_expiry(
    expirations: Sequence[str], today: _dt.date, *, min_dte: int = 20
) -> str | None:
    """Earliest expiry at least ``min_dte`` days out, preferring a standard MONTHLY.

    Tries :func:`nearest_monthly_expiry` first — a qualifying monthly (3rd Friday,
    >= ``min_dte`` out) is returned immediately, since monthlies carry the fuller
    strike ladder (see that function's docstring for why this matters for
    ``subscribe_atm_option``'s union-list strike/expiry pairing). Falls back to the
    nearest ANY-expiry (this function's original, sole contract) when no monthly
    qualifies — e.g. a short-dated chain with no monthly >= ``min_dte`` out.

    ``expirations`` are IBKR's "YYYYMMDD" strings (as delivered on the wire / parsed
    back out of ``OptionParams.expirations``, see ``subscribe_atm_option``). Returns
    ``None`` if no expiry qualifies, or if every entry is unparseable. Unparseable
    entries are skipped (never raise).

    Parameters
    ----------
    expirations:
        Candidate expiry strings, any order.
    today:
        Reference date for the day-to-expiry (DTE) computation.
    min_dte:
        Minimum days-to-expiry required to qualify (boundary is inclusive: DTE ==
        min_dte qualifies).
    """
    monthly = nearest_monthly_expiry(expirations, today, min_dte=min_dte)
    if monthly is not None:
        return monthly

    candidates: list[tuple[int, str]] = []
    for token in expirations:
        try:
            d = parse_expiry(token)
        except ValueError:
            logger.debug("nearest_expiry: skipping malformed expiry token %r", token)
            continue
        dte = (d - today).days
        if dte >= min_dte:
            candidates.append((dte, token))
    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0])
    return candidates[0][1]


def atm_strike(strikes: Sequence[float], spot: float) -> float | None:
    """Strike closest to ``spot``. ``None`` if ``strikes`` is empty or ``spot`` is not positive.

    Ties resolve to the lower strike (deterministic): when two strikes are
    equidistant from ``spot``, ``min`` breaks the tie on the strike value itself.
    """
    if not strikes or spot <= 0:
        return None
    return min(strikes, key=lambda k: (abs(k - spot), k))


def atm_strike_candidates(
    strikes: Sequence[float], spot: float, *, k: int = 5
) -> list[float]:
    """The ``k`` nearest strikes to ``spot``, nearest-first (ties to the lower strike).

    ``[]`` if ``strikes`` is empty or ``spot`` is not positive (mirrors :func:`atm_strike`'s
    guard). Used by :func:`subscribe_atm_option` to try several candidate strikes at a
    fixed expiry rather than betting everything on the single nearest one — because
    ``reqSecDefOptParams``'s ``strikes``/``expirations`` are separate UNION lists across
    the whole chain (see :func:`nearest_monthly_expiry`'s docstring), the single nearest
    strike is not guaranteed to actually be listed at the chosen expiry.
    """
    if not strikes or spot <= 0:
        return []
    ordered = sorted(strikes, key=lambda s: (abs(s - spot), s))
    return ordered[:k]


def subscribe_atm_option(
    supervisor: Any,
    symbol: str,
    spot: float,
    *,
    min_dte: int = 20,
    right: str = "C",
    exchange: str = "SMART",
    today: _dt.date | None = None,
    max_strike_candidates: int = 5,
) -> int | None:
    """Subscribe the nearest-expiry (>= ``min_dte``), at-the-money option and return its conId.

    **This is NOT a hard "at most one market-data line" guarantee** — that claim was
    live-verified false: SPY resolves to 2 ``ContractDetails`` on SMART
    and spent 2 lines on one ``request_market_data`` call before this fix. Two
    mitigations are in place: (1) the already-parsed ``OptionParams.trading_class``
    is now passed through to ``subscribe_option_greeks``, letting IBKR collapse a
    multi-trading-class underlying like SPY to a single ``ContractDetails``; (2)
    ``subscribe_option_greeks`` guards with ``rc.is_multi()`` (mirroring
    ``marketdata.py``'s ``request_intraday_bars``) and skips — spending zero lines —
    a contract still ambiguous after that. What is NOT fully closed: if
    ``request_market_data`` itself raises AFTER a successful (non-ambiguous)
    resolution, the candidate loop below still advances to the next strike, so the
    true worst case across one call remains bounded by ``max_strike_candidates``, not
    a hard 1. Uses the ATM CALL by default — a call+put straddle would double the
    line cost, which risks blowing a market-data-line quota (observed live:
    repeated restarts each re-subscribing without releasing prior lines exhausted
    the account's market-data-line allotment).

    Composes the existing bounded primitives: ``option_chain`` (one-shot chain
    discovery) -> ``nearest_expiry`` (prefers a standard monthly) /
    ``atm_strike_candidates`` (pure selection) -> ``subscribe_option_greeks`` (one
    bounded per-contract subscription, tried per candidate). Never raises — returns
    ``None`` on any failure (chain error, no qualifying expiry, no strikes, every
    candidate fails to subscribe).

    **Why multiple strike candidates, not just the nearest:** ``reqSecDefOptParams``
    (hence ``OptionParams.expirations``/``.strikes``) returns two separate UNION lists
    across the WHOLE chain, not a per-expiry strike ladder — not every
    ``(expiry, strike)`` pair drawn independently from those lists is a real,
    listed contract. The single-nearest-strike-at-the-nearest-expiry choice this
    function used to make could (and, live, did) pick a
    nonexistent combination: IBKR then rejects it with ``ErrorCode=200`` ("No
    security definition has been found for the request") and the name silently
    drops from any caller treating a lone failed attempt as "no contract exists."
    This function now tries up to ``max_strike_candidates`` nearest strikes (at the
    ONE chosen, monthly-preferred expiry — see ``nearest_expiry``) in order, moving
    to the next candidate whenever one fails to resolve.

    **Why this costs at most one line, not one per failed candidate:**
    ``subscribe_option_greeks`` resolves the contract via
    ``session.get_registered_contract`` (a ``reqContractDetails``-backed, line-free
    reference call) BEFORE ever calling ``request_market_data`` — a candidate whose
    contract does not exist fails (raises, or resolves with no usable conId) at that
    resolution step and returns ``None`` without ``request_market_data`` ever being
    reached. So a failing candidate spends zero market-data lines; only the ONE
    candidate that resolves successfully goes on to subscribe. IBKR still logs each
    failing candidate's resolution failure into the ``errors`` table as
    ``ErrorCode=200`` (the same code, regardless of which request type triggered
    it) — a caller diffing the ``errors`` table across an incremental-subscribe
    sequence should expect up to ``max_strike_candidates - 1`` such rows per name
    that needed more than one candidate, and should not treat a `200` alone as
    evidence of anything beyond "one candidate contract did not exist."

    Parameters
    ----------
    supervisor:
        A connected IbkrSupervisor.
    symbol:
        Underlying ticker (e.g. "AAPL").
    spot:
        Reference underlying price used to pick ATM strike candidates (e.g. the
        latest IB close already on hand — no extra market-data line is spent to
        get it). This is a caller-supplied number: ``ibda`` reaches no non-IB data
        source to obtain one, by design (see ``ibda.rates`` for the same rule
        applied to the risk-free rate).
    min_dte:
        Minimum days-to-expiry for the chosen expiry.
    right:
        "C" for call (default), "P" for put.
    exchange:
        Routing exchange (default "SMART").
    today:
        Reference date for DTE computation; defaults to ``date.today()`` (injected
        here so tests can pin it).
    max_strike_candidates:
        Maximum number of nearest-strike candidates to try before giving up
        (default 5 — see :func:`atm_strike_candidates`).
    """
    try:
        chain = option_chain(supervisor, symbol)
        if isinstance(chain, dict):
            logger.info(
                "subscribe_atm_option: %s chain error: %s", symbol, chain.get("error")
            )
            return None
        params = chain.smart
        if params is None:
            params = next(
                (p for p in chain.by_exchange if p.exchange == exchange), None
            )
        if params is None:
            logger.info(
                "subscribe_atm_option: %s has no %s chain params", symbol, exchange
            )
            return None
        expiry_tokens = [d.strftime("%Y%m%d") for d in params.expirations]
        ref_date = today if today is not None else _dt.date.today()
        expiry = nearest_expiry(expiry_tokens, ref_date, min_dte=min_dte)
        if expiry is None:
            logger.info(
                "subscribe_atm_option: %s has no expiry >= %d DTE", symbol, min_dte
            )
            return None
        candidates = atm_strike_candidates(params.strikes, spot, k=max_strike_candidates)
        if not candidates:
            logger.info("subscribe_atm_option: %s has no usable strikes", symbol)
            return None
        for strike in candidates:
            con_id = subscribe_option_greeks(
                supervisor, symbol, expiry, strike, right, exchange=exchange,
                trading_class=params.trading_class,
            )
            if con_id is not None:
                return con_id
            logger.info(
                "subscribe_atm_option: %s candidate strike=%s @ expiry=%s did not resolve "
                "(union-list expiry/strike pairing -- see this function's docstring); "
                "trying the next nearest strike, if any. No market-data line was spent.",
                symbol, strike, expiry,
            )
        logger.info(
            "subscribe_atm_option: %s exhausted all %d strike candidate(s) at expiry %s "
            "with no resolvable contract",
            symbol, len(candidates), expiry,
        )
        return None
    except Exception as exc:  # noqa: BLE001 — never raise into callers
        logger.info("subscribe_atm_option: %s failed: %s", symbol, exc)
        return None
