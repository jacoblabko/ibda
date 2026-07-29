"""ibda.analytics.timeseries — rolling and per-period performance, as Arrow.

Turns the canonical daily ``nav`` series into time-series analytics a consumer can
chart: a rolling window of Sharpe / volatility / drawdown, and a per-calendar-period
return table. Output is plain :class:`pyarrow.Table` (derived analytics, like
:mod:`ibda.analytics.indicators`), not a registered canonical schema.

All return math is reused from :mod:`ibda.analytics.performance` so the numbers
reconcile exactly with :func:`ibda.analytics.performance.compute_performance`.

Pure module: stdlib + pyarrow only. No engine, no vendor SDK.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from typing import TypeAlias, cast

import pyarrow as pa

from ibda.analytics.performance import (
    _annualized_volatility,
    _cumulative_return,
    _HasSnapshot,
    _HasTable,
    _max_drawdown,
    _dated_returns,
    _nav_series,
    _resolve_source,
    _select_account,
    _sharpe_from_returns,
)
from ibda.rates import DEFAULT_PERIODS_PER_YEAR, DEFAULT_RISK_FREE_ANNUAL, resolve_risk_free

#: Explicitly a ``TypeAlias``: ``pa.Table`` is untyped (Any), so mypy does not infer the
#: bare ``A | B | Any`` assignment as an alias and rejects it at every use site.
_Source: TypeAlias = _HasTable | _HasSnapshot | pa.Table



def _returns_with_timestamps(
    nav: pa.Table,
    timestamps: list[datetime],
    value_column: str,
    flows: Mapping[date, float] | None,
) -> tuple[list[float], list[datetime]]:
    """Returns paired with the timestamp of the NAV point each one ENDS at.

    The obvious ``ret_ts = timestamps[1:]`` assumes
    ``len(returns) == len(timestamps) - 1``. ``_dated_returns`` does **not** guarantee
    that: it *skips* any period whose prior NAV is ``0.0`` (documented in its own
    docstring). That is reachable on the default ``value_column="Total"`` — a Flex report
    whose first row precedes funding — and easily on the documented alternatives ``Cash``
    and ``Stock``, which are legitimately ``0.0`` on a fully-invested or all-cash day.

    Pairing returns to NAV points positionally cannot survive that skip: one skipped
    period shifts every subsequent label one NAV point early — ``rolling_performance``
    would stamp a window ending 2026-01-04 as 2026-01-03, contradicting its "labelled by
    the timestamp of the window's last NAV point" contract — and ``periodic_returns``
    would raise ``ValueError: zip() argument 2 is shorter than argument 1`` from its
    ``strict=True`` zip. Both pair on timestamps instead.

    Taking each return's own ``later_date`` from ``_dated_returns`` removes the assumption
    rather than patching it — there is no alignment left to get wrong.
    """
    dated = _dated_returns(nav, value_column=value_column, flows=flows)
    by_date: dict[date, datetime] = {}
    for ts in timestamps:
        by_date.setdefault(ts.date(), ts)
    returns = [r for _, _, r in dated]
    ret_ts = [by_date[later] for _, later, _ in dated]
    return returns, ret_ts


def rolling_performance(
    source: _Source,
    *,
    account: str | None = None,
    window: int = 63,
    risk_free_annual: str | float = DEFAULT_RISK_FREE_ANNUAL,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    value_column: str = "Total",
    adjust_for_flows: bool = True,
) -> pa.Table:
    """Rolling performance over a trailing *window* of trading days.

    Args:
        source: a DataPort, Result, or bare NAV Arrow table.
        window: trailing window length in trading days (number of returns per window).
        risk_free_annual: annual simple risk-free rate, or ``"auto"`` for the documented
            offline Treasury-yield proxy (:data:`ibda.rates.DEFAULT_RISK_FREE_ANNUAL`).
        periods_per_year: annualization factor (default 252 trading days).
        value_column: NAV value column name (default ``"Total"``).
        adjust_for_flows: derive external flows from the port's cash table (DataPort only).

    Returns:
        One output row per window once ``window`` returns are available, labelled by the
        timestamp of the window's last NAV point. Columns: ``Timestamp``, ``Return``
        (compounded over the window), ``Volatility`` (annualized), ``Sharpe`` (annualized),
        ``MaxDrawdown`` (non-positive).

    Raises:
        ValueError: if ``window < 2``, or *risk_free_annual* is a non-numeric,
            non-``"auto"`` string.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2 to define volatility; got {window}")
    rf, _rf_source = resolve_risk_free(risk_free_annual)
    # account= is forwarded to _resolve_source so a DataPort's derived flows come from the
    # SAME account as the NAV selected below — mirrors benchmark._aligned_returns exactly.
    # _resolve_source never filters NAV (only cash), so _select_account remains the sole
    # place NAV is filtered; no double-filtering.
    nav, flows = _resolve_source(source, adjust_for_flows=adjust_for_flows, account=account)
    nav = _select_account(nav, account)
    timestamps, _values = _nav_series(nav, value_column)
    returns, ret_ts = _returns_with_timestamps(nav, timestamps, value_column, flows)

    out_ts: list[datetime] = []
    out_ret: list[float] = []
    out_vol: list[float] = []
    out_sharpe: list[float] = []
    out_dd: list[float] = []
    for end in range(window - 1, len(returns)):
        win = returns[end - window + 1 : end + 1]
        out_ts.append(ret_ts[end])
        out_ret.append(_cumulative_return(win))
        out_vol.append(_annualized_volatility(win, periods_per_year=periods_per_year))
        out_sharpe.append(
            _sharpe_from_returns(win, risk_free_annual=rf, periods_per_year=periods_per_year)
        )
        out_dd.append(_max_drawdown(win))

    return cast(pa.Table, pa.table({
        "Timestamp": pa.array(out_ts, type=pa.timestamp("ns", tz="UTC")),
        "Return": pa.array(out_ret, type=pa.float64()),
        "Volatility": pa.array(out_vol, type=pa.float64()),
        "Sharpe": pa.array(out_sharpe, type=pa.float64()),
        "MaxDrawdown": pa.array(out_dd, type=pa.float64()),
    }))


def _period_key(ts: datetime, freq: str) -> str:
    """Calendar bucket label for *ts*: '2026-01' | '2026-Q2' | '2026'."""
    if freq == "monthly":
        return f"{ts.year:04d}-{ts.month:02d}"
    if freq == "quarterly":
        return f"{ts.year:04d}-Q{(ts.month - 1) // 3 + 1}"
    if freq == "yearly":
        return f"{ts.year:04d}"
    raise ValueError(f"unsupported freq {freq!r}; use 'monthly', 'quarterly', or 'yearly'")


def periodic_returns(
    source: _Source,
    *,
    account: str | None = None,
    freq: str = "monthly",
    risk_free_annual: str | float = DEFAULT_RISK_FREE_ANNUAL,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    value_column: str = "Total",
    adjust_for_flows: bool = True,
) -> pa.Table:
    """Per-calendar-period return table from a daily NAV series.

    Args:
        source: a DataPort, Result, or bare NAV Arrow table.
        freq: one of ``"monthly"``, ``"quarterly"``, ``"yearly"``.
        risk_free_annual: annual simple risk-free rate, or ``"auto"`` for the documented
            offline Treasury-yield proxy (:data:`ibda.rates.DEFAULT_RISK_FREE_ANNUAL`).
        periods_per_year: annualization factor (default 252 trading days).
        value_column: NAV value column name (default ``"Total"``).
        adjust_for_flows: derive external flows from the port's cash table (DataPort only).

    Returns:
        One row per period present in the data, in chronological order. Columns:
        ``Period`` (label), ``Start`` / ``End`` (period's first/last return timestamp),
        ``Return`` (compounded within period), ``Volatility`` (annualized),
        ``Sharpe`` (annualized; ``nan`` for single-observation periods).

    Raises:
        ValueError: if *freq* is not one of the supported values, or *risk_free_annual*
            is a non-numeric, non-``"auto"`` string.
    """
    rf, _rf_source = resolve_risk_free(risk_free_annual)
    # account= is forwarded to _resolve_source so a DataPort's derived flows come from the
    # SAME account as the NAV selected below — mirrors benchmark._aligned_returns exactly.
    # _resolve_source never filters NAV (only cash), so _select_account remains the sole
    # place NAV is filtered; no double-filtering.
    nav, flows = _resolve_source(source, adjust_for_flows=adjust_for_flows, account=account)
    nav = _select_account(nav, account)
    timestamps, _values = _nav_series(nav, value_column)
    returns, ret_ts = _returns_with_timestamps(nav, timestamps, value_column, flows)

    # Preserve first-seen order of period keys.
    order: list[str] = []
    buckets: dict[str, list[tuple[datetime, float]]] = {}
    for ts, r in zip(ret_ts, returns, strict=True):
        key = _period_key(ts, freq)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append((ts, r))

    periods: list[str] = []
    starts: list[datetime] = []
    ends: list[datetime] = []
    rets: list[float] = []
    vols: list[float] = []
    sharpes: list[float] = []
    for key in order:
        items = buckets[key]
        win = [r for _ts, r in items]
        periods.append(key)
        starts.append(items[0][0])
        ends.append(items[-1][0])
        rets.append(_cumulative_return(win))
        vols.append(_annualized_volatility(win, periods_per_year=periods_per_year))
        sharpes.append(
            _sharpe_from_returns(win, risk_free_annual=rf, periods_per_year=periods_per_year)
        )

    return cast(pa.Table, pa.table({
        "Period": pa.array(periods, type=pa.string()),
        "Start": pa.array(starts, type=pa.timestamp("ns", tz="UTC")),
        "End": pa.array(ends, type=pa.timestamp("ns", tz="UTC")),
        "Return": pa.array(rets, type=pa.float64()),
        "Volatility": pa.array(vols, type=pa.float64()),
        "Sharpe": pa.array(sharpes, type=pa.float64()),
    }))
