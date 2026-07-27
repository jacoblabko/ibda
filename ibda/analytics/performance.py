"""ibda.analytics.performance — historic return & risk-adjusted metrics.

Turns a daily NAV time series (the canonical ``nav`` table) into the metrics a
trader actually asks for: cumulative & annualized return, volatility, **Sharpe
ratio**, Sortino ratio, and max drawdown.

Why a NAV series and not raw P&L
--------------------------------
A stream of P&L numbers is *not* enough to compute Sharpe. Sharpe is

    mean(excess return) / stdev(return)   (annualized)

so it needs **returns**, not currency P&L: ``r_t = (NAV_t - NAV_{t-1} - F_t) / NAV_{t-1}``
where ``F_t`` is the net external cash flow (deposits +, withdrawals −) over the
period. That requires three things P&L alone lacks:

1. a *denominator* — prior-period NAV — to turn P&L into a return (notional value
   is the wrong base: it ignores cash and leverage);
2. a *regular periodicity* (daily here) so volatility is well defined;
3. *cash-flow stripping* so a deposit is not mistaken for a gain.

This module takes the daily NAV series (from Flex ``EquitySummaryByReportDateInBase``)
and, optionally, per-day external flows, and produces a time-weighted return (TWR)
series and the metrics derived from it.

Conventions
-----------
* **Flow timing:** flows are treated as occurring at period end —
  ``r_t = (NAV_t - NAV_{t-1} - F_t) / NAV_{t-1}``. This is the simple TWR
  convention; it matches IBKR's daily-valued NAV closely when flows are small
  relative to NAV.
* **Annualization:** ``periods_per_year`` (default 252 trading days). Volatility
  scales by ``sqrt(periods_per_year)``; the mean return scales linearly.
* **Risk-free rate:** ``risk_free_annual`` is an annual simple rate, divided by
  ``periods_per_year`` to get the per-period rate subtracted from each return.
* **Max drawdown** is computed on the flow-stripped wealth index (cumulative
  product of ``1 + r_t``), so deposits/withdrawals never appear as drawdown.

Pure module: imports only stdlib + pyarrow. No engine, no vendor SDK.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any, Protocol, Union, cast, runtime_checkable

import pyarrow as pa

from ibda.rates import DEFAULT_PERIODS_PER_YEAR, DEFAULT_RISK_FREE_ANNUAL, resolve_risk_free


@dataclass(frozen=True)
class PerformanceSummary:
    """Historic performance of an account over its NAV history.

    All ratios are annualized. ``num_periods`` is the count of return
    observations (one fewer than the number of NAV points). Ratios are ``nan``
    when undefined (e.g. zero-volatility history).
    """

    start: datetime
    end: datetime
    num_periods: int
    starting_nav: float
    ending_nav: float
    net_external_flows: float
    # True when external deposit/withdrawal flows were supplied and stripped from
    # returns; False when flows=None was passed and NAV-to-NAV changes are raw
    # (un-adjusted for deposits/withdrawals).
    flows_applied: bool
    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    hit_rate: float
    best_period: float
    worst_period: float
    risk_free_annual: float
    periods_per_year: int
    account: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict (datetimes rendered as ISO-8601 strings)."""
        out: dict[str, Any] = asdict(self)
        out["start"] = self.start.isoformat()
        out["end"] = self.end.isoformat()
        return out

    def render(self) -> str:
        """Render a plain-text report suitable for sharing with non-experts.

        Percentages, currency, and a one-line glossary so a reader who has never
        computed a Sharpe ratio can still interpret the numbers.
        """

        def pct(x: float) -> str:
            return "n/a" if math.isnan(x) else f"{x * 100:+.2f}%"

        def ratio(x: float) -> str:
            return "n/a" if math.isnan(x) else f"{x:.2f}"

        def money(x: float) -> str:
            return f"${x:,.2f}"

        acct = self.account or "—"
        span = f"{self.start.date().isoformat()} → {self.end.date().isoformat()}"
        rf = f"{self.risk_free_annual * 100:.2f}%"
        lines = [
            "Performance summary",
            f"  Account             {acct}",
            f"  Period              {span}   ({self.num_periods} trading days)",
            f"  Starting NAV        {money(self.starting_nav)}",
            f"  Ending NAV          {money(self.ending_nav)}",
            f"  Net deposits/wd     {money(self.net_external_flows)}"
            + ("" if self.flows_applied else "  (not flow-adjusted)"),
            "",
            f"  Cumulative return   {pct(self.cumulative_return)}",
            f"  Annualized return   {pct(self.annualized_return)}",
            f"  Annualized vol      {pct(self.annualized_volatility)}",
            f"  Sharpe ratio        {ratio(self.sharpe_ratio)}"
            f"   (risk-free {rf}, {self.periods_per_year}/yr)",
            f"  Sortino ratio       {ratio(self.sortino_ratio)}",
            f"  Calmar ratio        {ratio(self.calmar_ratio)}",
            f"  Max drawdown        {pct(self.max_drawdown)}",
            f"  Positive days       {pct(self.hit_rate).lstrip('+')}",
            f"  Best / worst day     {pct(self.best_period)} / {pct(self.worst_period)}",
            "",
            "  Sharpe = annualized return per unit of volatility; higher is better",
            "  (>1 is good, >2 very good). Sortino only penalizes downside moves;",
            "  Calmar = annualized return ÷ max drawdown. Max drawdown = worst",
            "  peak-to-trough drop.",
        ]
        return "\n".join(lines)


@runtime_checkable
class _HasTable(Protocol):
    """Structural type: a DataPort — anything exposing ``table(name)``."""

    def table(self, name: str) -> Any: ...


@runtime_checkable
class _HasSnapshot(Protocol):
    """Structural type: a Result — anything with a no-arg ``snapshot()``."""

    def snapshot(self) -> pa.Table: ...


def _sample_stdev(values: list[float]) -> float:
    """Sample standard deviation (ddof=1). Returns ``nan`` for fewer than 2 points."""
    n = len(values)
    if n < 2:
        return math.nan
    mean = math.fsum(values) / n
    var = math.fsum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(var)


def _downside_deviation(excess: list[float]) -> float:
    """Root-mean-square of the negative excess returns (target = risk-free rate).

    Uses the standard Sortino denominator: ``sqrt(mean(min(x, 0)**2))`` over all
    observations (non-negative excess contributes 0). Returns ``nan`` if empty.
    """
    if not excess:
        return math.nan
    downside = [min(x, 0.0) ** 2 for x in excess]
    return math.sqrt(math.fsum(downside) / len(excess))


def _max_drawdown(returns: list[float]) -> float:
    """Maximum drawdown of the wealth index built from *returns*.

    Returns a non-positive number (e.g. ``-0.12`` for a 12% peak-to-trough drop),
    or ``0.0`` when the series only ever rose.
    """
    wealth = 1.0
    peak = 1.0
    worst = 0.0
    for r in returns:
        wealth *= 1.0 + r
        peak = max(peak, wealth)
        drawdown = wealth / peak - 1.0
        worst = min(worst, drawdown)
    return worst


def _cumulative_return(returns: list[float]) -> float:
    """Compounded return over *returns*: ``prod(1 + r) - 1``. ``0.0`` if empty."""
    return math.prod(1.0 + r for r in returns) - 1.0


def _annualized_volatility(returns: list[float], *, periods_per_year: int) -> float:
    """Annualized stdev of *returns* (``nan`` for fewer than 2 points)."""
    stdev = _sample_stdev(returns)
    return stdev * math.sqrt(periods_per_year) if not math.isnan(stdev) else math.nan


def _sharpe_from_returns(
    returns: list[float],
    *,
    risk_free_annual: float,
    periods_per_year: int,
) -> float:
    """Annualized Sharpe of *returns*. ``nan`` when fewer than 2 points or zero variance.

    ``mean(excess) / stdev(returns) * sqrt(periods_per_year)``. The single formula
    :func:`compute_performance` calls for its ``sharpe_ratio`` field — no inline copy.
    """
    n = len(returns)
    if n < 2:
        return math.nan
    rf_period = risk_free_annual / periods_per_year
    excess = [r - rf_period for r in returns]
    mean_excess = math.fsum(excess) / n
    stdev = _sample_stdev(returns)
    if math.isnan(stdev) or stdev <= 0.0:
        return math.nan
    return mean_excess / stdev * math.sqrt(periods_per_year)


def _sortino_from_returns(
    returns: list[float],
    *,
    risk_free_annual: float,
    periods_per_year: int,
) -> float:
    """Annualized Sortino of *returns*. ``nan`` when fewer than 2 points or the
    downside deviation is zero/undefined.

    ``mean(excess) / downside_deviation(excess) * sqrt(periods_per_year)``. The single
    formula :func:`compute_performance` calls for its ``sortino_ratio`` field.
    """
    n = len(returns)
    if n < 2:
        return math.nan
    rf_period = risk_free_annual / periods_per_year
    excess = [r - rf_period for r in returns]
    mean_excess = math.fsum(excess) / n
    downside_dev = _downside_deviation(excess)
    if math.isnan(downside_dev) or downside_dev <= 0.0:
        return math.nan
    return mean_excess / downside_dev * math.sqrt(periods_per_year)


def _annualized_return_from_returns(returns: list[float], *, periods_per_year: int) -> float:
    """Annualized return implied by *returns*' compounded growth.

    ``nan`` when fewer than 2 points (a single period cannot be annualized without
    extrapolating one observation into a full year), or when growth <= 0.

    ``growth ** (periods_per_year / n) - 1``. The single formula
    :func:`compute_performance` calls for its ``annualized_return`` field.
    """
    n = len(returns)
    if n < 2:
        return math.nan
    growth = math.prod(1.0 + r for r in returns)
    return growth ** (periods_per_year / n) - 1.0 if growth > 0.0 else math.nan


def _nav_series(
    nav: pa.Table,
    value_column: str,
) -> tuple[list[datetime], list[float]]:
    """Extract a (timestamps, values) pair sorted ascending by timestamp."""
    if value_column not in nav.column_names:
        raise ValueError(
            f"nav table has no {value_column!r} column; columns={nav.column_names}"
        )
    if "Timestamp" not in nav.column_names:
        raise ValueError("nav table has no 'Timestamp' column")

    timestamps = cast("list[datetime]", nav.column("Timestamp").to_pylist())
    values = cast("list[float | None]", nav.column(value_column).to_pylist())

    paired = [
        (ts, float(v))
        for ts, v in zip(timestamps, values, strict=True)
        if ts is not None and v is not None
    ]
    paired.sort(key=lambda p: p[0])
    if not paired:
        raise ValueError("nav table is empty after dropping null rows")

    out_ts = [p[0] for p in paired]
    out_val = [p[1] for p in paired]
    return out_ts, out_val


def daily_returns(
    nav: pa.Table,
    *,
    value_column: str = "Total",
    flows: Mapping[date, float] | None = None,
) -> list[float]:
    """Time-weighted per-period returns from a daily NAV table.

    ``r_t = (NAV_t - NAV_{t-1} - F_t) / NAV_{t-1}`` where ``F_t`` is the net
    external cash flow on the date of ``NAV_t`` (0 when *flows* is None or absent
    for that date). Periods where the prior NAV is 0 are skipped.

    Args:
        nav: canonical ``nav`` Arrow table (``Timestamp`` + value column).
        value_column: NAV value column name (default ``"Total"``).
        flows: optional date → net external flow (deposits +, withdrawals −).

    Returns:
        List of period returns, length ``len(nav_points) - 1``.
    """
    return [r for _pd, _d, r in _dated_returns(nav, value_column=value_column, flows=flows)]


def _dated_returns(
    nav: pa.Table,
    *,
    value_column: str = "Total",
    flows: Mapping[date, float] | None = None,
) -> list[tuple[date, date, float]]:
    """Per-period returns paired with the *prior* and *later* NAV point's dates.

    The shared primitive behind :func:`daily_returns` (which drops both dates) and the
    benchmark date-alignment (which keys by the later date and also needs the prior
    date, so a cross-series join can confirm both sides span the same period —
    see :func:`ibda.analytics.benchmark._returns_by_date`). ``r_t = (NAV_t - NAV_{t-1}
    - F_t) / NAV_{t-1}``; periods whose prior NAV is 0 are skipped. Each tuple is
    ``(prior_date, later_date, return)``.
    """
    timestamps, values = _nav_series(nav, value_column)
    out: list[tuple[date, date, float]] = []
    for i in range(1, len(values)):
        prev = values[i - 1]
        if prev == 0.0:
            continue
        prior_date = timestamps[i - 1].date()
        d = timestamps[i].date()
        flow = flows.get(d, 0.0) if flows is not None else 0.0
        out.append((prior_date, d, (values[i] - prev - flow) / prev))
    return out


def _select_account(nav_table: pa.Table, account: str | None) -> pa.Table:
    """Filter *nav_table* to a single account, raising on genuine ambiguity.

    The canonical account-selection primitive for the performance family — used
    directly by :func:`compute_performance`/:func:`performance_summary` to filter
    the NAV table, used again inside :func:`_resolve_source` to filter the ``cash``
    table when deriving flows, and re-exported by :mod:`ibda.analytics.benchmark`
    (:func:`~ibda.analytics.benchmark._aligned_returns`), which calls it directly on
    the NAV table so ``relative_summary``/``rolling_relative`` filter NAV identically
    to the ``performance_summary``/``sharpe_ratio`` path. The parallel flow-filtering
    parity depends on the caller also forwarding ``account`` into
    :func:`_resolve_source` — ``_aligned_returns`` does both (NAV via its own
    :func:`_select_account` call, flows via ``account=`` on :func:`_resolve_source`).
    Same ``account`` convention as :func:`ibda.adapters.ibkr.flex.arrow.flex_performance`:
    ``None`` (the default) is fine for a single-account NAV, but a NAV covering more
    than one account with no *account* given is genuinely ambiguous and raises rather
    than silently picking one.

    A table with no ``"Account"`` column (e.g. a hand-built bare-returns table with no
    account concept) passes through unchanged regardless of *account*.

    Raises:
        ValueError: *account* is given but no rows match it, or *account* is ``None``
            and the table covers more than one account.
    """
    if "Account" not in nav_table.column_names:
        return nav_table
    accounts = cast("list[str | None]", nav_table.column("Account").to_pylist())
    present = sorted({a for a in accounts if a})
    if account is not None:
        indices = [i for i, a in enumerate(accounts) if a == account]
        if not indices:
            raise ValueError(
                f"no NAV rows for account {account!r}; accounts present: {present}"
            )
        return cast(pa.Table, nav_table.take(pa.array(indices, type=pa.int64())))
    if len(present) > 1:
        raise ValueError(
            f"NAV data covers multiple accounts {present}; pass account=... to choose one."
        )
    return nav_table


def compute_performance(
    source: pa.Table,
    *,
    account: str | None = None,
    flows: Mapping[date, float] | None = None,
    risk_free_annual: float = DEFAULT_RISK_FREE_ANNUAL,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    value_column: str = "Total",
) -> PerformanceSummary:
    """Compute a full :class:`PerformanceSummary` from a daily NAV table.

    The pure-Arrow performance core: every other entry point in this module and
    :mod:`ibda.adapters.ibkr.flex.arrow` funnels into this one function rather than
    re-deriving Sharpe/Sortino/drawdown itself. ``risk_free_annual`` here is float-only
    (already resolved) — callers wanting the ``"auto"`` sentinel use
    :func:`performance_summary` or :func:`ibda.flex_performance`.

    Args:
        source: canonical ``nav`` Arrow table — at least two NAV points are needed.
        account: required only if *source* covers more than one account — the
            account id to analyze. Same convention as
            :func:`ibda.adapters.ibkr.flex.arrow.flex_performance`'s ``account``: a
            multi-account NAV with no *account* given is rejected rather than
            silently picking one (see the ``ValueError`` below).
        flows: optional date → net external flow used to strip deposits/withdrawals
            from returns. When omitted, returns are pure NAV-to-NAV changes; if the
            account had material external flows the metrics will be distorted, so
            prefer supplying flows (``performance_summary`` derives them from the
            cash table automatically).
        risk_free_annual: annual simple risk-free rate (e.g. ``0.05`` for 5%).
        periods_per_year: annualization factor (default 252 trading days).
        value_column: NAV value column name (default ``"Total"``).

    Returns:
        A :class:`PerformanceSummary`.

    Raises:
        ValueError: if the table has fewer than two usable NAV points, *source*
            covers more than one account and *account* is ``None``, or *account*
            is given but absent from *source* (see :func:`_select_account`).
    """
    source = _select_account(source, account)
    timestamps, values = _nav_series(source, value_column)
    if len(values) < 2:
        raise ValueError(
            "need at least two NAV observations to compute performance; "
            f"got {len(values)}. Ensure the Flex query includes a multi-day "
            "EquitySummaryByReportDateInBase section."
        )

    returns = daily_returns(source, value_column=value_column, flows=flows)
    n = len(returns)

    sharpe = _sharpe_from_returns(
        returns, risk_free_annual=risk_free_annual, periods_per_year=periods_per_year
    )
    sortino = _sortino_from_returns(
        returns, risk_free_annual=risk_free_annual, periods_per_year=periods_per_year
    )
    cumulative_return = _cumulative_return(returns)
    annualized_return = _annualized_return_from_returns(returns, periods_per_year=periods_per_year)
    annualized_volatility = _annualized_volatility(returns, periods_per_year=periods_per_year)
    max_drawdown = _max_drawdown(returns)

    # Calmar = annualized return per unit of worst drawdown; undefined with no drawdown.
    calmar = (
        annualized_return / abs(max_drawdown)
        if max_drawdown < 0.0 and not math.isnan(annualized_return)
        else math.nan
    )
    hit_rate = sum(1 for r in returns if r > 0.0) / n
    best_period = max(returns)
    worst_period = min(returns)

    net_flows = 0.0
    if flows is not None:
        net_flows = math.fsum(flows.values())

    # source is already validated/filtered to at most one account by _select_account
    # above, so accts (if non-empty) all agree — accts[0] is no longer a silent pick
    # among *different* accounts, just reading the single account's label back out.
    account_label: str | None = None
    if "Account" in source.column_names:
        accts = [
            a for a in cast("list[str | None]", source.column("Account").to_pylist()) if a
        ]
        account_label = accts[0] if accts else None

    return PerformanceSummary(
        start=timestamps[0],
        end=timestamps[-1],
        num_periods=n,
        starting_nav=values[0],
        ending_nav=values[-1],
        net_external_flows=net_flows,
        flows_applied=flows is not None,
        cumulative_return=cumulative_return,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_drawdown,
        hit_rate=hit_rate,
        best_period=best_period,
        worst_period=worst_period,
        risk_free_annual=risk_free_annual,
        periods_per_year=periods_per_year,
        account=account_label,
    )


def sharpe_ratio(
    source: Union[_HasTable, _HasSnapshot, pa.Table],
    *,
    flows: Mapping[date, float] | None = None,
    risk_free_annual: str | float = DEFAULT_RISK_FREE_ANNUAL,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    value_column: str = "Total",
    adjust_for_flows: bool = True,
    account: str | None = None,
) -> float:
    """Annualized Sharpe ratio from a DataPort, a Result, or a daily NAV table.

    A one-number convenience over :func:`compute_performance` — call
    :func:`performance_summary` directly for the full :class:`PerformanceSummary`
    (Sortino, drawdown, etc.) instead of just the Sharpe ratio. Accepts the same
    source union as :func:`performance_summary` and resolves NAV/flows the same way
    (both delegate to :func:`_resolve_source`).

    Args:
        source: a :class:`~ibda.port.DataPort` — its ``"nav"`` table is snapshotted
            and (when ``adjust_for_flows``) deposits/withdrawals are pulled from its
            ``"cash"`` table; a :class:`~ibda.result.Result` (or anything with
            ``snapshot()``); or a bare NAV :class:`pyarrow.Table`.
        flows: optional date → net external flow used to strip deposits/withdrawals
            from returns (see :func:`compute_performance`). Takes precedence over any
            flows auto-derived from a DataPort's cash table; this is the only way to
            supply flows when *source* is a bare table (no cash table to derive from).
        risk_free_annual: annual simple risk-free rate, or ``"auto"`` for the
            documented offline Treasury-yield proxy (:data:`ibda.rates.DEFAULT_RISK_FREE_ANNUAL`).
        periods_per_year: annualization factor (default 252 trading days).
        value_column: NAV value column name (default ``"Total"``).
        adjust_for_flows: when *source* is a DataPort and *flows* is not given,
            derive external flows from its cash table (see :func:`performance_summary`).
        account: required only if the resolved NAV covers more than one account — the
            account id to analyze. Same convention as
            :func:`ibda.adapters.ibkr.flex.arrow.flex_performance`'s ``account``.

    Returns:
        The annualized Sharpe ratio (``nan`` when undefined, e.g. zero-volatility history).

    Raises:
        ValueError: if the resolved NAV table has fewer than two usable NAV points,
            *source* is a DataPort with no ``"nav"`` table, *risk_free_annual* is a
            non-numeric, non-``"auto"`` string, the resolved NAV covers more than one
            account and *account* is ``None``, or *account* is given but absent
            (see :func:`_select_account`).
    """
    rf, _rf_source = resolve_risk_free(risk_free_annual)
    nav, resolved_flows = _resolve_source(
        source, adjust_for_flows=adjust_for_flows, account=account
    )
    if flows is not None:
        resolved_flows = dict(flows)
    return compute_performance(
        nav,
        account=account,
        flows=resolved_flows,
        risk_free_annual=rf,
        periods_per_year=periods_per_year,
        value_column=value_column,
    ).sharpe_ratio


# Cash-transaction types that represent external capital flows (not P&L).
# "transfer" catches ACATS in-kind security transfers (Type="Transfer") which
# are mapped from Flex <Transfer> elements and must be stripped from NAV-based
# returns to avoid inflating/deflating Sharpe, Sortino, etc.
_FLOW_TYPE_MARKERS: tuple[str, ...] = ("deposit", "withdraw", "transfer")


def external_flows_from_cash(cash: pa.Table) -> dict[date, float]:
    """Aggregate deposit/withdrawal/transfer cash transactions into per-date net flows.

    Matches canonical ``cash`` rows whose ``Type`` mentions "deposit",
    "withdraw", or "transfer" (case-insensitive) and sums their signed
    ``Amount`` by date. Withdrawals already carry a negative ``Amount`` in Flex,
    so the sum is the net external flow directly usable as ``F_t``.

    ACATS in-kind security transfers have ``Type="Transfer"`` and are included
    so their cash-equivalent value is stripped from NAV-based returns (Sharpe,
    Sortino, Calmar). Unvalued transfers are not mapped to cash rows and are
    therefore not stripped; a warning is logged at parse time for those.
    """
    cols = cash.column_names
    if "Type" not in cols or "Amount" not in cols or "Timestamp" not in cols:
        return {}
    types = cast("list[str | None]", cash.column("Type").to_pylist())
    amounts = cast("list[float | None]", cash.column("Amount").to_pylist())
    times = cast("list[datetime | None]", cash.column("Timestamp").to_pylist())

    out: dict[date, float] = {}
    for typ, amt, ts in zip(types, amounts, times, strict=True):
        if typ is None or amt is None or ts is None:
            continue
        low = typ.lower()
        if any(marker in low for marker in _FLOW_TYPE_MARKERS):
            d = ts.date()
            out[d] = out.get(d, 0.0) + float(amt)
    return out


def _resolve_source(
    source: Union[_HasTable, _HasSnapshot, pa.Table],
    *,
    adjust_for_flows: bool,
    account: str | None = None,
) -> tuple[pa.Table, dict[date, float] | None]:
    """Resolve a DataPort / Result / bare Arrow table to a ``(nav_table, flows)`` pair.

    The shared resolution behind :func:`performance_summary`, :func:`sharpe_ratio`,
    and — via :mod:`ibda.analytics.benchmark`'s ``_aligned_returns`` —
    ``relative_summary``/``rolling_relative``, so all four accept the same source
    union and resolve NAV identically: a DataPort's ``"nav"`` table is snapshotted
    (with flows optionally derived from its ``"cash"`` table), a Result-like object
    is snapshotted directly (no flow adjustment, since there is no cash table to
    derive from), and a bare Arrow table passes through unchanged with
    ``flows=None``. Note that :func:`_resolve_source` itself never filters the NAV
    table by *account* — only ``cash``; NAV selection is the caller's job (see below).

    ``account`` is the same selector the caller separately applies to the NAV table
    via :func:`_select_account` — :func:`compute_performance` on the
    ``performance_summary``/``sharpe_ratio`` path, ``_aligned_returns`` itself on the
    ``relative_summary``/``rolling_relative`` path; when *adjust_for_flows*, the cash
    table is filtered by it too — via :func:`_select_account` again, since ``cash``
    has the same ``"Account"`` column shape as ``nav`` — so a multi-account
    DataPort's flows come from the same account the caller asked for, instead of
    mixing every account's deposits/withdrawals into one series. This mirrors
    :func:`ibda.adapters.ibkr.flex.arrow.performance_from_sections`, which filters
    ``cash_rows`` by account for the same reason. As in that function, the filter is
    only applied when *account* is explicit: with ``account=None`` the cash table is
    passed through unfiltered (matching the Flex path's convention), since a
    single-account book has no ambiguity to resolve, and a genuinely multi-account
    NAV with ``account=None`` is already rejected downstream — in
    :func:`compute_performance` on the ``performance_summary``/``sharpe_ratio`` path,
    or in ``_aligned_returns``'s own :func:`_select_account` call on the
    ``relative_summary``/``rolling_relative`` path — before any (momentarily
    unfiltered) flows would be used.
    """
    flows: dict[date, float] | None = None
    if isinstance(source, pa.Table):
        return source, None
    if isinstance(source, _HasTable):
        try:
            nav = source.table("nav").snapshot()
        except Exception as exc:  # noqa: BLE001 — surface a clear, actionable message
            raise ValueError(
                "no 'nav' table available — the Flex query must include an "
                "EquitySummaryByReportDateInBase section to compute performance."
            ) from exc
        if adjust_for_flows:
            try:
                cash = source.table("cash").snapshot()
                if account is not None:
                    cash = _select_account(cash, account)
                flows = external_flows_from_cash(cash)
            except Exception:  # noqa: BLE001 — flows are optional; proceed without
                flows = None
        return nav, flows
    if isinstance(source, _HasSnapshot):
        return source.snapshot(), None
    raise TypeError(f"unsupported source type for performance_summary: {type(source)!r}")


def performance_summary(
    source: Union[_HasTable, _HasSnapshot, pa.Table],
    *,
    risk_free_annual: str | float = DEFAULT_RISK_FREE_ANNUAL,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    adjust_for_flows: bool = True,
    value_column: str = "Total",
    account: str | None = None,
) -> PerformanceSummary:
    """Compute account performance from a port, a Result, or an Arrow table.

    The high-level entry point. Accepts any of:

    * a :class:`~ibda.port.DataPort` — its ``"nav"`` table is snapshotted, and
      (when ``adjust_for_flows``) deposits/withdrawals are pulled from its
      ``"cash"`` table to strip external flows from returns;
    * a :class:`~ibda.result.Result` (or anything with ``snapshot()``) — used
      directly as the NAV table (no automatic flow adjustment);
    * a bare :class:`pyarrow.Table` of NAV rows.

    Example::

        import ibda
        port = ibda.load_flex_file("reports/activity.xml")
        perf = ibda.performance_summary(port, risk_free_annual=0.05)
        print(perf.sharpe_ratio, perf.max_drawdown)

    Args:
        source: a DataPort, Result, or Arrow NAV table.
        risk_free_annual: annual simple risk-free rate, or ``"auto"`` for the
            documented offline Treasury-yield proxy (:data:`ibda.rates.DEFAULT_RISK_FREE_ANNUAL`).
        periods_per_year: annualization factor (default 252).
        adjust_for_flows: when *source* is a DataPort, derive external flows from
            its cash table so deposits/withdrawals don't count as performance.
        value_column: NAV value column name (default ``"Total"``).
        account: required only if the resolved NAV covers more than one account — the
            account id to analyze. Same convention as
            :func:`ibda.adapters.ibkr.flex.arrow.flex_performance`'s ``account``: a
            multi-account NAV with no *account* given is rejected rather than
            silently picking one (see the ``ValueError`` below).

    Returns:
        A :class:`PerformanceSummary`.

    Raises:
        ValueError: if no usable NAV history is available (e.g. the Flex query did
            not include the daily equity summary, so there is no ``"nav"`` table),
            *risk_free_annual* is a non-numeric, non-``"auto"`` string, the resolved
            NAV covers more than one account and *account* is ``None``, or *account*
            is given but absent (see :func:`_select_account`).
    """
    rf, _rf_source = resolve_risk_free(risk_free_annual)
    nav, flows = _resolve_source(source, adjust_for_flows=adjust_for_flows, account=account)

    return compute_performance(
        nav,
        account=account,
        flows=flows,
        risk_free_annual=rf,
        periods_per_year=periods_per_year,
        value_column=value_column,
    )
