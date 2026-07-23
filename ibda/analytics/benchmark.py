"""ibda.analytics.benchmark — performance relative to a user-chosen benchmark.

Aligns the account's daily NAV returns with a benchmark return series (any benchmark the
user picks — a symbol resolved via IB historical daily bars, or a price table they bring)
and computes the standard relative metrics: beta, annualized alpha (CAPM), correlation, R²,
tracking error, information ratio, and up/down capture.

**IB-only connectivity.** When *benchmark* is a symbol string, its daily closes are fetched
via :func:`ibda.adapters.ibkr.marketdata.historical_bars` (IB historical daily bars) — this
requires a connected ``supervisor=``. There is no offline/public fallback: pass a benchmark
price Arrow table directly (the ``pa.Table`` branch) for engine-free, no-connection use.

Pure module: stdlib + pyarrow. No engine in the import path — the IB-backed symbol lookup
lazily imports the vendor-confined market-data adapter only when a symbol string (not a
table) is passed as *benchmark*.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, time, timezone
from typing import Any, Union, cast

import pyarrow as pa

from ibda.analytics.performance import (
    _annualized_return_from_returns,
    _dated_returns,
    _HasSnapshot,
    _HasTable,
    _resolve_source,
    _sample_stdev,
    _select_account,
)
from ibda.rates import DEFAULT_PERIODS_PER_YEAR, DEFAULT_RISK_FREE_ANNUAL, resolve_risk_free

_Portfolio = Union[_HasTable, _HasSnapshot, pa.Table]

# Public-facing ``benchmark_range`` strings (kept in the pre-existing, human-friendly
# Yahoo-style vocabulary) mapped to the IB ``duration`` string
# :func:`ibda.adapters.ibkr.marketdata.historical_bars` expects. IB has no literal "max";
# ``"5 Y"`` is used as a practical default for that case — NOT a hard ceiling. Pass a raw
# IB duration string (e.g. ``"10 Y"``, ``"3650 D"``) directly to look back further.
_RANGE_TO_IB_DURATION: dict[str, str] = {
    "1mo": "1 M", "3mo": "3 M", "6mo": "6 M",
    "1y": "1 Y", "2y": "2 Y", "5y": "5 Y", "max": "5 Y",
}

# IB's historical-data ``duration`` grammar: an integer count followed by an optional
# space and exactly one unit letter — S(econds), D(ays), W(eeks), M(onths), Y(ears).
# See IB's reqHistoricalData docs. Used to validate a raw duration string passed
# through ``benchmark_range`` (any value that isn't one of the friendly keys above).
# Matched case-insensitively and with the space optional (``"10 y"``, ``"6y"``,
# ``"10Y"`` are all accepted, like the friendly keys' no-space lowercase convention,
# e.g. ``"1y"``); the matched unit is upper-cased before being handed to IB, which
# requires an uppercase unit letter.
_IB_DURATION_RE = re.compile(r"^(\d+)\s*([SDWMY])$", re.IGNORECASE)


def _ib_duration(benchmark_range: str) -> str:
    """Map a ``benchmark_range`` string to an IB ``duration`` string.

    Recognized short forms (``"1mo"``, ``"1y"``, ``"max"``, …) are translated via
    :data:`_RANGE_TO_IB_DURATION`. ``"max"`` maps to ``"5 Y"`` — a default, not a hard
    ceiling: pass a raw IB duration string directly (e.g. ``"10 Y"``, ``"15 Y"``,
    ``"3650 D"``) to look back further than 5 years for a true since-inception beta.

    Any value that isn't a known short form is validated against IB's duration grammar
    (``<int> <S|D|W|M|Y>``, e.g. ``"18 M"``), matched case-insensitively (``"10 y"``,
    ``"6y"`` are accepted), and normalized to ``"<int> <UPPER-UNIT>"`` (IB's
    ``reqHistoricalData`` requires an uppercase unit letter).

    Raises:
        ValueError: *benchmark_range* is neither a known short form nor a
            well-formed raw IB duration string.
    """
    key = benchmark_range.strip().lower()
    if key in _RANGE_TO_IB_DURATION:
        return _RANGE_TO_IB_DURATION[key]
    raw = benchmark_range.strip()
    match = _IB_DURATION_RE.match(raw)
    if not match:
        raise ValueError(
            f"benchmark_range={benchmark_range!r} is neither a known short form "
            f"({sorted(_RANGE_TO_IB_DURATION)}) nor a well-formed IB duration string "
            "('<int> <S|D|W|M|Y>', e.g. \"10 Y\", \"3650 D\")."
        )
    count, unit = match.group(1), match.group(2)
    return f"{count} {unit.upper()}"


@dataclass(frozen=True)
class RelativeSummary:
    """Portfolio performance relative to a chosen benchmark. Ratios annualized.

    Note: :func:`relative_metrics` — the undated pure kernel — leaves ``benchmark_label``
    at the sentinel ``"<returns>"`` and ``start``/``end`` at ``date.min``; it has no
    calendar to attach to a bare pair of return series. Only :func:`relative_summary`
    (and, per-window, :func:`rolling_relative`) fill those fields in with the real
    label and date range. :meth:`render` detects and omits the sentinel values.
    """

    benchmark_label: str
    num_periods: int
    dropped_periods: int
    start: date
    end: date
    beta: float
    alpha_annualized: float
    correlation: float
    r_squared: float
    tracking_error: float
    information_ratio: float
    up_capture: float
    down_capture: float
    portfolio_annualized_return: float
    benchmark_annualized_return: float
    risk_free_annual: float
    periods_per_year: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict (dates rendered as ISO-8601 strings)."""
        out: dict[str, Any] = asdict(self)
        out["start"] = self.start.isoformat()
        out["end"] = self.end.isoformat()
        return out

    def render(self) -> str:
        """Render a plain-text report with a one-paragraph novice glossary.

        When produced by :func:`relative_metrics` (the undated pure kernel),
        ``benchmark_label``/``start``/``end`` are unset sentinels (see the class
        docstring); the title and calendar line are omitted in that case rather
        than printing ``"<returns>"`` or ``0001-01-01``.
        """

        def pct(x: float) -> str:
            return "n/a" if math.isnan(x) else f"{x * 100:+.2f}%"

        def num(x: float) -> str:
            return "n/a" if math.isnan(x) else f"{x:.3f}"

        dated = self.start != date.min and self.end != date.min
        labeled = self.benchmark_label != "<returns>"

        lines: list[str] = []
        lines.append(f"Performance vs {self.benchmark_label}" if labeled else "Performance")
        if dated:
            lines.append(
                f"  Period              {self.start.isoformat()} → {self.end.isoformat()}"
                f"   ({self.num_periods} days, {self.dropped_periods} unmatched)"
            )
        lines += [
            f"  Beta                {num(self.beta)}",
            f"  Alpha (annualized)  {pct(self.alpha_annualized)}",
            f"  Correlation / R²    {num(self.correlation)} / {num(self.r_squared)}",
            f"  Tracking error      {pct(self.tracking_error)}",
            f"  Information ratio   {num(self.information_ratio)}",
            f"  Up / down capture   {num(self.up_capture)} / {num(self.down_capture)}",
            f"  Return  port/bench  {pct(self.portfolio_annualized_return)}"
            f" / {pct(self.benchmark_annualized_return)}",
            "",
            "  Beta = sensitivity to the benchmark (1 = moves with it). Alpha = annualized",
            "  excess return the benchmark doesn't explain. Tracking error = volatility of",
            "  the return difference; information ratio = alpha per unit of it. Capture =",
            "  share of benchmark up/down moves the portfolio caught (1.0 = matched).",
        ]
        return "\n".join(lines)


def _capture(port: list[float], bench: list[float], *, up: bool) -> float:
    idx = [i for i, b in enumerate(bench) if (b > 0.0 if up else b < 0.0)]
    if not idx:
        return math.nan
    pg = math.prod(1.0 + port[i] for i in idx) - 1.0
    bg = math.prod(1.0 + bench[i] for i in idx) - 1.0
    return pg / bg if bg != 0.0 else math.nan


def relative_metrics(
    portfolio_returns: list[float],
    benchmark_returns: list[float],
    *,
    risk_free_annual: float = DEFAULT_RISK_FREE_ANNUAL,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> RelativeSummary:
    """Relative metrics from two ALREADY-ALIGNED, equal-length daily return series.

    Beta/alpha are computed on excess returns (CAPM); correlation/R² on raw returns;
    tracking error and information ratio on the active return (portfolio − benchmark).
    ``start``/``end`` are left at ``date.min`` here — :func:`relative_summary` (which
    knows the calendar) fills them in. Callers wanting dated output use that entry point.

    This is the pure numeric kernel (mirrors :func:`ibda.analytics.performance.compute_performance`):
    ``risk_free_annual`` is float-only here — callers wanting the ``"auto"`` sentinel use
    :func:`relative_summary` or :func:`rolling_relative`.

    Raises:
        ValueError: if the series differ in length or have fewer than two points.
    """
    rp = list(portfolio_returns)
    rb = list(benchmark_returns)
    if len(rp) != len(rb):
        raise ValueError(f"portfolio/benchmark return length mismatch: {len(rp)} vs {len(rb)}")
    n = len(rp)
    if n < 2:
        raise ValueError(f"need at least two aligned returns; got {n}")

    rf = risk_free_annual / periods_per_year
    ep = [r - rf for r in rp]
    eb = [r - rf for r in rb]
    mean_ep = math.fsum(ep) / n
    mean_eb = math.fsum(eb) / n
    var_eb = math.fsum((b - mean_eb) ** 2 for b in eb) / (n - 1)
    cov = math.fsum((ep[i] - mean_ep) * (eb[i] - mean_eb) for i in range(n)) / (n - 1)
    beta = cov / var_eb if var_eb > 0.0 else math.nan
    alpha_annualized = (
        (mean_ep - beta * mean_eb) * periods_per_year if not math.isnan(beta) else math.nan
    )

    sp = _sample_stdev(rp)
    sb = _sample_stdev(rb)
    mp = math.fsum(rp) / n
    mb = math.fsum(rb) / n
    cov_raw = math.fsum((rp[i] - mp) * (rb[i] - mb) for i in range(n)) / (n - 1)
    correlation = (
        cov_raw / (sp * sb)
        if not math.isnan(sp) and not math.isnan(sb) and sp > 0.0 and sb > 0.0
        else math.nan
    )
    r_squared = correlation ** 2 if not math.isnan(correlation) else math.nan

    active = [rp[i] - rb[i] for i in range(n)]
    te_per = _sample_stdev(active)
    tracking_error = te_per * math.sqrt(periods_per_year) if not math.isnan(te_per) else math.nan
    information_ratio = (
        (math.fsum(active) / n) / te_per * math.sqrt(periods_per_year)
        if not math.isnan(te_per) and te_per > 0.0 else math.nan
    )

    return RelativeSummary(
        benchmark_label="<returns>",
        num_periods=n,
        dropped_periods=0,
        start=date.min,
        end=date.min,
        beta=beta,
        alpha_annualized=alpha_annualized,
        correlation=correlation,
        r_squared=r_squared,
        tracking_error=tracking_error,
        information_ratio=information_ratio,
        up_capture=_capture(rp, rb, up=True),
        down_capture=_capture(rp, rb, up=False),
        portfolio_annualized_return=_annualized_return_from_returns(
            rp, periods_per_year=periods_per_year
        ),
        benchmark_annualized_return=_annualized_return_from_returns(
            rb, periods_per_year=periods_per_year
        ),
        risk_free_annual=risk_free_annual,
        periods_per_year=periods_per_year,
    )


def _returns_by_date(
    table: pa.Table,
    value_column: str,
    flows: Mapping[date, float] | None = None,
) -> dict[date, float]:
    """Per-date daily returns from a price/NAV table (keyed by the later date).

    Thin wrapper over :func:`performance._dated_returns` — the same primitive
    :func:`performance.daily_returns` uses — so the return formula lives in one place.

    Raises:
        ValueError: two rows produce the same return date. This is not a normal
            single-account daily series — a duplicate date signals genuinely
            ambiguous input (e.g. an unfiltered multi-account NAV producing
            multiple rows per calendar date), so it is rejected rather than
            silently collapsed by last-value-wins.
    """
    out: dict[date, float] = {}
    for d, r in _dated_returns(table, value_column=value_column, flows=flows):
        if d in out:
            raise ValueError(
                f"duplicate return date {d.isoformat()} — ambiguous input (e.g. an "
                "unfiltered multi-account NAV producing more than one row per date); "
                "pass account=... to relative_summary/rolling_relative to disambiguate, "
                "or ensure the source has exactly one row per calendar date."
            )
        out[d] = r
    return out


def _aligned_returns(
    portfolio: _Portfolio,
    benchmark: pa.Table | str,
    *,
    value_column: str,
    benchmark_value_column: str,
    benchmark_range: str,
    adjust_for_flows: bool,
    supervisor: object | None,
    account: str | None = None,
) -> tuple[list[date], list[float], list[float], str, int]:
    """Resolve portfolio + benchmark to date-aligned daily returns.

    Returns ``(common_dates, portfolio_returns, benchmark_returns, label, dropped)`` where
    returns are computed per series then inner-joined on calendar date (a gap never fabricates
    a multi-day return). ``dropped`` is the count of per-series return-dates with no match.

    Raises:
        ValueError: see :func:`_select_account` (multi-account NAV with no *account*
            given, or *account* given but absent) and :func:`_returns_by_date`
            (duplicate return date in either series).
    """
    # account= is forwarded to _resolve_source so a DataPort's derived flows come from
    # the SAME account as the NAV we're about to select — otherwise (the pre-fix bug)
    # flows would be the union of every account's deposits/withdrawals while NAV was
    # filtered to one. _resolve_source itself never filters NAV (only cash), so the
    # explicit _select_account call below remains the sole place NAV gets filtered —
    # no double-filtering, and _select_account's ValueError cases are unaffected.
    nav_table, flows = _resolve_source(
        portfolio, adjust_for_flows=adjust_for_flows, account=account
    )
    nav_table = _select_account(nav_table, account)

    label: str
    if isinstance(benchmark, str):
        label = benchmark.strip().upper()
        if supervisor is None:
            raise ValueError(
                "relative_summary/rolling_relative require a connected supervisor= to "
                "fetch IB historical daily bars for a benchmark symbol (there is no offline "
                "fallback); pass a pa.Table of benchmark prices directly for engine-free use, "
                "or supervisor=<connected IbkrSupervisor>."
            )
        from ibda.adapters.ibkr.marketdata import (  # noqa: PLC0415 — lazy: vendor path
            historical_bars,
        )

        # keep_up_to_date=False: this is a one-shot snapshot, not a live feed.
        # historical_bars defaults to True (deephaven-ib's own default, preserved for
        # callers elsewhere that genuinely want a ticking line, e.g. a live VaR feed),
        # and this call is NOT memoized — so every relative_summary/rolling_relative
        # invocation was opening a fresh, never-released reqHistoricalData subscription,
        # accumulating toward IBKR's ~50-concurrent-historical cap.
        # False is safe here and strictly better: it returns the SAME rows (measured
        # live on 2026-07-09 — see historical_bars' docstring for the trial detail),
        # and since the fetch happens per call, "frozen at fetch time" is exactly as
        # fresh as the analytic that requested it.
        bench_table = historical_bars(
            supervisor,  # type: ignore[arg-type]
            label,
            duration=_ib_duration(benchmark_range),
            keep_up_to_date=False,
        )
    else:
        bench_table = benchmark
        syms = (
            [s for s in cast("list[str | None]", bench_table.column("Sym").to_pylist()) if s]
            if "Sym" in bench_table.column_names
            else []
        )
        label = syms[0] if syms else "<custom>"

    p_by_date = _returns_by_date(nav_table, value_column, flows)
    b_by_date = _returns_by_date(bench_table, benchmark_value_column)
    common = sorted(set(p_by_date) & set(b_by_date))
    rp = [p_by_date[d] for d in common]
    rb = [b_by_date[d] for d in common]
    dropped = (len(p_by_date) - len(common)) + (len(b_by_date) - len(common))
    return common, rp, rb, label, dropped


def relative_summary(
    portfolio: _Portfolio,
    benchmark: pa.Table | str,
    *,
    risk_free_annual: str | float = DEFAULT_RISK_FREE_ANNUAL,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    value_column: str = "Total",
    benchmark_value_column: str = "Close",
    benchmark_range: str = "1y",
    adjust_for_flows: bool = True,
    supervisor: object | None = None,
    account: str | None = None,
) -> RelativeSummary:
    """Compute benchmark-relative performance for *portfolio* vs *benchmark*.

    Args:
        portfolio: a DataPort / Result / NAV Arrow table (NAV-to-NAV returns, flow-stripped
            when a DataPort).
        benchmark: a price Arrow table (``Timestamp`` + *benchmark_value_column*) or a symbol
            string. When a symbol: daily closes are fetched via IB historical bars
            (:func:`ibda.adapters.ibkr.marketdata.historical_bars`), which requires a connected
            *supervisor* and a market-data entitlement — there is no offline/public fallback.
            Pass a benchmark price table directly to compute this engine-free with no connection.
        risk_free_annual: annual simple risk-free rate, or ``"auto"`` for the documented
            offline Treasury-yield proxy (:data:`ibda.rates.DEFAULT_RISK_FREE_ANNUAL`).
        periods_per_year: annualization factor (default 252 trading days).
        value_column: portfolio NAV value column name.
        benchmark_value_column: benchmark price column name.
        benchmark_range: benchmark history length when *benchmark* is a symbol — a friendly
            key (``"1mo"``, ``"1y"``, ``"max"``, …; see :data:`_RANGE_TO_IB_DURATION`) or a
            raw IB duration string (e.g. ``"10 Y"``, ``"3650 D"``). ``"max"`` is ``"5 Y"``,
            a default not a hard ceiling — pass a raw duration directly for a longer
            since-inception window.
        adjust_for_flows: derive external flows from a DataPort's cash table.
        supervisor: a connected ``IbkrSupervisor``, required when *benchmark* is a symbol.
        account: required only if the resolved NAV covers more than one account — the
            account id to analyze. Both the NAV rows and, when *portfolio* is a
            DataPort and *adjust_for_flows*, the external flows derived from its
            ``"cash"`` table are filtered to this account — the same convention as
            :func:`ibda.adapters.ibkr.flex.arrow.flex_performance`'s ``account``,
            which filters NAV and cash rows to one account rather than mixing every
            account's deposits/withdrawals into the series. A multi-account NAV with
            no *account* given is rejected rather than silently mixed together (see
            the ``ValueError`` below).

    Returns are computed per series then inner-joined on calendar date, so a gap never
    fabricates a multi-day return; unmatched dates are counted in ``dropped_periods``.

    Returns:
        A :class:`RelativeSummary`.

    Raises:
        ValueError: fewer than two overlapping dates, a missing symbol/series, *benchmark*
            is a symbol string and *supervisor* is ``None``, *risk_free_annual* is a
            non-numeric, non-``"auto"`` string, *benchmark_range* is neither a known
            short form nor a well-formed IB duration string, the resolved NAV covers more
            than one account and *account* is ``None``, *account* is given but absent from
            the NAV, or either return series has a duplicate return date.
    """
    rf, _rf_source = resolve_risk_free(risk_free_annual)
    common, rp, rb, label, dropped = _aligned_returns(
        portfolio, benchmark,
        value_column=value_column, benchmark_value_column=benchmark_value_column,
        benchmark_range=benchmark_range, adjust_for_flows=adjust_for_flows,
        supervisor=supervisor, account=account,
    )
    if len(common) < 2:
        raise ValueError(
            f"fewer than two overlapping dates between portfolio and {label} "
            f"(overlap {len(common)})"
        )
    base = relative_metrics(rp, rb, risk_free_annual=rf, periods_per_year=periods_per_year)
    return replace(
        base, benchmark_label=label, dropped_periods=dropped, start=common[0], end=common[-1]
    )


def rolling_relative(
    portfolio: _Portfolio,
    benchmark: pa.Table | str,
    *,
    window: int = 63,
    risk_free_annual: str | float = DEFAULT_RISK_FREE_ANNUAL,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    value_column: str = "Total",
    benchmark_value_column: str = "Close",
    benchmark_range: str = "1y",
    adjust_for_flows: bool = True,
    supervisor: object | None = None,
    account: str | None = None,
) -> pa.Table:
    """Rolling (time-varying) relative metrics over a trailing *window* of trading days.

    Args:
        portfolio: see :func:`relative_summary`.
        benchmark: see :func:`relative_summary`.
        window: trailing window length in trading days (number of aligned returns per window).
        risk_free_annual: annual simple risk-free rate, or ``"auto"`` for the documented
            offline Treasury-yield proxy (:data:`ibda.rates.DEFAULT_RISK_FREE_ANNUAL`).
        periods_per_year: annualization factor (default 252 trading days).
        value_column: portfolio NAV value column name.
        benchmark_value_column: benchmark price column name.
        benchmark_range: benchmark history length when *benchmark* is a symbol — a friendly
            key or a raw IB duration string; see :func:`relative_summary`.
        adjust_for_flows: derive external flows from a DataPort's cash table.
        supervisor: a connected ``IbkrSupervisor``, required when *benchmark* is a symbol.
        account: required only if the resolved NAV covers more than one account; see
            :func:`relative_summary`'s ``account`` for the exact convention.

    Returns:
        One row per full window, labelled by the window's last aligned date. Columns:
        ``Timestamp`` (UTC), ``Beta``, ``Alpha`` (annualized), ``Correlation``,
        ``TrackingError`` (annualized). Same source-agnostic inputs as
        :func:`relative_summary`; output is plain Arrow (like ``indicators`` /
        ``timeseries``). Fewer than *window* aligned returns -> a typed zero-row table.

    Raises:
        ValueError: if ``window < 2``, *benchmark* is a symbol string and *supervisor*
            is ``None``, *risk_free_annual* is a non-numeric, non-``"auto"`` string,
            *benchmark_range* is neither a known short form nor a well-formed IB
            duration string, the resolved NAV covers more than one account and
            *account* is ``None``, *account* is given but absent from the NAV, or
            either return series has a duplicate return date.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2 to define a relative metric; got {window}")
    rf, _rf_source = resolve_risk_free(risk_free_annual)
    common, rp, rb, _label, _dropped = _aligned_returns(
        portfolio, benchmark,
        value_column=value_column, benchmark_value_column=benchmark_value_column,
        benchmark_range=benchmark_range, adjust_for_flows=adjust_for_flows,
        supervisor=supervisor, account=account,
    )

    out_ts: list[datetime] = []
    betas: list[float] = []
    alphas: list[float] = []
    corrs: list[float] = []
    tes: list[float] = []
    for end in range(window - 1, len(rp)):
        lo = end - window + 1
        m = relative_metrics(
            rp[lo : end + 1], rb[lo : end + 1],
            risk_free_annual=rf, periods_per_year=periods_per_year,
        )
        out_ts.append(datetime.combine(common[end], time(0, tzinfo=timezone.utc)))
        betas.append(m.beta)
        alphas.append(m.alpha_annualized)
        corrs.append(m.correlation)
        tes.append(m.tracking_error)

    return cast(pa.Table, pa.table({
        "Timestamp": pa.array(out_ts, type=pa.timestamp("ns", tz="UTC")),
        "Beta": pa.array(betas, type=pa.float64()),
        "Alpha": pa.array(alphas, type=pa.float64()),
        "Correlation": pa.array(corrs, type=pa.float64()),
        "TrackingError": pa.array(tes, type=pa.float64()),
    }))
