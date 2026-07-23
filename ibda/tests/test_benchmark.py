"""Tests for ibda.analytics.benchmark — relative performance metrics."""
from __future__ import annotations

import datetime as dt
from typing import Any

import pyarrow as pa
import pytest

from ibda.analytics.benchmark import _ib_duration, relative_metrics, relative_summary
from ibda.schema import BAR, NAV


# --- relative_metrics (pure kernel) ----------------------------------------


def test_identity_benchmark_beta_one_alpha_zero() -> None:
    rets = [0.01, -0.004, 0.012, 0.0, -0.003, 0.008, 0.002]
    s = relative_metrics(rets, rets, risk_free_annual=0.0, periods_per_year=252)
    assert s.beta == pytest.approx(1.0)
    assert s.alpha_annualized == pytest.approx(0.0, abs=1e-9)
    assert s.correlation == pytest.approx(1.0)
    assert s.r_squared == pytest.approx(1.0)
    assert s.tracking_error == pytest.approx(0.0, abs=1e-12)
    assert s.up_capture == pytest.approx(1.0)
    assert s.down_capture == pytest.approx(1.0)


def test_levered_benchmark_beta_two() -> None:
    bench = [0.01, -0.004, 0.012, 0.005, -0.003, 0.008]
    port = [2.0 * b for b in bench]
    s = relative_metrics(port, bench, risk_free_annual=0.0, periods_per_year=252)
    assert s.beta == pytest.approx(2.0)
    assert s.correlation == pytest.approx(1.0)


def test_too_few_points_raises() -> None:
    with pytest.raises(ValueError, match="at least two"):
        relative_metrics([0.01], [0.01])


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length"):
        relative_metrics([0.01, 0.02], [0.01])


def test_relative_metrics_render_omits_unset_sentinels() -> None:
    """relative_metrics leaves benchmark_label/start/end unset; render() must not leak
    the "<returns>" label sentinel or the date.min "0001-01-01" placeholder into the
    report — relative_summary is the entry point that fills those in."""
    rets = [0.01, -0.004, 0.012, 0.0, -0.003, 0.008, 0.002]
    s = relative_metrics(rets, rets, risk_free_annual=0.0, periods_per_year=252)
    text = s.render()
    assert "0001-01-01" not in text
    assert "<returns>" not in text
    assert "Beta" in text  # the rest of the report still renders


def test_relative_summary_render_shows_real_label_and_period() -> None:
    """The dated entry point (relative_summary) still prints the real label + period."""
    days = [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(8)]
    navs = [100.0, 101.0, 100.5, 102.0, 101.5, 103.0, 104.0, 103.5]
    nav = _nav_table([{"Account": "U1",
                       "Timestamp": dt.datetime.combine(d, dt.time(0, tzinfo=dt.timezone.utc)),
                       "Total": v} for d, v in zip(days, navs)])
    bench = _bar_table("QQQ", list(zip(days, navs)))
    s = relative_summary(nav, bench, risk_free_annual=0.0)
    text = s.render()
    assert "QQQ" in text
    assert days[1].isoformat() in text and days[-1].isoformat() in text


# --- relative_summary (alignment + source resolution) ------------------


def _nav_table(rows: list[dict[str, Any]]) -> pa.Table:
    cols: dict[str, pa.Array] = {}
    for col in NAV.columns:
        cols[col.name] = pa.array([r.get(col.name) for r in rows], type=col.dtype.to_arrow())
    return pa.table(cols, schema=NAV.to_arrow_schema())


def _bar_table(sym: str, rows: list[tuple[dt.date, float]]) -> pa.Table:
    recs = [{"Sym": sym,
             "Timestamp": dt.datetime.combine(d, dt.time(0, tzinfo=dt.timezone.utc)),
             "Open": v, "High": v, "Low": v, "Close": v, "Volume": 1.0} for d, v in rows]
    cols: dict[str, pa.Array] = {}
    for col in BAR.columns:
        cols[col.name] = pa.array([r.get(col.name) for r in recs], type=col.dtype.to_arrow())
    return pa.table(cols, schema=BAR.to_arrow_schema())


def test_relative_summary_identity_via_tables() -> None:
    days = [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(8)]
    navs = [100.0, 101.0, 100.5, 102.0, 101.5, 103.0, 104.0, 103.5]
    nav = _nav_table([{"Account": "U1",
                       "Timestamp": dt.datetime.combine(d, dt.time(0, tzinfo=dt.timezone.utc)),
                       "Total": v} for d, v in zip(days, navs)])
    bench = _bar_table("QQQ", list(zip(days, navs)))  # same path -> beta 1, alpha 0
    s = relative_summary(nav, bench, risk_free_annual=0.0)
    assert s.benchmark_label == "QQQ"
    assert s.beta == pytest.approx(1.0)
    assert s.alpha_annualized == pytest.approx(0.0, abs=1e-9)
    assert s.num_periods == 7
    assert s.start == days[1] and s.end == days[-1]


def test_relative_summary_drops_unmatched_dates() -> None:
    days = [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(6)]
    nav = _nav_table([{"Account": "U1",
                       "Timestamp": dt.datetime.combine(d, dt.time(0, tzinfo=dt.timezone.utc)),
                       "Total": 100.0 + i} for i, d in enumerate(days)])
    # benchmark missing the last two days
    bench = _bar_table("QQQ", [(d, 50.0 + i) for i, d in enumerate(days[:4])])
    s = relative_summary(nav, bench, risk_free_annual=0.0)
    assert s.dropped_periods >= 1
    assert s.num_periods >= 2


def test_relative_summary_symbol_requires_supervisor() -> None:
    """No offline fallback: a symbol benchmark with no supervisor is a clear ValueError."""
    days = [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(8)]
    navs = [100.0, 101.0, 100.5, 102.0, 101.5, 103.0, 104.0, 103.5]
    nav = _nav_table([{"Account": "U1",
                       "Timestamp": dt.datetime.combine(d, dt.time(0, tzinfo=dt.timezone.utc)),
                       "Total": v} for d, v in zip(days, navs)])
    with pytest.raises(ValueError, match="supervisor"):
        relative_summary(nav, "SPY", risk_free_annual=0.0)


def test_relative_summary_symbol_sources_via_ib(monkeypatch: pytest.MonkeyPatch) -> None:
    """A symbol benchmark is resolved via IB historical daily bars, given a supervisor=."""
    from ibda.adapters.ibkr import marketdata

    days = [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(8)]
    navs = [100.0, 101.0, 100.5, 102.0, 101.5, 103.0, 104.0, 103.5]
    nav = _nav_table([{"Account": "U1",
                       "Timestamp": dt.datetime.combine(d, dt.time(0, tzinfo=dt.timezone.utc)),
                       "Total": v} for d, v in zip(days, navs)])
    fake = _bar_table("SPY", list(zip(days, navs)))
    seen: dict[str, Any] = {}

    def _fake_historical_bars(supervisor: object, symbol: str, **kw: Any) -> pa.Table:
        seen["supervisor"] = supervisor
        seen["symbol"] = symbol
        seen["kw"] = kw
        return fake

    monkeypatch.setattr(marketdata, "historical_bars", _fake_historical_bars)
    sentinel_supervisor = object()
    s = relative_summary(nav, "SPY", risk_free_annual=0.0, supervisor=sentinel_supervisor)
    assert s.benchmark_label == "SPY"
    assert s.beta == pytest.approx(1.0)
    assert seen["supervisor"] is sentinel_supervisor
    assert seen["symbol"] == "SPY"
    assert seen["kw"]["duration"] == "1 Y"  # default benchmark_range="1y" -> IB "1 Y"


# --- _ib_duration: friendly keys + raw IB duration passthrough ---------


@pytest.mark.parametrize(
    "benchmark_range,expected",
    [
        ("1mo", "1 M"), ("3mo", "3 M"), ("6mo", "6 M"),
        ("1y", "1 Y"), ("2y", "2 Y"), ("5y", "5 Y"),
        ("max", "5 Y"),  # documented default -- not a hard ceiling, see raw-duration tests below
    ],
)
def test_ib_duration_friendly_keys_unchanged(benchmark_range: str, expected: str) -> None:
    assert _ib_duration(benchmark_range) == expected


@pytest.mark.parametrize("raw", ["10 Y", "15 Y", "3650 D", "1 W", "18 M", "1 S"])
def test_ib_duration_accepts_raw_ib_duration_string(raw: str) -> None:
    """A raw IB duration string (not a friendly key) passes through unchanged.

    This is how a caller lifts the 5-year `"max"` cap: pass e.g. "10 Y" directly.
    """
    assert _ib_duration(raw) == raw


@pytest.mark.parametrize(
    "raw,expected",
    [("10 y", "10 Y"), ("6y", "6 Y"), ("10Y", "10 Y"), ("18 m", "18 M")],
)
def test_ib_duration_is_case_insensitive_and_normalizes_unit(raw: str, expected: str) -> None:
    """Lower-case / no-space raw durations are accepted (like the friendly keys'
    no-space lowercase convention) and normalized to IB's required uppercase unit."""
    assert _ib_duration(raw) == expected


@pytest.mark.parametrize(
    "bad",
    ["10 years", "Y10", "-5 Y", "10 X", "ten Y", "", "5 YY"],
)
def test_ib_duration_rejects_malformed_raw_string(bad: str) -> None:
    with pytest.raises(ValueError, match="benchmark_range"):
        _ib_duration(bad)


def test_relative_summary_accepts_raw_duration_beyond_five_year_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """benchmark_range="10 Y" reaches historical_bars unchanged, lifting the "max" cap."""
    from ibda.adapters.ibkr import marketdata

    days = [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(8)]
    navs = [100.0, 101.0, 100.5, 102.0, 101.5, 103.0, 104.0, 103.5]
    nav = _nav_table([{"Account": "U1",
                       "Timestamp": dt.datetime.combine(d, dt.time(0, tzinfo=dt.timezone.utc)),
                       "Total": v} for d, v in zip(days, navs)])
    fake = _bar_table("QQQ", list(zip(days, navs)))
    seen: dict[str, Any] = {}

    def _fake_historical_bars(supervisor: object, symbol: str, **kw: Any) -> pa.Table:
        seen["kw"] = kw
        return fake

    monkeypatch.setattr(marketdata, "historical_bars", _fake_historical_bars)
    relative_summary(
        nav, "QQQ", risk_free_annual=0.0, supervisor=object(), benchmark_range="10 Y",
    )
    assert seen["kw"]["duration"] == "10 Y"


def test_relative_summary_rejects_malformed_benchmark_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed benchmark_range raises before ever calling historical_bars."""
    from ibda.adapters.ibkr import marketdata

    days = [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(8)]
    navs = [100.0, 101.0, 100.5, 102.0, 101.5, 103.0, 104.0, 103.5]
    nav = _nav_table([{"Account": "U1",
                       "Timestamp": dt.datetime.combine(d, dt.time(0, tzinfo=dt.timezone.utc)),
                       "Total": v} for d, v in zip(days, navs)])
    called = False

    def _fake_historical_bars(supervisor: object, symbol: str, **kw: Any) -> pa.Table:
        nonlocal called
        called = True
        raise AssertionError("historical_bars must not be called for a malformed duration")

    monkeypatch.setattr(marketdata, "historical_bars", _fake_historical_bars)
    with pytest.raises(ValueError, match="benchmark_range"):
        relative_summary(
            nav, "QQQ", risk_free_annual=0.0, supervisor=object(), benchmark_range="not-a-duration",
        )
    assert not called


def test_rolling_relative_columns_and_beta() -> None:
    days = [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(12)]
    # benchmark price path; portfolio = 2x benchmark daily moves (compounded)
    bench_px = [100.0]
    for r in [0.01, -0.005, 0.012, 0.004, -0.003, 0.008, 0.002, -0.006, 0.009, 0.001, -0.002]:
        bench_px.append(bench_px[-1] * (1 + r))
    port_px = [1_000_000.0]
    for r in [0.01, -0.005, 0.012, 0.004, -0.003, 0.008, 0.002, -0.006, 0.009, 0.001, -0.002]:
        port_px.append(port_px[-1] * (1 + 2.0 * r))
    nav = _nav_table([{"Account": "U1",
                       "Timestamp": dt.datetime.combine(d, dt.time(0, tzinfo=dt.timezone.utc)),
                       "Total": v} for d, v in zip(days, port_px)])
    bench = _bar_table("QQQ", list(zip(days, bench_px)))

    from ibda.analytics.benchmark import rolling_relative
    out = rolling_relative(nav, bench, window=4, risk_free_annual=0.0)
    assert set(out.column_names) == {
        "Timestamp", "Beta", "Alpha", "Correlation", "TrackingError",
    }
    # 12 NAV points -> 11 aligned returns -> 11 - 4 + 1 = 8 windows.
    assert out.num_rows == 8
    # 2x-levered portfolio -> beta ~2 in every window.
    for b in out.column("Beta").to_pylist():
        assert b == pytest.approx(2.0, abs=1e-6)


def test_rolling_relative_window_too_large_is_empty() -> None:
    from ibda.analytics.benchmark import rolling_relative
    days = [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(3)]
    vals = [100.0, 101.0, 102.0]
    nav = _nav_table([{"Account": "U1",
                       "Timestamp": dt.datetime.combine(d, dt.time(0, tzinfo=dt.timezone.utc)),
                       "Total": v} for d, v in zip(days, vals)])
    bench = _bar_table("QQQ", list(zip(days, vals)))
    out = rolling_relative(nav, bench, window=10)
    assert out.num_rows == 0
    assert "Beta" in out.column_names


def test_rolling_relative_window_below_two_raises() -> None:
    from ibda.analytics.benchmark import rolling_relative
    days = [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(4)]
    vals = [100.0, 101.0, 102.0, 103.0]
    nav = _nav_table([{"Account": "U1",
                       "Timestamp": dt.datetime.combine(d, dt.time(0, tzinfo=dt.timezone.utc)),
                       "Total": v} for d, v in zip(days, vals)])
    bench = _bar_table("QQQ", list(zip(days, vals)))
    with pytest.raises(ValueError, match="window"):
        rolling_relative(nav, bench, window=1)


# --- risk_free_annual="auto" accepted by the library facade ----------------


def test_relative_summary_accepts_auto_risk_free() -> None:
    from ibda.rates import DEFAULT_RISK_FREE_ANNUAL

    days = [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(8)]
    navs = [100.0, 101.0, 100.5, 102.0, 101.5, 103.0, 104.0, 103.5]
    nav = _nav_table([{"Account": "U1",
                       "Timestamp": dt.datetime.combine(d, dt.time(0, tzinfo=dt.timezone.utc)),
                       "Total": v} for d, v in zip(days, navs)])
    bench = _bar_table("QQQ", list(zip(days, navs)))
    s = relative_summary(nav, bench, risk_free_annual="auto")
    assert s.risk_free_annual == DEFAULT_RISK_FREE_ANNUAL


def test_rolling_relative_accepts_auto_risk_free() -> None:
    from ibda.analytics.benchmark import rolling_relative

    days = [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(6)]
    vals = [100.0, 101.0, 100.5, 102.0, 101.5, 103.0]
    nav = _nav_table([{"Account": "U1",
                       "Timestamp": dt.datetime.combine(d, dt.time(0, tzinfo=dt.timezone.utc)),
                       "Total": v} for d, v in zip(days, vals)])
    bench = _bar_table("QQQ", list(zip(days, vals)))
    out = rolling_relative(nav, bench, window=3, risk_free_annual="auto")
    assert out.num_rows > 0


# --- account filter (mirrors flex_performance's convention) -----------------


def _multi_account_nav_table() -> tuple[list[dt.date], pa.Table]:
    """A two-account NAV table: U1 tracks the identity path, U2 is twice-levered —
    so filtering to the wrong/no account would visibly change beta."""
    days = [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(8)]
    navs = [100.0, 101.0, 100.5, 102.0, 101.5, 103.0, 104.0, 103.5]
    rows: list[dict[str, Any]] = []
    for d, v in zip(days, navs):
        ts = dt.datetime.combine(d, dt.time(0, tzinfo=dt.timezone.utc))
        rows.append({"Account": "U1", "Timestamp": ts, "Total": v})
        rows.append({"Account": "U2", "Timestamp": ts, "Total": 2.0 * v})
    return days, _nav_table(rows)


def test_relative_summary_multi_account_nav_without_account_raises() -> None:
    """A multi-account NAV with no account= given is rejected outright — mirrors
    flex_performance's behavior for a multi-account Flex report."""
    days, nav = _multi_account_nav_table()
    navs = [100.0, 101.0, 100.5, 102.0, 101.5, 103.0, 104.0, 103.5]
    bench = _bar_table("QQQ", list(zip(days, navs)))
    with pytest.raises(ValueError, match="multiple accounts"):
        relative_summary(nav, bench, risk_free_annual=0.0)


def test_relative_summary_multi_account_nav_with_account_selects_that_account() -> None:
    """account="U1" on a multi-account NAV filters to just that account's rows,
    matching the single-account result computed on the identity path (beta ~1)."""
    days, nav = _multi_account_nav_table()
    navs = [100.0, 101.0, 100.5, 102.0, 101.5, 103.0, 104.0, 103.5]
    bench = _bar_table("QQQ", list(zip(days, navs)))
    s = relative_summary(nav, bench, risk_free_annual=0.0, account="U1")
    assert s.beta == pytest.approx(1.0)


def test_relative_summary_unknown_account_raises() -> None:
    """A named account absent from the NAV is a clear ValueError naming it."""
    days, nav = _multi_account_nav_table()
    navs = [100.0, 101.0, 100.5, 102.0, 101.5, 103.0, 104.0, 103.5]
    bench = _bar_table("QQQ", list(zip(days, navs)))
    with pytest.raises(ValueError, match="U9"):
        relative_summary(nav, bench, risk_free_annual=0.0, account="U9")


def test_rolling_relative_multi_account_nav_requires_account() -> None:
    """rolling_relative applies the same account convention as relative_summary."""
    from ibda.analytics.benchmark import rolling_relative

    days, nav = _multi_account_nav_table()
    navs = [100.0, 101.0, 100.5, 102.0, 101.5, 103.0, 104.0, 103.5]
    bench = _bar_table("QQQ", list(zip(days, navs)))
    with pytest.raises(ValueError, match="multiple accounts"):
        rolling_relative(nav, bench, window=3, risk_free_annual=0.0)

    out = rolling_relative(nav, bench, window=3, risk_free_annual=0.0, account="U1")
    assert out.num_rows > 0


# --- account filter, flows half -- pairs with _multi_account_nav_table() above,
# which covers the NAV half of the same invariant: account filters NAV and flows
# IDENTICALLY. Mirrors ibda/tests/test_performance.py's DataPort cash-flow-filter
# tests, applied to the benchmark path. A DataPort source is required here (not
# the bare pa.Table used above) since _resolve_source only derives/filters flows
# for a _HasTable source -- a bare Arrow table never touches the cash-derivation
# code path this covers.
# -----------------------------------------------------------------------------


def _multi_account_nav_table_with_flows() -> tuple[list[dt.date], pa.Table]:
    """Three NAV points per account (the minimum relative_summary needs: two
    aligned daily returns) -- U1/U2 diverge so mixing flows across accounts, or
    filtering to the wrong one, visibly changes the numbers."""
    days = [dt.date(2026, 6, 1) + dt.timedelta(days=i) for i in range(3)]
    u1 = [100_000.0, 101_000.0, 102_000.0]
    u2 = [200_000.0, 199_000.0, 198_000.0]
    rows: list[dict[str, Any]] = []
    for d, v1, v2 in zip(days, u1, u2):
        ts = dt.datetime.combine(d, dt.time(0, tzinfo=dt.timezone.utc))
        rows.append({"Account": "U1", "Timestamp": ts, "Total": v1})
        rows.append({"Account": "U2", "Timestamp": ts, "Total": v2})
    return days, _nav_table(rows)


def _multi_account_cash_table(day: dt.date) -> pa.Table:
    """Same-date cash flows for two accounts: U1 deposits, U2 withdraws.

    Mirrors ibda/tests/test_performance.py's ``_multi_account_cash_table``
    fixture, used there to prove ``performance_summary``/``sharpe_ratio`` derive
    flows for only the requested account -- applied here to the benchmark path.
    """
    ts = dt.datetime.combine(day, dt.time(0, tzinfo=dt.timezone.utc))
    return pa.table(
        {
            "Account": pa.array(["U1", "U2"], type=pa.string()),
            "Timestamp": pa.array([ts, ts], type=pa.timestamp("ns", tz="UTC")),
            "Type": pa.array(
                ["Deposits/Withdrawals", "Deposits/Withdrawals"], type=pa.string()
            ),
            "Sym": pa.array([None, None], type=pa.string()),
            "Amount": pa.array([5_000.0, -3_000.0], type=pa.float64()),
            "Currency": pa.array(["USD", "USD"], type=pa.string()),
        }
    )


class _FakeResult:
    """Structural ``_HasSnapshot``: wraps a fixed table."""

    def __init__(self, table: pa.Table) -> None:
        self._table = table

    def snapshot(self) -> pa.Table:
        return self._table


class _FakePort:
    """Minimal DataPort-like object: ``table(name).snapshot() -> Arrow``.

    Mirrors ibda/tests/test_performance.py's ``_FakePort`` -- needed to exercise
    ``_resolve_source``'s DataPort branch (a bare ``pa.Table`` source never
    derives flows at all, so it can't exhibit this bug).
    """

    def __init__(self, tables: dict[str, pa.Table]) -> None:
        self._tables = tables

    def table(self, name: str) -> _FakeResult:
        return _FakeResult(self._tables[name])


def test_aligned_returns_port_filters_cash_flows_by_account() -> None:
    """Multi-account DataPort: account="U1" must use only U1's cash flows.

    Regression for the bug where ``_aligned_returns`` called
    ``_resolve_source(portfolio, adjust_for_flows=...)`` WITHOUT ``account=``, so
    cash was never filtered even though NAV was (via ``_select_account`` above) --
    U1's return got corrupted by U2's withdrawal (and vice versa). U1's NAV goes
    100_000 -> 101_000 -> 102_000; U1's own +5_000 deposit lands on day 1, so the
    flow-adjusted return there is (101_000 - 100_000 - 5_000) / 100_000 = -4%. If
    U2's -3_000 withdrawal leaked into U1's flows (net +2_000), the return would
    be -1% instead -- this assertion fails against the unforwarded-account code.
    """
    from ibda.analytics.benchmark import _aligned_returns, _returns_by_date
    from ibda.analytics.performance import _select_account, external_flows_from_cash

    days, nav = _multi_account_nav_table_with_flows()
    cash = _multi_account_cash_table(days[1])
    port = _FakePort({"nav": nav, "cash": cash})
    bench = _bar_table("QQQ", [(d, 50.0 + i) for i, d in enumerate(days)])

    _common1, rp_u1, _rb1, _label1, _dropped1 = _aligned_returns(
        port, bench,
        value_column="Total", benchmark_value_column="Close",
        benchmark_range="1y", adjust_for_flows=True, supervisor=None, account="U1",
    )
    assert rp_u1[0] == pytest.approx(-0.04)
    assert rp_u1[1] == pytest.approx((102_000.0 - 101_000.0) / 101_000.0)

    _common2, rp_u2, _rb2, _label2, _dropped2 = _aligned_returns(
        port, bench,
        value_column="Total", benchmark_value_column="Close",
        benchmark_range="1y", adjust_for_flows=True, supervisor=None, account="U2",
    )
    # U2's NAV goes 200_000 -> 199_000 -> 198_000; U2's own -3_000 withdrawal makes
    # day 1's adjusted return (199_000 - 200_000 + 3_000) / 200_000 = +1%.
    assert rp_u2[0] == pytest.approx(0.01)
    assert rp_u2[1] == pytest.approx((198_000.0 - 199_000.0) / 199_000.0)

    # Reproduce the pre-fix (mixed-account) result directly: deriving flows from
    # the UNFILTERED cash table (both accounts' amounts summed per date, net
    # +2_000) and applying that to U1's own NAV series gives a different, wrong
    # answer -- proving the account filter is load-bearing, not cosmetic.
    mixed_flows = external_flows_from_cash(cash)
    u1_nav_only = _select_account(nav, "U1")
    mixed = _returns_by_date(u1_nav_only, "Total", mixed_flows)
    assert mixed[days[1]] == pytest.approx(-0.01)
    assert mixed[days[1]] != pytest.approx(rp_u1[0])


def test_relative_summary_port_filters_cash_flows_by_account() -> None:
    """Public-API regression: relative_summary(port, ..., account="U1") on a
    multi-account DataPort must not let U2's cash flows corrupt U1's return
    series -- the exact failure mode reported (a spurious return silently
    corrupting beta/alpha/tracking-error/capture)."""
    from ibda.analytics.performance import _annualized_return_from_returns

    days, nav = _multi_account_nav_table_with_flows()
    cash = _multi_account_cash_table(days[1])
    port = _FakePort({"nav": nav, "cash": cash})
    bench = _bar_table("QQQ", [(d, 100.0 + i) for i, d in enumerate(days)])

    s_u1 = relative_summary(port, bench, risk_free_annual=0.0, account="U1")
    expected_rp_u1 = [-0.04, (102_000.0 - 101_000.0) / 101_000.0]
    assert s_u1.portfolio_annualized_return == pytest.approx(
        _annualized_return_from_returns(expected_rp_u1, periods_per_year=252)
    )

    s_u2 = relative_summary(port, bench, risk_free_annual=0.0, account="U2")
    expected_rp_u2 = [0.01, (198_000.0 - 199_000.0) / 199_000.0]
    assert s_u2.portfolio_annualized_return == pytest.approx(
        _annualized_return_from_returns(expected_rp_u2, periods_per_year=252)
    )


def test_aligned_returns_single_account_port_unaffected_by_account_param() -> None:
    """Regression: a single-account DataPort with cash flows -- passing account=
    explicitly or omitting it gives the same, correctly flow-adjusted result."""
    from ibda.analytics.benchmark import _aligned_returns

    days = [dt.date(2026, 6, 1) + dt.timedelta(days=i) for i in range(3)]
    values = [100_000.0, 150_000.0, 150_000.0]
    rows = [
        {"Account": "U1",
         "Timestamp": dt.datetime.combine(d, dt.time(0, tzinfo=dt.timezone.utc)),
         "Total": v} for d, v in zip(days, values)
    ]
    nav = _nav_table(rows)
    ts = dt.datetime.combine(days[1], dt.time(0, tzinfo=dt.timezone.utc))
    cash = pa.table(
        {
            "Account": pa.array(["U1"], type=pa.string()),
            "Timestamp": pa.array([ts], type=pa.timestamp("ns", tz="UTC")),
            "Type": pa.array(["Deposits/Withdrawals"], type=pa.string()),
            "Sym": pa.array([None], type=pa.string()),
            "Amount": pa.array([50_000.0], type=pa.float64()),
            "Currency": pa.array(["USD"], type=pa.string()),
        }
    )
    port = _FakePort({"nav": nav, "cash": cash})
    bench = _bar_table("QQQ", [(d, 50.0 + i) for i, d in enumerate(days)])

    common_none, rp_none, _rb1, _l1, _d1 = _aligned_returns(
        port, bench, value_column="Total", benchmark_value_column="Close",
        benchmark_range="1y", adjust_for_flows=True, supervisor=None, account=None,
    )
    common_u1, rp_u1, _rb2, _l2, _d2 = _aligned_returns(
        port, bench, value_column="Total", benchmark_value_column="Close",
        benchmark_range="1y", adjust_for_flows=True, supervisor=None, account="U1",
    )
    assert common_none == common_u1
    assert rp_none == pytest.approx(rp_u1)
    # The 50_000 deposit exactly offsets the 50_000 NAV jump -> zero adjusted return.
    assert rp_u1[0] == pytest.approx(0.0)
    assert rp_u1[1] == pytest.approx(0.0)


# --- duplicate-date guard (_returns_by_date) --------------------------------


def test_relative_summary_duplicate_date_raises() -> None:
    """A duplicate calendar date in the (already account-filtered) NAV series is
    genuinely ambiguous input — reject it with a clear error naming the date,
    rather than silently collapsing it (last-value-wins) as a plain dict() would."""
    days = [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(5)]
    navs = [100.0, 101.0, 100.5, 102.0, 101.5]
    rows = [{"Account": "U1",
             "Timestamp": dt.datetime.combine(d, dt.time(0, tzinfo=dt.timezone.utc)),
             "Total": v} for d, v in zip(days, navs)]
    # Duplicate the last day's row under the SAME account — same account filter
    # can't disambiguate this; only the duplicate-date guard catches it.
    rows.append({"Account": "U1",
                 "Timestamp": dt.datetime.combine(days[-1], dt.time(0, tzinfo=dt.timezone.utc)),
                 "Total": navs[-1] + 5.0})
    nav = _nav_table(rows)
    bench = _bar_table("QQQ", list(zip(days, navs)))
    with pytest.raises(ValueError, match=f"duplicate return date {days[-1].isoformat()}"):
        relative_summary(nav, bench, risk_free_annual=0.0)
