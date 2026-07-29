"""Tests for ibda.analytics.timeseries — rolling + per-period performance."""
from __future__ import annotations

import datetime as dt
from typing import Any, cast

import pyarrow as pa
import pytest

from ibda.analytics.timeseries import periodic_returns, rolling_performance
from ibda.schema import NAV


def _nav_table(rows: list[dict[str, Any]]) -> pa.Table:
    """Build a pyarrow NAV table from canonical nav row dicts (no engine)."""
    cols: dict[str, pa.Array] = {}
    for col in NAV.columns:
        values = [r.get(col.name) for r in rows]
        cols[col.name] = pa.array(values, type=col.dtype.to_arrow())
    return pa.table(cols, schema=NAV.to_arrow_schema())


def _linear_nav(start: dt.date, totals: list[float]) -> pa.Table:
    """One NAV row per consecutive calendar day from *start*."""
    rows = [
        {"Account": "U1", "Timestamp": dt.datetime.combine(
            start + dt.timedelta(days=i), dt.time(0, tzinfo=dt.timezone.utc)),
         "Total": v}
        for i, v in enumerate(totals)
    ]
    return _nav_table(rows)


def test_rolling_performance_columns_and_row_count() -> None:
    nav = _linear_nav(dt.date(2026, 1, 1), [100.0 + i for i in range(10)])
    out = rolling_performance(nav, window=3, periods_per_year=252)
    assert set(out.column_names) == {
        "Timestamp", "Return", "Volatility", "Sharpe", "MaxDrawdown",
    }
    # 10 NAV points -> 9 returns -> 9 - 3 + 1 = 7 full windows.
    assert out.num_rows == 7


def test_rolling_window_larger_than_series_is_empty() -> None:
    nav = _linear_nav(dt.date(2026, 1, 1), [100.0, 101.0, 102.0])
    out = rolling_performance(nav, window=10)
    assert out.num_rows == 0
    assert set(out.column_names) == {
        "Timestamp", "Return", "Volatility", "Sharpe", "MaxDrawdown",
    }


def test_periodic_returns_monthly_buckets() -> None:
    # Two calendar months of daily NAV.
    jan = [100.0 + i for i in range(20)]
    feb = [jan[-1] + i for i in range(1, 11)]
    rows = []
    # Start mid-January so 30 consecutive calendar days span Jan AND Feb
    # (Jan has 31 days, so a Jan-1 start would stay entirely within January).
    d = dt.date(2026, 1, 20)
    for v in jan + feb:
        rows.append({"Account": "U1",
                     "Timestamp": dt.datetime.combine(d, dt.time(0, tzinfo=dt.timezone.utc)),
                     "Total": v})
        d += dt.timedelta(days=1)
    nav = _nav_table(rows)
    out = periodic_returns(nav, freq="monthly")
    assert set(out.column_names) == {"Period", "Start", "End", "Return", "Volatility", "Sharpe"}
    periods = out.column("Period").to_pylist()
    assert periods == ["2026-01", "2026-02"]


def test_periodic_invalid_freq_raises() -> None:
    nav = _linear_nav(dt.date(2026, 1, 1), [100.0, 101.0, 102.0])
    with pytest.raises(ValueError, match="freq"):
        periodic_returns(nav, freq="weekly")


def test_rolling_full_window_reconciles_with_compute_performance() -> None:
    """A single rolling window spanning the whole series equals compute_performance."""
    from ibda.analytics.performance import compute_performance

    nav = _linear_nav(dt.date(2026, 1, 1), [100.0, 102.0, 101.0, 104.0, 103.5, 106.0])
    n_returns = nav.num_rows - 1
    roll = rolling_performance(nav, window=n_returns, risk_free_annual=0.04, periods_per_year=252)
    assert roll.num_rows == 1
    perf = compute_performance(nav, risk_free_annual=0.04, periods_per_year=252)
    row = {k: roll.column(k).to_pylist()[0] for k in roll.column_names}
    assert row["Return"] == pytest.approx(perf.cumulative_return)
    assert row["Volatility"] == pytest.approx(perf.annualized_volatility)
    assert row["Sharpe"] == pytest.approx(perf.sharpe_ratio)
    assert row["MaxDrawdown"] == pytest.approx(perf.max_drawdown)


# --- risk_free_annual="auto" accepted by the library facade ----------------


def test_rolling_performance_accepts_auto_risk_free() -> None:
    nav = _linear_nav(dt.date(2026, 1, 1), [100.0 + i for i in range(10)])
    out = rolling_performance(nav, window=3, risk_free_annual="auto")
    assert out.num_rows == 7


def test_periodic_returns_accepts_auto_risk_free() -> None:
    nav = _linear_nav(dt.date(2026, 1, 1), [100.0 + i for i in range(10)])
    out = periodic_returns(nav, freq="monthly", risk_free_annual="auto")
    assert out.num_rows > 0


# ---------------------------------------------------------------------------
# Multi-account NAV, and returns that do not pair 1:1 with NAV points
# ---------------------------------------------------------------------------
#
# Both defects come from the same place: these two functions were the only members of the
# performance family that neither filtered by account nor asked `_dated_returns` for each
# return's own date. `compute_performance` raises on multi-account NAV and
# `benchmark._aligned_returns` calls `_select_account`; these did neither, which is what
# makes the omission an oversight rather than a design choice.


def _two_account_nav() -> pa.Table:
    """Two accounts, each genuinely +3.0% over the same four days."""
    rows: list[dict[str, Any]] = []
    for acct, base in (("U1", 100_000.0), ("U2", 500_000.0)):
        for i in range(4):
            rows.append({
                "Account": acct,
                "Timestamp": dt.datetime(2026, 1, 1 + i, tzinfo=dt.timezone.utc),
                "Total": base * (1.0 + 0.01 * i),
            })
    rows.sort(key=lambda r: cast("dt.datetime", r["Timestamp"]))
    return _nav_table(rows)


def test_rolling_performance_rejects_ambiguous_multi_account_nav() -> None:
    """Interleaved accounts make every "return" a ratio between DIFFERENT accounts' NAVs.

    Unrejected, that is Return +405% and Volatility ~4397 on a book where both accounts
    were +3.0% — which is why this raises instead.
    """
    with pytest.raises(ValueError, match="multiple accounts"):
        rolling_performance(_two_account_nav(), window=3)


def test_periodic_returns_rejects_ambiguous_multi_account_nav() -> None:
    """Unrejected, the interleaved series reports Return +414.99%, Sharpe 12.03."""
    with pytest.raises(ValueError, match="multiple accounts"):
        periodic_returns(_two_account_nav(), freq="monthly")


def test_account_selection_recovers_the_true_return() -> None:
    """With the account named, both functions report the truth: +3.0%."""
    nav = _two_account_nav()
    assert periodic_returns(nav, freq="monthly", account="U1").to_pydict()["Return"] == [
        pytest.approx(0.03)
    ]
    assert rolling_performance(nav, window=3, account="U1").to_pydict()["Return"] == [
        pytest.approx(0.03)
    ]


def _nav_with_a_zero_point() -> pa.Table:
    """A NAV series whose first point is 0.0 — e.g. a report predating funding.

    `_dated_returns` skips the period whose prior NAV is 0.0, so 4 NAV points yield 2
    returns, not 3 — so pairing them against `timestamps[1:]` positionally misaligns both
    functions.
    """
    return _linear_nav(dt.date(2026, 1, 1), [0.0, 100_000.0, 101_000.0, 102_000.0])


def test_rolling_window_is_labelled_by_its_last_nav_point_despite_a_skipped_period() -> None:
    """The docstring promises the window's LAST NAV timestamp; positional pairing shifted it.

    The single window compounds the returns ending 2026-01-03 and 2026-01-04, so it is
    stamped 2026-01-04. Positional pairing would stamp it 2026-01-03 — one NAV point
    early, and one point earlier again for every additional skipped period.
    """
    out = rolling_performance(_nav_with_a_zero_point(), window=2)
    assert out.num_rows == 1
    assert out.to_pydict()["Timestamp"][0].date() == dt.date(2026, 1, 4)


def test_periodic_returns_does_not_crash_on_a_skipped_period() -> None:
    """Positional pairing raises `ValueError: zip() argument 2 is shorter than argument 1`."""
    out = periodic_returns(_nav_with_a_zero_point(), freq="monthly")
    assert out.num_rows == 1
    # (1.01)(1.0099...) - 1 over the two usable periods.
    assert out.to_pydict()["Return"][0] == pytest.approx(0.02)
