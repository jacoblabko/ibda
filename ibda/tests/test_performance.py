"""Tests for ibda.analytics.performance and the Flex nav-section pipeline.

All offline / no-engine: NAV Arrow tables are built directly with pyarrow from
the canonical NAV schema, so these run without a Deephaven JVM. They cover:

* parse_statement extracting the daily equity summary into a 'nav' section;
* flex_sections_to_canonical mapping nav rows to canonical columns;
* compute_performance / sharpe_ratio math (returns, Sharpe, drawdown);
* external-flow stripping;
* performance_summary dispatch over Result-like and DataPort-like sources.
"""
from __future__ import annotations

import math
import statistics
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from ibda.adapters.ibkr.flex.mapping import flex_sections_to_canonical
from ibda.adapters.ibkr.flex.parse import parse_statement
from ibda.analytics.performance import (
    compute_performance,
    daily_returns,
    external_flows_from_cash,
    performance_summary,
    sharpe_ratio,
)
from ibda.schema import NAV

_FIXTURE = Path(__file__).parent / "fixtures" / "flex" / "report_full.xml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sections() -> dict[str, Any]:
    parsed = parse_statement(_FIXTURE.read_text())
    assert parsed["status"] == "ok"
    return dict(parsed["sections"])


def _nav_table(rows: list[dict[str, Any]]) -> pa.Table:
    """Build a pyarrow NAV table from canonical nav row dicts (no engine)."""
    cols: dict[str, pa.Array] = {}
    for col in NAV.columns:
        values = [r.get(col.name) for r in rows]
        cols[col.name] = pa.array(values, type=col.dtype.to_arrow())
    return pa.table(cols, schema=NAV.to_arrow_schema())


def _nav_from_series(
    points: list[tuple[datetime, float]],
) -> pa.Table:
    rows = [
        {"Account": "U1", "Timestamp": ts, "Total": v, "Cash": None, "Stock": None}
        for ts, v in points
    ]
    return _nav_table(rows)


# ---------------------------------------------------------------------------
# Flex parse + mapping of the daily equity summary
# ---------------------------------------------------------------------------


def test_parse_extracts_nav_series() -> None:
    nav = _sections()["nav"]
    assert len(nav) == 6  # fixture has six daily equity rows
    first = nav[0]
    # reportDate compact YYYYMMDD must be normalized to dashed form
    assert first["report_date"] == "2026-06-01"
    assert first["total"] == 1000000.00
    assert nav[-1]["total"] == 1004580.00


def test_mapping_nav_rows_conform_to_schema() -> None:
    canon = flex_sections_to_canonical(_sections())
    nav = canon["nav"]
    assert len(nav) == 6
    cols = set(NAV.column_names)
    for row in nav:
        assert set(row).issubset(cols)
        assert isinstance(row["Total"], float)
        assert isinstance(row["Timestamp"], datetime)
        assert row["Timestamp"].tzinfo == timezone.utc
    # 2026-06-01 ET midnight (EDT, UTC-4) -> 2026-06-01T04:00Z
    assert nav[0]["Timestamp"] == datetime(2026, 6, 1, 4, 0, 0, tzinfo=timezone.utc)


def test_nav_table_validates_against_schema() -> None:
    canon = flex_sections_to_canonical(_sections())
    table = _nav_table(canon["nav"])
    NAV.validate(table)  # raises on mismatch
    assert table.num_rows == 6


# ---------------------------------------------------------------------------
# Returns + performance math
# ---------------------------------------------------------------------------


def test_returns_telescope_to_total_growth_without_flows() -> None:
    canon = flex_sections_to_canonical(_sections())
    table = _nav_table(canon["nav"])
    rets = daily_returns(table)
    assert len(rets) == 5  # 6 NAV points -> 5 returns
    growth = math.prod(1.0 + r for r in rets)
    # With no external flows the product telescopes to last/first NAV.
    assert growth == pytest.approx(1004580.0 / 1000000.0)


def test_compute_performance_matches_independent_formula() -> None:
    canon = flex_sections_to_canonical(_sections())
    table = _nav_table(canon["nav"])
    perf = compute_performance(table, risk_free_annual=0.0, periods_per_year=252)

    rets = daily_returns(table)
    exp_sharpe = statistics.fmean(rets) / statistics.stdev(rets) * math.sqrt(252)

    assert perf.num_periods == 5
    assert perf.starting_nav == 1000000.0
    assert perf.ending_nav == 1004580.0
    assert perf.cumulative_return == pytest.approx(0.00458)
    assert perf.sharpe_ratio == pytest.approx(exp_sharpe)
    assert perf.sharpe_ratio > 0
    assert math.isfinite(perf.annualized_return)
    assert math.isfinite(perf.annualized_volatility)


def test_sharpe_ratio_convenience_matches_summary() -> None:
    canon = flex_sections_to_canonical(_sections())
    table = _nav_table(canon["nav"])
    assert sharpe_ratio(table) == pytest.approx(compute_performance(table).sharpe_ratio)


def test_risk_free_rate_lowers_sharpe() -> None:
    canon = flex_sections_to_canonical(_sections())
    table = _nav_table(canon["nav"])
    base = sharpe_ratio(table, risk_free_annual=0.0)
    with_rf = sharpe_ratio(table, risk_free_annual=0.05)
    assert with_rf < base


def test_max_drawdown_is_nonpositive_and_captures_dip() -> None:
    canon = flex_sections_to_canonical(_sections())
    perf = compute_performance(_nav_table(canon["nav"]))
    # NAV dips from 1001000 to 1000400 on day 3: drawdown = 1000400/1001000 - 1.
    assert perf.max_drawdown == pytest.approx(1000400.0 / 1001000.0 - 1.0)
    assert perf.max_drawdown < 0


def test_fewer_than_two_points_raises() -> None:
    table = _nav_from_series([(datetime(2026, 6, 1, tzinfo=timezone.utc), 100.0)])
    with pytest.raises(ValueError, match="at least two"):
        compute_performance(table)


def test_flat_nav_gives_zero_return_and_nan_sharpe() -> None:
    pts = [
        (datetime(2026, 6, d, tzinfo=timezone.utc), 100.0) for d in (1, 2, 3)
    ]
    perf = compute_performance(_nav_from_series(pts))
    assert perf.cumulative_return == pytest.approx(0.0)
    assert math.isnan(perf.sharpe_ratio)  # zero volatility -> undefined


# ---------------------------------------------------------------------------
# External-flow stripping
# ---------------------------------------------------------------------------


def test_flows_strip_deposit_from_return() -> None:
    d1 = datetime(2026, 6, 1, 4, tzinfo=timezone.utc)
    d2 = datetime(2026, 6, 2, 4, tzinfo=timezone.utc)
    table = _nav_from_series([(d1, 100_000.0), (d2, 150_000.0)])

    # Without flows, the +50k deposit looks like a +50% return.
    assert daily_returns(table)[0] == pytest.approx(0.50)
    # With the deposit recorded, the performance return is ~0.
    flows = {d2.date(): 50_000.0}
    assert daily_returns(table, flows=flows)[0] == pytest.approx(0.0)


def test_net_external_flows_reported() -> None:
    d1 = datetime(2026, 6, 1, 4, tzinfo=timezone.utc)
    d2 = datetime(2026, 6, 2, 4, tzinfo=timezone.utc)
    table = _nav_from_series([(d1, 100_000.0), (d2, 150_000.0)])
    perf = compute_performance(table, flows={d2.date(): 50_000.0})
    assert perf.net_external_flows == pytest.approx(50_000.0)


def test_a_flow_on_a_day_with_no_nav_row_is_still_stripped() -> None:
    """Flows were keyed by the later NAV date, so non-trading days matched nothing.

    2026-06-05 is a Friday and 2026-06-08 the following Monday; a deposit dated the Saturday
    belongs to that Friday->Monday return period. Keying on the NAV date dropped it entirely
    while still reporting it in `net_external_flows`. Measured on a 100k book, the same
    $50,000 gave +50.5% cumulative return dated Saturday against +0.33% dated Monday.
    """
    fri = datetime(2026, 6, 5, 4, tzinfo=timezone.utc)
    mon = datetime(2026, 6, 8, 4, tzinfo=timezone.utc)
    table = _nav_from_series([(fri, 100_000.0), (mon, 150_000.0)])

    saturday = daily_returns(table, flows={date(2026, 6, 6): 50_000.0})[0]
    monday = daily_returns(table, flows={date(2026, 6, 8): 50_000.0})[0]
    assert saturday == pytest.approx(0.0)
    assert saturday == pytest.approx(monday)


def test_flows_outside_the_nav_window_are_not_counted_as_applied() -> None:
    """`net_external_flows` must report what was subtracted, not every flow in the report.

    A deposit predating the series is already inside `starting_nav`; claiming it in the flow
    total tells a reader it was handled when no return ever saw it.
    """
    d1 = datetime(2026, 6, 5, 4, tzinfo=timezone.utc)
    d2 = datetime(2026, 6, 8, 4, tzinfo=timezone.utc)
    table = _nav_from_series([(d1, 100_000.0), (d2, 150_000.0)])

    before = compute_performance(table, flows={date(2026, 6, 1): 50_000.0})
    assert before.net_external_flows == pytest.approx(0.0)
    assert daily_returns(table, flows={date(2026, 6, 1): 50_000.0})[0] == pytest.approx(0.50)

    inside = compute_performance(table, flows={date(2026, 6, 6): 50_000.0})
    assert inside.net_external_flows == pytest.approx(50_000.0)


# ---------------------------------------------------------------------------
# performance_summary source dispatch
# ---------------------------------------------------------------------------


class _FakeResult:
    def __init__(self, table: pa.Table) -> None:
        self._table = table

    def snapshot(self) -> pa.Table:
        return self._table


class _FakePort:
    """Minimal DataPort-like object: table(name).snapshot() -> Arrow."""

    def __init__(self, tables: dict[str, pa.Table]) -> None:
        self._tables = tables

    def table(self, name: str) -> _FakeResult:
        return _FakeResult(self._tables[name])


def test_performance_summary_accepts_arrow_table() -> None:
    canon = flex_sections_to_canonical(_sections())
    perf = performance_summary(_nav_table(canon["nav"]))
    assert perf.num_periods == 5


def test_performance_summary_accepts_result_like() -> None:
    canon = flex_sections_to_canonical(_sections())
    perf = performance_summary(_FakeResult(_nav_table(canon["nav"])))
    assert perf.ending_nav == 1004580.0


def test_performance_summary_port_derives_flows_from_cash() -> None:
    d1 = datetime(2026, 6, 1, 4, tzinfo=timezone.utc)
    d2 = datetime(2026, 6, 2, 4, tzinfo=timezone.utc)
    nav = _nav_from_series([(d1, 100_000.0), (d2, 150_000.0)])
    cash = pa.table(
        {
            "Account": pa.array(["U1"], type=pa.string()),
            "Timestamp": pa.array([d2], type=pa.timestamp("ns", tz="UTC")),
            "Type": pa.array(["Deposits/Withdrawals"], type=pa.string()),
            "Sym": pa.array([None], type=pa.string()),
            "Amount": pa.array([50_000.0], type=pa.float64()),
            "Currency": pa.array(["USD"], type=pa.string()),
        }
    )
    port = _FakePort({"nav": nav, "cash": cash})
    # Deposit is stripped -> ~0 return -> ~0 cumulative.
    perf = performance_summary(port)
    assert perf.cumulative_return == pytest.approx(0.0)
    # Disabling flow adjustment lets the deposit masquerade as a 50% return.
    perf_raw = performance_summary(port, adjust_for_flows=False)
    assert perf_raw.cumulative_return == pytest.approx(0.50)


def test_performance_summary_missing_nav_table_raises() -> None:
    port = _FakePort({"cash": pa.table({"x": pa.array([1])})})
    with pytest.raises(ValueError, match="no 'nav' table"):
        performance_summary(port)


# ---------------------------------------------------------------------------
# sharpe_ratio source dispatch (widened to match performance_summary's union)
# ---------------------------------------------------------------------------


def test_sharpe_ratio_accepts_result_like() -> None:
    canon = flex_sections_to_canonical(_sections())
    table = _nav_table(canon["nav"])
    assert sharpe_ratio(_FakeResult(table)) == pytest.approx(sharpe_ratio(table))


def test_sharpe_ratio_accepts_dataport_like_and_derives_flows_from_cash() -> None:
    d1 = datetime(2026, 6, 1, 4, tzinfo=timezone.utc)
    d2 = datetime(2026, 6, 2, 4, tzinfo=timezone.utc)
    nav = _nav_from_series([(d1, 100_000.0), (d2, 150_000.0)])
    cash = pa.table(
        {
            "Account": pa.array(["U1"], type=pa.string()),
            "Timestamp": pa.array([d2], type=pa.timestamp("ns", tz="UTC")),
            "Type": pa.array(["Deposits/Withdrawals"], type=pa.string()),
            "Sym": pa.array([None], type=pa.string()),
            "Amount": pa.array([50_000.0], type=pa.float64()),
            "Currency": pa.array(["USD"], type=pa.string()),
        }
    )
    port = _FakePort({"nav": nav, "cash": cash})
    # Deposit stripped -> ~0 return -> zero-volatility, single-period Sharpe is nan
    # for both entry points (they resolve NAV/flows identically).
    assert math.isnan(sharpe_ratio(port))
    assert math.isnan(performance_summary(port).sharpe_ratio)


def test_sharpe_ratio_matches_performance_summary_for_dataport() -> None:
    canon = flex_sections_to_canonical(_sections())
    port = _FakePort({"nav": _nav_table(canon["nav"])})
    assert sharpe_ratio(port, risk_free_annual=0.03) == pytest.approx(
        performance_summary(port, risk_free_annual=0.03).sharpe_ratio
    )


def test_as_dict_is_json_safe() -> None:
    canon = flex_sections_to_canonical(_sections())
    d = compute_performance(_nav_table(canon["nav"])).as_dict()
    assert isinstance(d["start"], str)
    assert isinstance(d["sharpe_ratio"], float)
    assert "max_drawdown" in d


def test_compute_performance_sets_account_from_nav() -> None:
    canon = flex_sections_to_canonical(_sections())
    perf = compute_performance(_nav_table(canon["nav"]))
    assert perf.account == "U0000000"


def _multi_account_nav_table() -> pa.Table:
    """Two-account NAV table (U1, U2) with disjoint value series for account tests."""
    rows = [
        {"Account": "U1", "Timestamp": datetime(2026, 6, 1, tzinfo=timezone.utc),
         "Total": 100000.0, "Cash": None, "Stock": None},
        {"Account": "U1", "Timestamp": datetime(2026, 6, 2, tzinfo=timezone.utc),
         "Total": 101000.0, "Cash": None, "Stock": None},
        {"Account": "U2", "Timestamp": datetime(2026, 6, 1, tzinfo=timezone.utc),
         "Total": 200000.0, "Cash": None, "Stock": None},
        {"Account": "U2", "Timestamp": datetime(2026, 6, 2, tzinfo=timezone.utc),
         "Total": 199000.0, "Cash": None, "Stock": None},
    ]
    return _nav_table(rows)


def test_compute_performance_multi_account_requires_selection() -> None:
    table = _multi_account_nav_table()
    with pytest.raises(ValueError, match=r"multiple accounts \['U1', 'U2'\]"):
        compute_performance(table)


def test_compute_performance_account_selects_correct_series() -> None:
    table = _multi_account_nav_table()
    perf = compute_performance(table, account="U2")
    assert perf.account == "U2"
    assert perf.starting_nav == 200000.0
    assert perf.ending_nav == 199000.0
    assert perf.cumulative_return == pytest.approx(199000.0 / 200000.0 - 1.0)

    perf_u1 = compute_performance(table, account="U1")
    assert perf_u1.account == "U1"
    assert perf_u1.starting_nav == 100000.0
    assert perf_u1.ending_nav == 101000.0


def test_compute_performance_unknown_account_raises() -> None:
    table = _multi_account_nav_table()
    with pytest.raises(ValueError, match=r"no NAV rows for account 'U9'"):
        compute_performance(table, account="U9")


def test_compute_performance_single_account_unaffected_by_account_param() -> None:
    """Regression: a single-account NAV (or the account=None default) still works."""
    canon = flex_sections_to_canonical(_sections())
    table = _nav_table(canon["nav"])
    perf_default = compute_performance(table)
    perf_explicit = compute_performance(table, account="U0000000")
    assert perf_default.account == perf_explicit.account == "U0000000"
    assert perf_default.sharpe_ratio == pytest.approx(perf_explicit.sharpe_ratio)


def test_compute_performance_no_account_column_unaffected() -> None:
    """Regression: a bare returns table with no 'Account' column passes through."""
    rows = [
        {"Timestamp": datetime(2026, 6, 1, tzinfo=timezone.utc), "Total": 100.0},
        {"Timestamp": datetime(2026, 6, 2, tzinfo=timezone.utc), "Total": 101.0},
    ]
    table = pa.table({
        "Timestamp": pa.array([r["Timestamp"] for r in rows], type=pa.timestamp("ns", tz="UTC")),
        "Total": pa.array([r["Total"] for r in rows], type=pa.float64()),
    })
    perf = compute_performance(table, account="anything")
    assert perf.account is None
    assert perf.starting_nav == 100.0


def test_performance_summary_multi_account_requires_selection_and_account_selects() -> None:
    table = _multi_account_nav_table()
    with pytest.raises(ValueError, match="multiple accounts"):
        performance_summary(table)
    perf = performance_summary(table, account="U1")
    assert perf.account == "U1"
    assert perf.starting_nav == 100000.0


# ---------------------------------------------------------------------------
# DataPort cash-flow derivation must be filtered by the same account as NAV
# ---------------------------------------------------------------------------


def _multi_account_cash_table() -> pa.Table:
    """Same-date cash flows for two accounts: U1 deposits, U2 withdraws.

    Used to prove ``performance_summary``/``sharpe_ratio`` derive flows for
    *only* the requested account, not a mix of every account in the cash table.
    """
    # A market-hours stamp (10:00 ET), not midnight UTC: external_flows_from_cash
    # buckets a cash row by its account-local calendar day, so a midnight-UTC stamp
    # would denote the PREVIOUS local day and this fixture's intended flow date
    # would depend on the bucketing rule rather than on the test's subject
    # (account filtering). 14:00Z is the same calendar day in both zones.
    d2 = datetime(2026, 6, 2, 14, 0, tzinfo=timezone.utc)
    return pa.table(
        {
            "Account": pa.array(["U1", "U2"], type=pa.string()),
            "Timestamp": pa.array([d2, d2], type=pa.timestamp("ns", tz="UTC")),
            "Type": pa.array(
                ["Deposits/Withdrawals", "Deposits/Withdrawals"], type=pa.string()
            ),
            "Sym": pa.array([None, None], type=pa.string()),
            "Amount": pa.array([5_000.0, -3_000.0], type=pa.float64()),
            "Currency": pa.array(["USD", "USD"], type=pa.string()),
        }
    )


def test_performance_summary_port_filters_cash_flows_by_account() -> None:
    """Multi-account DataPort: account="U1" must use only U1's cash flows.

    NAV for U1 goes 100_000 -> 101_000; U1's own deposit is +5_000, so the
    flow-adjusted return is (101_000 - 100_000 - 5_000) / 100_000 = -4%. If
    U2's -3_000 withdrawal leaked into U1's flows (the pre-fix bug), the net
    flow would be +2_000 and the return would be -1% instead.
    """
    nav = _multi_account_nav_table()
    cash = _multi_account_cash_table()
    port = _FakePort({"nav": nav, "cash": cash})

    perf_u1 = performance_summary(port, account="U1")
    assert perf_u1.account == "U1"
    assert perf_u1.cumulative_return == pytest.approx(-0.04)

    perf_u2 = performance_summary(port, account="U2")
    assert perf_u2.account == "U2"
    # NAV for U2 goes 200_000 -> 199_000; U2's own withdrawal is -3_000, so the
    # flow-adjusted return is (199_000 - 200_000 + 3_000) / 200_000 = +1%.
    assert perf_u2.cumulative_return == pytest.approx(0.01)

    # Reproduce the pre-fix (mixed-account) result directly: deriving flows
    # from the *unfiltered* cash table (both accounts' amounts summed per
    # date) and feeding that into compute_performance for U1's NAV gives a
    # different, wrong, answer -- proving the account filter is load-bearing.
    mixed_flows = external_flows_from_cash(cash)
    perf_u1_mixed = compute_performance(nav, account="U1", flows=mixed_flows)
    assert perf_u1_mixed.cumulative_return == pytest.approx(-0.01)
    assert perf_u1_mixed.cumulative_return != pytest.approx(perf_u1.cumulative_return)


def test_sharpe_ratio_port_filters_cash_flows_by_account() -> None:
    """sharpe_ratio resolves flows the same way performance_summary does.

    (A 2-point NAV series yields a single return, so Sharpe is undefined --
    ``nan`` -- for both accounts here; the point is that both entry points
    agree, i.e. resolve flows identically via the same account filter.)
    """
    nav = _multi_account_nav_table()
    cash = _multi_account_cash_table()
    port = _FakePort({"nav": nav, "cash": cash})
    assert math.isnan(sharpe_ratio(port, account="U1"))
    assert math.isnan(performance_summary(port, account="U1").sharpe_ratio)
    assert math.isnan(sharpe_ratio(port, account="U2"))
    assert math.isnan(performance_summary(port, account="U2").sharpe_ratio)


def test_performance_summary_single_account_port_unaffected_by_cash_filter() -> None:
    """Regression: single-account DataPort with cash flows -- unchanged behavior."""
    d1 = datetime(2026, 6, 1, 4, tzinfo=timezone.utc)
    d2 = datetime(2026, 6, 2, 4, tzinfo=timezone.utc)
    nav = _nav_from_series([(d1, 100_000.0), (d2, 150_000.0)])
    cash = pa.table(
        {
            "Account": pa.array(["U1"], type=pa.string()),
            "Timestamp": pa.array([d2], type=pa.timestamp("ns", tz="UTC")),
            "Type": pa.array(["Deposits/Withdrawals"], type=pa.string()),
            "Sym": pa.array([None], type=pa.string()),
            "Amount": pa.array([50_000.0], type=pa.float64()),
            "Currency": pa.array(["USD"], type=pa.string()),
        }
    )
    port = _FakePort({"nav": nav, "cash": cash})
    perf_default = performance_summary(port)
    perf_explicit = performance_summary(port, account="U1")
    assert perf_default.cumulative_return == pytest.approx(0.0)
    assert perf_explicit.cumulative_return == pytest.approx(perf_default.cumulative_return)


def test_performance_summary_port_cash_without_account_column_unaffected() -> None:
    """Regression: a cash table with no 'Account' column is unaffected by *account*."""
    d1 = datetime(2026, 6, 1, 4, tzinfo=timezone.utc)
    d2 = datetime(2026, 6, 2, 4, tzinfo=timezone.utc)
    nav = _nav_from_series([(d1, 100_000.0), (d2, 150_000.0)])
    cash = pa.table(
        {
            "Timestamp": pa.array([d2], type=pa.timestamp("ns", tz="UTC")),
            "Type": pa.array(["Deposits/Withdrawals"], type=pa.string()),
            "Sym": pa.array([None], type=pa.string()),
            "Amount": pa.array([50_000.0], type=pa.float64()),
            "Currency": pa.array(["USD"], type=pa.string()),
        }
    )
    port = _FakePort({"nav": nav, "cash": cash})
    perf_no_account = performance_summary(port)
    perf_with_account = performance_summary(port, account="U1")
    assert perf_no_account.cumulative_return == pytest.approx(0.0)
    assert perf_with_account.cumulative_return == pytest.approx(
        perf_no_account.cumulative_return
    )


def test_performance_summary_port_unknown_account_in_cash_degrades_gracefully() -> None:
    """NAV/cash account-set disagreement: an *account* present in NAV but absent
    from the cash table must not crash the whole call -- flow derivation for
    that account degrades to "no flows" (matching the existing broad
    exception-swallow around flow derivation), while NAV selection still
    succeeds normally.
    """
    nav = _multi_account_nav_table()  # covers U1 and U2
    cash = pa.table(
        {
            "Account": pa.array(["U2"], type=pa.string()),
            # 10:00 ET — unambiguously 2026-06-02 in both UTC and the local zone
            # cash flows are bucketed by (see _multi_account_cash_table).
            "Timestamp": pa.array(
                [datetime(2026, 6, 2, 14, 0, tzinfo=timezone.utc)],
                type=pa.timestamp("ns", tz="UTC"),
            ),
            "Type": pa.array(["Deposits/Withdrawals"], type=pa.string()),
            "Sym": pa.array([None], type=pa.string()),
            "Amount": pa.array([-3_000.0], type=pa.float64()),
            "Currency": pa.array(["USD"], type=pa.string()),
        }
    )
    port = _FakePort({"nav": nav, "cash": cash})
    # U1 has no cash rows at all -- flow derivation for U1 finds nothing to
    # filter, degrades to flows=None, and NAV-to-NAV is used unadjusted.
    perf_u1 = performance_summary(port, account="U1")
    assert perf_u1.account == "U1"
    assert not perf_u1.flows_applied
    assert perf_u1.cumulative_return == pytest.approx(101000.0 / 100000.0 - 1.0)
    # U2 has its own cash row and is filtered/adjusted normally.
    perf_u2 = performance_summary(port, account="U2")
    assert perf_u2.flows_applied
    assert perf_u2.cumulative_return == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Flow day-anchoring: cash is bucketed by the account-local calendar day.
# ---------------------------------------------------------------------------


def _flow_cash_table(
    stamps: list[datetime],
    amounts: list[float],
    currencies: list[str | None] | None = None,
    rates: list[float | None] | None = None,
) -> pa.Table:
    """Build a minimal cash table of deposit rows, one per stamp."""
    n = len(stamps)
    cols: dict[str, pa.Array] = {
        "Account": pa.array(["U1"] * n, type=pa.string()),
        "Timestamp": pa.array(stamps, type=pa.timestamp("ns", tz="UTC")),
        "Type": pa.array(["Deposits/Withdrawals"] * n, type=pa.string()),
        "Sym": pa.array([None] * n, type=pa.string()),
        "Amount": pa.array(amounts, type=pa.float64()),
        "Currency": pa.array(
            currencies if currencies is not None else ["USD"] * n, type=pa.string()
        ),
    }
    if rates is not None:
        cols["FxRateToBase"] = pa.array(rates, type=pa.float64())
    return pa.table(cols)


def test_flow_day_anchoring_matches_the_flex_adapter_timezone() -> None:
    """The zone flows are bucketed in is the one the Flex adapter anchors NAV to.

    Two constants describing the same account setting can drift apart silently and
    the symptom — a flow subtracted from the wrong day — looks like a data problem,
    not a configuration one. Pin them together.
    """
    from ibda.adapters.ibkr.flex.mapping import _FLEX_TZ
    from ibda.analytics.performance import _NAV_DAY_TZ

    assert _NAV_DAY_TZ == _FLEX_TZ


def test_external_flows_buckets_an_evening_stamp_to_the_local_day() -> None:
    """21:00 local (already tomorrow in UTC) still belongs to today's NAV day."""
    from datetime import date as _date

    # 2026-06-02 21:00 America/New_York (EDT, UTC-4) == 2026-06-03T01:00Z.
    stamp = datetime(2026, 6, 3, 1, 0, tzinfo=timezone.utc)
    flows = external_flows_from_cash(_flow_cash_table([stamp], [50_000.0]))
    assert flows == {_date(2026, 6, 2): pytest.approx(50_000.0)}


def test_external_flows_naive_timestamp_is_read_as_utc() -> None:
    """A naive Timestamp is treated as UTC, never as the process's local zone."""
    from datetime import date as _date

    naive = datetime(2026, 6, 3, 1, 0)
    aware = datetime(2026, 6, 3, 1, 0, tzinfo=timezone.utc)
    naive_table = pa.table(
        {
            "Timestamp": pa.array([naive], type=pa.timestamp("ns")),
            "Type": pa.array(["Deposits/Withdrawals"], type=pa.string()),
            "Amount": pa.array([50_000.0], type=pa.float64()),
        }
    )
    assert external_flows_from_cash(naive_table) == {_date(2026, 6, 2): 50_000.0}
    assert external_flows_from_cash(_flow_cash_table([aware], [50_000.0])) == {
        _date(2026, 6, 2): 50_000.0
    }


# ---------------------------------------------------------------------------
# Currency contract: flows are summed in base currency, never in mixed units.
# ---------------------------------------------------------------------------

_D2 = datetime(2026, 6, 2, 14, 0, tzinfo=timezone.utc)  # 10:00 ET


def test_external_flows_converts_local_to_base_with_fx_rate() -> None:
    """A 1,000 EUR deposit at FxRateToBase=1.08 contributes 1,080 base currency."""
    from datetime import date as _date

    flows = external_flows_from_cash(
        _flow_cash_table([_D2], [1_000.0], currencies=["EUR"], rates=[1.08])
    )
    assert flows == {_date(2026, 6, 2): pytest.approx(1_080.0)}


def test_external_flows_passes_through_base_currency_rows_unchanged() -> None:
    """A base-currency row with no usable rate is summed at face value.

    Covers both shapes a single-currency book produces: no FxRateToBase column at
    all, and the column present but null.
    """
    from datetime import date as _date

    no_column = external_flows_from_cash(_flow_cash_table([_D2], [5_000.0]))
    null_rate = external_flows_from_cash(
        _flow_cash_table([_D2], [5_000.0], currencies=["USD"], rates=[None])
    )
    assert no_column == {_date(2026, 6, 2): pytest.approx(5_000.0)}
    assert null_rate == no_column


def test_external_flows_null_currency_is_treated_as_base() -> None:
    """A row with no Currency at all keeps today's behaviour: summed as-is."""
    from datetime import date as _date

    flows = external_flows_from_cash(
        _flow_cash_table([_D2], [5_000.0], currencies=[None], rates=[None])
    )
    assert flows == {_date(2026, 6, 2): pytest.approx(5_000.0)}


def test_external_flows_skips_foreign_rows_with_no_rate_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unconvertible foreign row is dropped with a WARNING, never summed raw.

    Adding a 1,000 EUR magnitude to a USD flow total states a wrong number with
    full confidence. Dropping it leaves the period's return over-attributed to
    trading by an amount the log names.
    """
    import logging

    from datetime import date as _date

    table = _flow_cash_table(
        [_D2, _D2], [1_000.0, 500.0], currencies=["EUR", "USD"], rates=[None, None]
    )
    with caplog.at_level(logging.WARNING, logger="ibda.analytics.performance"):
        flows = external_flows_from_cash(table)

    # Only the USD row survives; the EUR magnitude is NOT added in.
    assert flows == {_date(2026, 6, 2): pytest.approx(500.0)}
    messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("EUR" in m and "1000.0" in m for m in messages), (
        f"expected a WARNING naming the currency and amount; got: {messages}"
    )
    assert any("2026-06-02" in m for m in messages), (
        f"expected the warning to name the date; got: {messages}"
    )


def test_external_flows_base_currency_is_configurable() -> None:
    """A non-USD base account converts the other way round."""
    from datetime import date as _date

    table = _flow_cash_table(
        [_D2, _D2], [1_000.0, 500.0], currencies=["EUR", "USD"], rates=[None, None]
    )
    flows = external_flows_from_cash(table, base_currency="eur")
    # EUR is now base and passes through; the USD row is the unconvertible one.
    assert flows == {_date(2026, 6, 2): pytest.approx(1_000.0)}


def test_external_flows_ignores_a_nonpositive_rate() -> None:
    """A 0.0 or negative rate is not a rate; it must not zero out or flip a flow."""
    from datetime import date as _date

    zero = external_flows_from_cash(
        _flow_cash_table([_D2], [5_000.0], currencies=["USD"], rates=[0.0])
    )
    assert zero == {_date(2026, 6, 2): pytest.approx(5_000.0)}


def test_extended_risk_metrics() -> None:
    """Calmar, hit rate, and best/worst day are computed and self-consistent."""
    canon = flex_sections_to_canonical(_sections())
    table = _nav_table(canon["nav"])
    rets = daily_returns(table)
    perf = compute_performance(table)

    # hit rate = fraction of positive days (4 of 5 fixture returns are up)
    assert perf.hit_rate == pytest.approx(sum(1 for r in rets if r > 0) / len(rets))
    assert perf.hit_rate == pytest.approx(0.8)
    assert perf.best_period == pytest.approx(max(rets))
    assert perf.worst_period == pytest.approx(min(rets))
    # Calmar = annualized return / |max drawdown|
    assert perf.calmar_ratio == pytest.approx(
        perf.annualized_return / abs(perf.max_drawdown)
    )
    assert "calmar_ratio" in perf.as_dict()


def test_calmar_is_nan_without_drawdown() -> None:
    pts = [
        (datetime(2026, 6, 1, tzinfo=timezone.utc), 100.0),
        (datetime(2026, 6, 2, tzinfo=timezone.utc), 101.0),
        (datetime(2026, 6, 3, tzinfo=timezone.utc), 102.0),
    ]
    perf = compute_performance(_nav_from_series(pts))  # monotonically rising
    assert perf.max_drawdown == 0.0
    assert math.isnan(perf.calmar_ratio)
    assert perf.hit_rate == pytest.approx(1.0)


def test_render_is_human_readable() -> None:
    canon = flex_sections_to_canonical(_sections())
    text = compute_performance(_nav_table(canon["nav"])).render()
    assert "Sharpe ratio" in text
    assert "Sortino ratio" in text
    assert "Calmar ratio" in text
    assert "Max drawdown" in text
    assert "Positive days" in text
    assert "Best / worst day" in text
    assert "U0000000" in text
    # percentages and the novice glossary line are present
    assert "%" in text
    assert "higher is better" in text


# ---------------------------------------------------------------------------
# flows_applied field
# ---------------------------------------------------------------------------


def test_flows_applied_true_when_flows_supplied() -> None:
    d1 = datetime(2026, 6, 1, 4, tzinfo=timezone.utc)
    d2 = datetime(2026, 6, 2, 4, tzinfo=timezone.utc)
    table = _nav_from_series([(d1, 100_000.0), (d2, 150_000.0)])
    perf = compute_performance(table, flows={d2.date(): 50_000.0})
    assert perf.flows_applied is True


def test_flows_applied_false_when_no_flows() -> None:
    canon = flex_sections_to_canonical(_sections())
    table = _nav_table(canon["nav"])
    perf = compute_performance(table)
    assert perf.flows_applied is False


def test_render_shows_not_flow_adjusted_marker_when_flows_not_applied() -> None:
    canon = flex_sections_to_canonical(_sections())
    text = compute_performance(_nav_table(canon["nav"])).render()
    assert "not flow-adjusted" in text


def test_render_no_not_flow_adjusted_marker_when_flows_applied() -> None:
    d1 = datetime(2026, 6, 1, 4, tzinfo=timezone.utc)
    d2 = datetime(2026, 6, 2, 4, tzinfo=timezone.utc)
    table = _nav_from_series([(d1, 100_000.0), (d2, 150_000.0)])
    perf = compute_performance(table, flows={d2.date(): 50_000.0})
    assert "not flow-adjusted" not in perf.render()


# ---------------------------------------------------------------------------
# Engine-free Flex → Arrow + one-call flex_performance
# ---------------------------------------------------------------------------


def test_flex_arrow_tables_build_and_validate() -> None:
    from ibda.adapters.ibkr.flex.arrow import flex_arrow_tables
    from ibda.schema import CASH, EXECUTION

    tables = flex_arrow_tables(_sections())
    assert set(tables) == {"execution", "cash", "nav"}
    EXECUTION.validate(tables["execution"])
    CASH.validate(tables["cash"])
    NAV.validate(tables["nav"])
    assert tables["nav"].num_rows == 6


def test_flex_performance_from_file_path() -> None:
    import ibda

    perf = ibda.flex_performance(str(_FIXTURE))
    assert perf.num_periods == 5
    assert perf.account == "U0000000"
    assert perf.starting_nav == 1_000_000.0
    assert perf.ending_nav == 1_004_580.0
    assert perf.sharpe_ratio > 0


def test_flex_performance_from_xml_string() -> None:
    import ibda

    perf = ibda.flex_performance(_FIXTURE.read_text())
    assert perf.cumulative_return == pytest.approx(0.00458)


def test_flex_performance_strips_deposit_via_cash() -> None:
    """A report with a deposit + matching NAV jump nets to ~0 return."""
    import ibda

    xml = """<FlexQueryResponse>
    <FlexStatements count="1">
    <FlexStatement accountId="U1" fromDate="2026-06-01" toDate="2026-06-02">
    <CashTransactions>
    <CashTransaction accountId="U1" type="Deposits/Withdrawals" dateTime="2026-06-02"
      amount="50000.00" currency="USD" description="WIRE IN"/>
    </CashTransactions>
    <EquitySummaryInBase>
    <EquitySummaryByReportDateInBase accountId="U1" reportDate="20260601" total="100000.00"/>
    <EquitySummaryByReportDateInBase accountId="U1" reportDate="20260602" total="150000.00"/>
    </EquitySummaryInBase>
    </FlexStatement>
    </FlexStatements>
    </FlexQueryResponse>"""
    perf = ibda.flex_performance(xml)
    assert perf.net_external_flows == pytest.approx(50_000.0)
    assert perf.cumulative_return == pytest.approx(0.0)


def test_flex_performance_no_nav_section_raises() -> None:
    import ibda

    xml = """<FlexQueryResponse>
    <FlexStatements count="1">
    <FlexStatement accountId="U1" fromDate="2026-06-01" toDate="2026-06-02">
    <Trades>
    <Trade accountId="U1" symbol="AAPL" assetCategory="STK" dateTime="2026-06-01 10:00:00"
      buySell="BUY" quantity="1" tradePrice="100" currency="USD"/>
    </Trades>
    </FlexStatement>
    </FlexStatements>
    </FlexQueryResponse>"""
    with pytest.raises(ValueError, match="no daily NAV series"):
        ibda.flex_performance(xml)


def test_flex_performance_multi_account_requires_selection() -> None:
    import ibda

    xml = """<FlexQueryResponse>
    <FlexStatements count="1">
    <FlexStatement accountId="MULTI" fromDate="2026-06-01" toDate="2026-06-02">
    <EquitySummaryInBase>
    <EquitySummaryByReportDateInBase accountId="U1" reportDate="20260601" total="100000.00"/>
    <EquitySummaryByReportDateInBase accountId="U1" reportDate="20260602" total="101000.00"/>
    <EquitySummaryByReportDateInBase accountId="U2" reportDate="20260601" total="200000.00"/>
    <EquitySummaryByReportDateInBase accountId="U2" reportDate="20260602" total="199000.00"/>
    </EquitySummaryInBase>
    </FlexStatement>
    </FlexStatements>
    </FlexQueryResponse>"""
    with pytest.raises(ValueError, match="multiple accounts"):
        ibda.flex_performance(xml)
    # selecting one account works and isolates its series
    perf = ibda.flex_performance(xml, account="U2")
    assert perf.account == "U2"
    assert perf.cumulative_return == pytest.approx(199000.0 / 200000.0 - 1.0)


def test_flex_performance_in_progress_raises_flexparseerror() -> None:
    import ibda
    from ibda.errors import FlexParseError

    xml = (
        '<FlexStatementResponse timestamp="x"><Status>Warn</Status>'
        "<ErrorCode>1019</ErrorCode><ErrorMessage>generating</ErrorMessage>"
        "</FlexStatementResponse>"
    )
    with pytest.raises(FlexParseError, match="in_progress"):
        ibda.flex_performance(xml)


# ---------------------------------------------------------------------------
# CLI (python -m ibda)
# ---------------------------------------------------------------------------


def test_cli_text_report(capsys: pytest.CaptureFixture[str]) -> None:
    from ibda.__main__ import main

    rc = main([str(_FIXTURE), "--risk-free", "0.04"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Performance summary" in out
    assert "Sharpe ratio" in out


def test_cli_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    import json

    from ibda.__main__ import main

    rc = main([str(_FIXTURE), "--risk-free", "0.04", "--json"])
    captured = capsys.readouterr()
    assert rc == 0
    payload = json.loads(captured.out)  # stdout is pure JSON
    assert payload["account"] == "U0000000"
    assert payload["num_periods"] == 5
    assert payload["risk_free_annual"] == pytest.approx(0.04)
    # the risk-free provenance line goes to stderr, not stdout
    assert "risk-free: 4.00%" in captured.err


def test_cli_risk_free_auto_uses_offline_constant(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--risk-free auto (the default) resolves to the documented offline constant —
    no network call (ibda's connectivity is IB-only)."""
    import json

    from ibda.__main__ import main
    from ibda.rates import DEFAULT_RISK_FREE_ANNUAL

    rc = main([str(_FIXTURE), "--json"])  # no --risk-free -> auto
    captured = capsys.readouterr()
    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["risk_free_annual"] == pytest.approx(DEFAULT_RISK_FREE_ANNUAL)
    assert "offline default" in captured.err


def test_cli_missing_file_reports_error(capsys: pytest.CaptureFixture[str]) -> None:
    from ibda.__main__ import main

    rc = main(["/nonexistent/does_not_exist.xml", "--risk-free", "0.0"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "error:" in err


# ---------------------------------------------------------------------------
# Risk-free rate (ibda.rates)
# ---------------------------------------------------------------------------


def test_default_risk_free_is_nonzero() -> None:
    from ibda.rates import DEFAULT_RISK_FREE_ANNUAL

    assert DEFAULT_RISK_FREE_ANNUAL > 0  # never silently 0%


def test_performance_default_risk_free_is_applied() -> None:
    """The performance API defaults to the Treasury proxy, not 0%."""
    from ibda.rates import DEFAULT_RISK_FREE_ANNUAL

    canon = flex_sections_to_canonical(_sections())
    table = _nav_table(canon["nav"])
    assert compute_performance(table).risk_free_annual == DEFAULT_RISK_FREE_ANNUAL
    # A higher risk-free hurdle yields a strictly lower Sharpe than rf=0.
    assert compute_performance(table).sharpe_ratio < sharpe_ratio(table, risk_free_annual=0.0)


def test_resolve_risk_free_modes() -> None:
    from ibda.rates import DEFAULT_RISK_FREE_ANNUAL, DEFAULT_RISK_FREE_AS_OF, resolve_risk_free

    # explicit numeric
    rate, src = resolve_risk_free(0.05)
    assert rate == pytest.approx(0.05)
    assert src == "user-specified"

    # explicit numeric string
    rate, src = resolve_risk_free("0.05")
    assert rate == pytest.approx(0.05)
    assert src == "user-specified"

    # auto / None -> the documented offline constant (no network call); provenance
    # names the as-of date so a consumer logging it can gauge staleness.
    rate, src = resolve_risk_free("auto")
    assert rate == DEFAULT_RISK_FREE_ANNUAL
    assert "offline default" in src
    assert DEFAULT_RISK_FREE_AS_OF in src

    rate, src = resolve_risk_free(None)
    assert rate == DEFAULT_RISK_FREE_ANNUAL
    assert "offline default" in src
    assert DEFAULT_RISK_FREE_AS_OF in src


def test_resolve_risk_free_bad_string_raises() -> None:
    from ibda.rates import resolve_risk_free

    with pytest.raises(ValueError):
        resolve_risk_free("not-a-number")


# ---------------------------------------------------------------------------
# risk_free_annual="auto" accepted by the library facade (not just the MCP layer)
# ---------------------------------------------------------------------------


def test_sharpe_ratio_accepts_auto_risk_free() -> None:
    from ibda.rates import DEFAULT_RISK_FREE_ANNUAL

    canon = flex_sections_to_canonical(_sections())
    table = _nav_table(canon["nav"])
    assert sharpe_ratio(table, risk_free_annual="auto") == pytest.approx(
        sharpe_ratio(table, risk_free_annual=DEFAULT_RISK_FREE_ANNUAL)
    )


def test_performance_summary_accepts_auto_risk_free() -> None:
    from ibda.rates import DEFAULT_RISK_FREE_ANNUAL

    canon = flex_sections_to_canonical(_sections())
    perf = performance_summary(_nav_table(canon["nav"]), risk_free_annual="auto")
    assert perf.risk_free_annual == DEFAULT_RISK_FREE_ANNUAL


def test_flex_performance_accepts_auto_risk_free() -> None:
    import ibda
    from ibda.rates import DEFAULT_RISK_FREE_ANNUAL

    perf = ibda.flex_performance(str(_FIXTURE), risk_free_annual="auto")
    assert perf.risk_free_annual == DEFAULT_RISK_FREE_ANNUAL


def test_formula_helpers_match_compute_performance() -> None:
    """The extracted helpers ARE the formulas compute_performance composes — no duplication."""
    from ibda.analytics.performance import (
        _annualized_return_from_returns,
        _annualized_volatility,
        _cumulative_return,
        _sharpe_from_returns,
        _sortino_from_returns,
    )

    rets = [0.01, -0.005, 0.012, 0.003, -0.002, 0.008]
    ppy = 252
    rf = 0.04
    assert _cumulative_return(rets) == pytest.approx(math.prod(1.0 + r for r in rets) - 1.0)
    exp_vol = statistics.stdev(rets) * math.sqrt(ppy)
    assert _annualized_volatility(rets, periods_per_year=ppy) == pytest.approx(exp_vol)
    rf_period = rf / ppy
    excess = [r - rf_period for r in rets]
    exp_sharpe = statistics.fmean(excess) / statistics.stdev(rets) * math.sqrt(ppy)
    assert _sharpe_from_returns(
        rets, risk_free_annual=rf, periods_per_year=ppy
    ) == pytest.approx(exp_sharpe)

    downside = math.sqrt(math.fsum(min(x, 0.0) ** 2 for x in excess) / len(excess))
    exp_sortino = statistics.fmean(excess) / downside * math.sqrt(ppy)
    assert _sortino_from_returns(
        rets, risk_free_annual=rf, periods_per_year=ppy
    ) == pytest.approx(exp_sortino)

    growth = math.prod(1.0 + r for r in rets)
    exp_annualized_return = growth ** (ppy / len(rets)) - 1.0
    assert _annualized_return_from_returns(
        rets, periods_per_year=ppy
    ) == pytest.approx(exp_annualized_return)

    # compute_performance is composed from these same helpers — its numbers must match.
    from datetime import datetime, timedelta, timezone

    import pyarrow as pa

    nav_vals = [1000.0]
    for r in rets:
        nav_vals.append(nav_vals[-1] * (1.0 + r))
    ts = [datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=i) for i in range(len(nav_vals))]
    table = pa.table({
        "Timestamp": pa.array(ts, type=pa.timestamp("ns", tz="UTC")),
        "Total": pa.array(nav_vals, type=pa.float64()),
    })
    perf = compute_performance(table, risk_free_annual=rf, periods_per_year=ppy)
    assert perf.sortino_ratio == pytest.approx(exp_sortino)
    assert perf.annualized_return == pytest.approx(exp_annualized_return)


def test_sharpe_helper_nan_on_degenerate() -> None:
    from ibda.analytics.performance import _sharpe_from_returns

    assert math.isnan(_sharpe_from_returns([], risk_free_annual=0.0, periods_per_year=252))
    assert math.isnan(_sharpe_from_returns([0.01], risk_free_annual=0.0, periods_per_year=252))
    assert math.isnan(  # zero variance -> undefined Sharpe
        _sharpe_from_returns([0.01, 0.01, 0.01], risk_free_annual=0.0, periods_per_year=252)
    )


def test_sortino_helper_nan_on_degenerate() -> None:
    from ibda.analytics.performance import _sortino_from_returns

    assert math.isnan(_sortino_from_returns([], risk_free_annual=0.0, periods_per_year=252))
    assert math.isnan(  # single return -> undefined, mirrors Sharpe's n<2 guard
        _sortino_from_returns([0.01], risk_free_annual=0.0, periods_per_year=252)
    )
    assert math.isnan(  # no downside -> zero downside deviation -> undefined Sortino
        _sortino_from_returns([0.01, 0.02, 0.03], risk_free_annual=0.0, periods_per_year=252)
    )


def test_annualized_return_helper_nan_on_empty_or_total_loss() -> None:
    from ibda.analytics.performance import _annualized_return_from_returns

    assert math.isnan(_annualized_return_from_returns([], periods_per_year=252))
    assert math.isnan(  # single return -> would extrapolate one day to a full year
        _annualized_return_from_returns([0.01], periods_per_year=252)
    )
    assert math.isnan(  # -100% return -> zero/negative growth -> undefined annualized return
        _annualized_return_from_returns([-1.0], periods_per_year=252)
    )


def test_sortino_and_annualized_return_multi_point_unchanged() -> None:
    """Confirm the multi-point (n>=2) formulas are unaffected by the new n<2 guards."""
    from ibda.analytics.performance import (
        _annualized_return_from_returns,
        _sortino_from_returns,
    )

    rets = [0.01, -0.005, 0.012, 0.003, -0.002, 0.008]
    ppy = 252
    rf = 0.04
    rf_period = rf / ppy
    excess = [r - rf_period for r in rets]
    downside = math.sqrt(math.fsum(min(x, 0.0) ** 2 for x in excess) / len(excess))
    exp_sortino = statistics.fmean(excess) / downside * math.sqrt(ppy)
    assert _sortino_from_returns(
        rets, risk_free_annual=rf, periods_per_year=ppy
    ) == pytest.approx(exp_sortino)

    growth = math.prod(1.0 + r for r in rets)
    exp_annualized_return = growth ** (ppy / len(rets)) - 1.0
    assert _annualized_return_from_returns(
        rets, periods_per_year=ppy
    ) == pytest.approx(exp_annualized_return)


def test_all_periods_skipped_raises_an_actionable_error_not_zerodivision() -> None:
    """`len(values) >= 2` does not imply at least one usable return period.

    `_dated_returns` skips any period whose prior NAV is 0.0, so `[0.0, 100000.0]` — a Flex
    report whose first row precedes funding — yielded zero returns and then divided by n in
    `hit_rate`. Unguarded that is a bare `ZeroDivisionError`, with `max(returns)` raising on
    the empty sequence immediately after.
    """
    import datetime as _dt

    import pyarrow as pa

    nav = pa.table({
        "Account": ["U1", "U1"],
        "Timestamp": [
            _dt.datetime(2026, 1, 1, tzinfo=_dt.timezone.utc),
            _dt.datetime(2026, 1, 2, tzinfo=_dt.timezone.utc),
        ],
        "Total": [0.0, 100_000.0],
    })
    with pytest.raises(ValueError, match="no usable return periods"):
        compute_performance(nav)
