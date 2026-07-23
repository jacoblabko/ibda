"""Tests for ACATS in-kind transfer handling — parse, map, and flow-stripping.

Covers:
* _map_transfer unit tests: sign, valuation precedence, unvalued -> None
* report_full.xml: unvalued GOOGL transfer logs a warning, no Transfer cash row
* transfer_acats.xml: end-to-end via flex_performance; transfer day return is
  NOT inflated; flows_applied=True; net_external_flows reflects transfer value
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

import pandas as pd

from ibda.adapters.ibkr.flex.mapping import _map_transfer, flex_sections_to_canonical
from ibda.adapters.ibkr.flex.parse import parse_statement

_FIXTURE_FULL = Path(__file__).parent / "fixtures" / "flex" / "report_full.xml"
_FIXTURE_ACATS = Path(__file__).parent / "fixtures" / "flex" / "transfer_acats.xml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sections(fixture: Path) -> dict[str, Any]:
    parsed = parse_statement(fixture.read_text())
    assert parsed["status"] == "ok"
    return dict(parsed["sections"])


def _make_transfer(**kwargs: Any) -> dict[str, Any]:
    """Construct a minimal transfer dict for _map_transfer unit tests."""
    base: dict[str, Any] = {
        "kind": "transfer",
        "symbol": "AAPL",
        "date_time": "2026-06-02",
        "direction": "IN",
        "quantity": None,
        "positionAmount": None,
        "positionAmountInBase": None,
        "transferPrice": None,
        "cashTransfer": None,
        "currency": "USD",
        "account": "U0000001",
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# _map_transfer unit tests
# ---------------------------------------------------------------------------


class TestMapTransfer:
    def test_in_direction_yields_positive_amount(self) -> None:
        t = _make_transfer(positionAmountInBase=50000.0, direction="IN")
        result = _map_transfer(t)
        assert result is not None
        assert result["Amount"] == 50000.0

    def test_out_direction_yields_negative_amount(self) -> None:
        t = _make_transfer(positionAmountInBase=50000.0, direction="OUT")
        result = _map_transfer(t)
        assert result is not None
        assert result["Amount"] == -50000.0

    def test_direction_case_insensitive(self) -> None:
        t = _make_transfer(positionAmountInBase=10000.0, direction="out")
        result = _map_transfer(t)
        assert result is not None
        assert result["Amount"] == -10000.0

    def test_missing_direction_treated_as_in(self) -> None:
        t = _make_transfer(positionAmountInBase=10000.0)
        t.pop("direction", None)
        result = _map_transfer(t)
        assert result is not None
        assert result["Amount"] > 0.0

    def test_valuation_precedence_position_amount_in_base_preferred(self) -> None:
        """positionAmountInBase takes priority over all other sources."""
        t = _make_transfer(
            positionAmountInBase=90000.0,
            positionAmount=80000.0,
            quantity=200.0,
            transferPrice=350.0,
            cashTransfer=70000.0,
        )
        result = _map_transfer(t)
        assert result is not None
        assert result["Amount"] == 90000.0

    def test_valuation_precedence_position_amount_over_qty_price(self) -> None:
        """positionAmount preferred over quantity*transferPrice."""
        t = _make_transfer(
            positionAmount=80000.0,
            quantity=200.0,
            transferPrice=350.0,  # 200*350 = 70000 — should be ignored
            cashTransfer=60000.0,
        )
        result = _map_transfer(t)
        assert result is not None
        assert result["Amount"] == 80000.0

    def test_valuation_qty_times_price_when_no_position_amount(self) -> None:
        """Falls back to quantity * transferPrice when position amounts absent."""
        t = _make_transfer(quantity=200.0, transferPrice=350.0)
        result = _map_transfer(t)
        assert result is not None
        assert result["Amount"] == pytest.approx(70000.0)

    def test_valuation_cash_transfer_last_resort(self) -> None:
        t = _make_transfer(cashTransfer=55000.0)
        result = _map_transfer(t)
        assert result is not None
        assert result["Amount"] == 55000.0

    def test_unvalued_returns_none(self) -> None:
        """No valuation source -> None (caller will warn and skip)."""
        t = _make_transfer()  # all valuation fields None
        result = _map_transfer(t)
        assert result is None

    def test_canonical_fields_present(self) -> None:
        t = _make_transfer(positionAmountInBase=12345.0, direction="IN")
        result = _map_transfer(t)
        assert result is not None
        assert result["Type"] == "Transfer"
        assert result["Account"] == "U0000001"
        assert result["Sym"] == "AAPL"
        assert result["Currency"] == "USD"
        assert "Timestamp" in result

    def test_abs_of_negative_position_amount(self) -> None:
        """Magnitude is always abs() of the chosen value; sign from direction only."""
        t = _make_transfer(positionAmountInBase=-84000.0, direction="IN")
        result = _map_transfer(t)
        assert result is not None
        assert result["Amount"] == 84000.0


# ---------------------------------------------------------------------------
# report_full.xml: unvalued GOOGL transfer is warned and skipped
# ---------------------------------------------------------------------------


class TestUnvaluedTransferFromReportFull:
    def test_no_transfer_cash_row_in_output(self) -> None:
        """The unvalued GOOGL ACATS in report_full.xml must not produce a Transfer row."""
        canon = flex_sections_to_canonical(_sections(_FIXTURE_FULL))
        transfer_rows = [r for r in canon["cash"] if r.get("Type") == "Transfer"]
        assert transfer_rows == [], (
            f"Expected no Transfer cash rows from report_full.xml, got: {transfer_rows}"
        )

    def test_cash_row_count_unchanged(self) -> None:
        """The existing 3 cash rows (Dividends + Interest + Fee) are still present."""
        canon = flex_sections_to_canonical(_sections(_FIXTURE_FULL))
        assert len(canon["cash"]) == 3

    def test_warning_logged_for_unvalued_googl_transfer(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Loading report_full.xml must log a warning about the GOOGL transfer."""
        with caplog.at_level(logging.WARNING, logger="ibda.adapters.ibkr.flex.mapping"):
            flex_sections_to_canonical(_sections(_FIXTURE_FULL))

        # At least one warning mentioning GOOGL and/or the skip reason
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("GOOGL" in r.message for r in warnings), (
            "Expected a warning mentioning GOOGL for the unvalued ACATS transfer"
        )
        assert any("NOT be stripped" in r.message or "unvalued" in r.message.lower() for r in warnings), (
            "Warning should explain the transfer will NOT be stripped from returns"
        )


# ---------------------------------------------------------------------------
# transfer_acats.xml: end-to-end via flex_performance
# ---------------------------------------------------------------------------


class TestAcatsEndToEnd:
    """ACATS-valued transfer fixture: transfer is stripped from returns."""

    def test_transfer_cash_row_present(self) -> None:
        canon = flex_sections_to_canonical(_sections(_FIXTURE_ACATS))
        transfer_rows = [r for r in canon["cash"] if r.get("Type") == "Transfer"]
        assert len(transfer_rows) == 1
        row = transfer_rows[0]
        assert row["Amount"] == pytest.approx(84000.0)
        assert row["Sym"] == "MSFT"
        assert row["Currency"] == "USD"

    def test_flows_applied_true(self) -> None:
        from ibda.adapters.ibkr.flex.arrow import flex_performance

        perf = flex_performance(str(_FIXTURE_ACATS))
        assert perf.flows_applied is True

    def test_net_external_flows_reflects_transfer(self) -> None:
        from ibda.adapters.ibkr.flex.arrow import flex_performance

        perf = flex_performance(str(_FIXTURE_ACATS))
        # The only external flow is the 84000 MSFT transfer on day-2
        assert perf.net_external_flows == pytest.approx(84000.0)

    def test_transfer_day_return_not_inflated(self) -> None:
        """With flow-stripping, the day-2 return should be near 0 (no market gain).

        Without stripping, day-2 return would be ~(384000-300000)/300000 = +28%.
        With stripping:  (384000-300000-84000)/300000 = 0/300000 = 0.0.
        """
        from ibda.analytics.performance import daily_returns, external_flows_from_cash
        from ibda.adapters.ibkr.flex.arrow import arrow_table_from_rows
        from ibda.adapters.ibkr.flex.mapping import flex_sections_to_canonical
        from ibda.adapters.ibkr.flex.parse import parse_statement_or_raise
        from ibda.schema import CASH, NAV

        sections = parse_statement_or_raise(_FIXTURE_ACATS.read_text())
        canonical = flex_sections_to_canonical(sections)
        nav_table = arrow_table_from_rows(NAV, canonical["nav"])
        cash_table = arrow_table_from_rows(CASH, canonical["cash"])

        flows = external_flows_from_cash(cash_table)
        returns_with_flows = daily_returns(nav_table, flows=flows)
        returns_no_flows = daily_returns(nav_table, flows=None)

        # Day-2 (index 0) return: with flows ~0.0, without flows ~0.28
        assert returns_with_flows[0] == pytest.approx(0.0, abs=1e-9)
        assert returns_no_flows[0] == pytest.approx(0.28, abs=1e-3)

    def test_performance_not_distorted_by_transfer(self) -> None:
        """Sharpe / cumulative return should reflect only trading gains, not the transfer."""
        from ibda.adapters.ibkr.flex.arrow import flex_performance

        perf = flex_performance(str(_FIXTURE_ACATS))
        # Cumulative return: only day-3 (+1500/384000) and day-4 (-500/385500) matter
        # (day-2 is stripped). With 3 periods total including the stripped one, the
        # flow-adjusted returns are [0.0, ~0.00391, ~-0.00130]. Product - 1 ≈ +0.00260.
        assert -0.05 < perf.cumulative_return < 0.05, (
            f"cumulative_return={perf.cumulative_return:.4f} is suspiciously large — "
            "transfer may not have been stripped"
        )


# ---------------------------------------------------------------------------
# Transfer date-field fallback (date= / reportDate= / settleDate=)
# ---------------------------------------------------------------------------

_TRANSFER_DATE_ONLY = """
<FlexQueryResponse queryName="Test" type="AF">
<FlexStatements count="1">
<FlexStatement accountId="U9999999" fromDate="2026-06-01" toDate="2026-06-03" period="Custom">
<Trades/>
<CashTransactions/>
<CorporateActions/>
<Transfers>
  <!-- Uses date= only (no dateTime=) — should resolve to 2026-06-02, NOT 1970. -->
  <Transfer accountId="U9999999" symbol="TSLA" type="ACATS"
    date="20260602" direction="IN"
    quantity="100" transferPrice="180.00"
    positionAmount="18000.00" positionAmountInBase="18000.00"
    currency="USD" description="ACATS TRANSFER IN"/>
</Transfers>
<EquitySummaryInBase>
  <EquitySummaryByReportDateInBase accountId="U9999999" reportDate="2026-06-01"
    cash="50000.00" stock="0.00" total="50000.00"/>
  <EquitySummaryByReportDateInBase accountId="U9999999" reportDate="2026-06-02"
    cash="50000.00" stock="18000.00" total="68000.00"/>
  <EquitySummaryByReportDateInBase accountId="U9999999" reportDate="2026-06-03"
    cash="50000.00" stock="18500.00" total="68500.00"/>
</EquitySummaryInBase>
<ChangeInNAV accountId="U9999999" fromDate="2026-06-01" toDate="2026-06-03"
  startingValue="50000.00" endingValue="68500.00" mtm="500.00" realized="0.00"/>
</FlexStatement>
</FlexStatements>
</FlexQueryResponse>
"""

_TRANSFER_DATETIME = """
<FlexQueryResponse queryName="Test" type="AF">
<FlexStatements count="1">
<FlexStatement accountId="U9999998" fromDate="2026-06-01" toDate="2026-06-03" period="Custom">
<Trades/>
<CashTransactions/>
<CorporateActions/>
<Transfers>
  <!-- Uses dateTime= — should resolve to 2026-06-02 (dateTime takes priority). -->
  <Transfer accountId="U9999998" symbol="AMZN" type="ACATS"
    dateTime="20260602;094850" date="20260601" direction="IN"
    quantity="50" transferPrice="200.00"
    positionAmount="10000.00" positionAmountInBase="10000.00"
    currency="USD" description="ACATS TRANSFER IN"/>
</Transfers>
<EquitySummaryInBase>
  <EquitySummaryByReportDateInBase accountId="U9999998" reportDate="2026-06-01"
    cash="30000.00" stock="0.00" total="30000.00"/>
  <EquitySummaryByReportDateInBase accountId="U9999998" reportDate="2026-06-02"
    cash="30000.00" stock="10000.00" total="40000.00"/>
  <EquitySummaryByReportDateInBase accountId="U9999998" reportDate="2026-06-03"
    cash="30000.00" stock="10200.00" total="40200.00"/>
</EquitySummaryInBase>
<ChangeInNAV accountId="U9999998" fromDate="2026-06-01" toDate="2026-06-03"
  startingValue="30000.00" endingValue="40200.00" mtm="200.00" realized="0.00"/>
</FlexStatement>
</FlexStatements>
</FlexQueryResponse>
"""


class TestTransferDateFallback:
    """Transfer date-field priority: dateTime > date > reportDate > settleDate."""

    def _parse_transfer_rows(self, xml: str) -> list[dict[str, Any]]:
        parsed = parse_statement(xml)
        assert parsed["status"] == "ok"
        sections = dict(parsed["sections"])
        canon = flex_sections_to_canonical(sections)
        return [r for r in canon["cash"] if r.get("Type") == "Transfer"]

    def test_date_only_attribute_resolves_to_correct_date(self) -> None:
        """Transfer with date='20260602' (no dateTime) must yield Timestamp 2026-06-02."""
        rows = self._parse_transfer_rows(_TRANSFER_DATE_ONLY)
        assert len(rows) == 1, f"Expected 1 Transfer row, got: {rows}"
        ts = rows[0]["Timestamp"]
        assert ts is not None, "Timestamp must not be None"
        dt = pd.Timestamp(ts)
        assert dt.date() == pd.Timestamp("2026-06-02").date(), (
            f"Expected 2026-06-02 from date= attribute, got {dt.date()} — "
            "possible 1970 epoch fallback if date= attribute was ignored"
        )

    def test_datetime_attribute_takes_priority_over_date(self) -> None:
        """Transfer with dateTime='20260602;094850' and date='20260601' uses dateTime."""
        rows = self._parse_transfer_rows(_TRANSFER_DATETIME)
        assert len(rows) == 1, f"Expected 1 Transfer row, got: {rows}"
        ts = rows[0]["Timestamp"]
        assert ts is not None
        dt = pd.Timestamp(ts)
        assert dt.date() == pd.Timestamp("2026-06-02").date(), (
            f"Expected 2026-06-02 from dateTime= (priority over date=), got {dt.date()}"
        )
        # The time component should be non-zero — confirms dateTime= (which carries
        # HH:MM:SS) was used rather than the date= fallback (which would produce
        # midnight). The exact UTC hour varies with timezone normalization.
        assert dt.second == 50 or dt.minute != 0 or dt.hour != 0, (
            f"Expected non-midnight time from dateTime=20260602;094850, got {dt.time()}"
        )

    def test_no_mark_value_field_in_parsed_transfer(self) -> None:
        """Parsed transfer dict must NOT contain markValue key."""
        parsed = parse_statement(_TRANSFER_DATE_ONLY)
        assert parsed["status"] == "ok"
        sections = dict(parsed["sections"])
        corp_actions = sections.get("corporate_actions", [])
        transfers = [r for r in corp_actions if r.get("kind") == "transfer"]
        assert len(transfers) == 1
        assert "markValue" not in transfers[0], (
            "markValue is not a valid Transfer field and must not appear in parsed output"
        )
