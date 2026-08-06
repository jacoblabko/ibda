"""Tests for ibda.adapters.ibkr.flex.mapping — pure Flex -> canonical row mapping.

All tests are offline: they parse the committed XML fixture and assert that
flex_sections_to_canonical returns rows conformant with EXECUTION and CASH schemas.
"""
from __future__ import annotations

import logging
import textwrap
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from ibda.adapters.ibkr.flex.mapping import (
    _parse_dt,
    _parse_dt_or_none,
    flex_sections_to_canonical,
)
from ibda.adapters.ibkr.flex.arrow import performance_from_sections
from ibda.adapters.ibkr.flex.parse import _to_float, parse_statement
from ibda.analytics.performance import compute_performance, external_flows_from_cash
from ibda.schema import CASH, EXECUTION, NAV

_FIXTURE = Path(__file__).parent / "fixtures" / "flex" / "report_full.xml"


def _sections() -> dict[str, Any]:
    parsed = parse_statement(_FIXTURE.read_text())
    assert parsed["status"] == "ok"
    result: dict[str, Any] = parsed["sections"]
    return result


def test_execution_rows_conform_to_schema_and_values() -> None:
    canon = flex_sections_to_canonical(_sections())
    execs = canon["execution"]
    assert len(execs) == 2  # report_full.xml has 2 trades
    cols = set(EXECUTION.column_names)
    for row in execs:
        assert set(row).issubset(cols)              # only canonical columns
        assert row["ExecId"]                         # non-null (required)
        assert row["Side"] in ("BUY", "SELL")
        assert isinstance(row["Qty"], float)
        assert isinstance(row["Price"], float)
    # first trade is AAPL BUY 100 @ 314.18 (from the fixture)
    aapl = next(r for r in execs if r["Sym"] == "AAPL" and r["Side"] == "BUY")
    assert aapl["Side"] == "BUY" and aapl["Qty"] == 100.0 and aapl["Price"] == 314.18


def test_execution_stk_rows_have_sec_type_and_multiplier() -> None:
    """STK rows get SecType=='STK' and Multiplier==1.0 (no multiplier attribute in Flex)."""
    canon = flex_sections_to_canonical(_sections())
    execs = canon["execution"]
    for row in execs:
        # Both trades in report_full.xml are STK rows.
        assert row["SecType"] == "STK", (
            f"Expected SecType='STK' for row {row.get('Sym')!r}, got {row['SecType']!r}"
        )
        assert row["Multiplier"] == 1.0, (
            f"Expected Multiplier=1.0 for STK row {row.get('Sym')!r}, got {row['Multiplier']!r}"
        )


def test_execution_realized_pnl_flows_through_from_fifo_pnl_realized() -> None:
    """fifoPnlRealized on the Trade element lands on the canonical RealizedPnl column.

    report_full.xml's AAPL BUY is an opening trade (fifoPnlRealized="0"); the
    matching SELL realizes 581.00 (fifoPnlRealized="581.00").
    """
    canon = flex_sections_to_canonical(_sections())
    execs = canon["execution"]
    buy = next(r for r in execs if r["Side"] == "BUY")
    sell = next(r for r in execs if r["Side"] == "SELL")
    assert buy["RealizedPnl"] == 0.0
    assert sell["RealizedPnl"] == 581.00


def test_execution_order_ref_flows_through_from_order_reference() -> None:
    """orderReference on the Trade element lands on the canonical OrderRef column.

    report_full.xml's SELL carries orderReference="demo-x1" (a closing trade); the
    opening BUY has none -> None, giving a natural null/non-null pair.
    """
    canon = flex_sections_to_canonical(_sections())
    execs = canon["execution"]
    buy = next(r for r in execs if r["Side"] == "BUY")
    sell = next(r for r in execs if r["Side"] == "SELL")
    assert sell["OrderRef"] == "demo-x1"
    assert buy["OrderRef"] is None


def test_execution_sell_qty_is_unsigned() -> None:
    """Qty is always positive; Side carries direction."""
    canon = flex_sections_to_canonical(_sections())
    sell = next(r for r in canon["execution"] if r["Side"] == "SELL")
    assert sell["Qty"] == 100.0   # abs(-100) from the fixture
    assert sell["Price"] == 320.00


def test_execution_synthetic_exec_id_when_absent() -> None:
    """Fixture has no ibExecID/tradeID, so a deterministic synthetic id is used.

    The id is a ``synx-`` prefixed content hash rather than the old
    ``"{symbol}-{dateTime}-{quantity}-{tradePrice}"`` string. The string form embedded
    the symbol and so read nicely, but it omitted the ACCOUNT — which made two
    accounts' identical fills collide the moment executions started being
    de-duplicated — and it ended in a price, which ``reconcile.normalize_exec_id``
    strips as a trailing ``.NN``. The marker also lets ``_dedupe_execution_rows``
    report fabricated-id collapses separately from real ibExecID ones.
    """
    canon = flex_sections_to_canonical(_sections())
    for row in canon["execution"]:
        assert row["ExecId"]  # non-empty
        assert row["ExecId"].startswith("synx-"), (
            "a fabricated ExecId must be marked so it is distinguishable from a real "
            f"ibExecID; got {row['ExecId']!r}"
        )


def test_synthetic_exec_id_is_deterministic_and_account_scoped() -> None:
    """The two properties the dedupe pass depends on.

    Deterministic, or an overlapping re-ingest cannot be collapsed. Account-scoped, or
    collapsing it would silently drop one account's copy of a fill both accounts made.
    """
    from ibda.adapters.ibkr.flex.mapping import _synthetic_exec_id

    def _mk(account: str) -> str:
        return _synthetic_exec_id(
            account=account,
            symbol="AAPL",
            date_time="20260602;103100",
            quantity=100.0,
            trade_price=200.0,
        )

    assert _mk("U111") == _mk("U111"), "the same fill must always hash to the same id"
    assert _mk("U111") != _mk("U222"), (
        "two accounts' identical fills must NOT collide — that is what made the old "
        "string form unsafe to de-duplicate"
    )


def test_parse_trades_extracts_order_reference_and_open_close() -> None:
    """Raw trade dicts carry orderReference/openCloseIndicator when Flex emits them."""
    trades = _sections()["trades"]
    tagged = next(t for t in trades if t["order_reference"] is not None)
    assert tagged["order_reference"] == "demo-x1"
    assert tagged["open_close"] == "C"


def test_parse_trades_order_reference_null_safe_when_absent() -> None:
    """Trades without orderReference/openCloseIndicator attrs get None, not a missing key."""
    trades = _sections()["trades"]
    untagged = next(t for t in trades if t["order_reference"] is None)
    assert "order_reference" in untagged
    assert "open_close" in untagged
    assert untagged["open_close"] is None


def test_cash_rows_conform_to_schema() -> None:
    canon = flex_sections_to_canonical(_sections())
    cash = canon["cash"]
    assert len(cash) == 3  # report_full.xml has 3 cash transactions
    cols = set(CASH.column_names)
    for row in cash:
        assert set(row).issubset(cols)
        assert row["Type"]
        assert isinstance(row["Amount"], float)


def test_cash_dividend_amount_and_symbol() -> None:
    canon = flex_sections_to_canonical(_sections())
    div = next(r for r in canon["cash"] if r["Type"] == "Dividends")
    assert div["Amount"] == 42.50
    assert div["Sym"] == "MSFT"


# ---------------------------------------------------------------------------
# Timezone-correctness tests — _parse_dt converts ET local → UTC, DST-aware
# ---------------------------------------------------------------------------


def test_parse_dt_edt_datetime_converts_to_utc() -> None:
    """2026-06-02 10:31:00 ET (EDT, UTC-4) → 2026-06-02T14:31:00Z."""
    result = _parse_dt("2026-06-02 10:31:00")
    expected = datetime(2026, 6, 2, 14, 31, 0, tzinfo=timezone.utc)
    assert result == expected


def test_parse_dt_est_datetime_converts_to_utc() -> None:
    """2026-01-15 10:31:00 ET (EST, UTC-5) → 2026-01-15T15:31:00Z."""
    result = _parse_dt("2026-01-15 10:31:00")
    expected = datetime(2026, 1, 15, 15, 31, 0, tzinfo=timezone.utc)
    assert result == expected


def test_parse_dt_date_only_treats_as_midnight_et() -> None:
    """Date-only strings are treated as ET midnight; 2026-06-03 ET → 2026-06-03T04:00Z (EDT)."""
    result = _parse_dt("2026-06-03")
    expected = datetime(2026, 6, 3, 4, 0, 0, tzinfo=timezone.utc)
    assert result == expected


def test_parse_dt_empty_returns_utc_epoch() -> None:
    """Empty / None input returns the UTC epoch unchanged (no tz shift)."""
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    assert _parse_dt("") == epoch
    assert _parse_dt(None) == epoch


def test_parse_dt_unparseable_returns_utc_epoch() -> None:
    """Garbage input returns epoch without raising."""
    result = _parse_dt("not-a-date")
    assert result == datetime(1970, 1, 1, tzinfo=timezone.utc)


def test_fixture_timestamps_are_utc_offset_from_edt() -> None:
    """Fixture trade at 2026-06-02 10:31 ET (EDT) maps to 14:31 UTC."""
    canon = flex_sections_to_canonical(_sections())
    buy = next(r for r in canon["execution"] if r["Side"] == "BUY")
    ts: datetime = buy["Timestamp"]
    assert ts.tzinfo == timezone.utc
    assert ts == datetime(2026, 6, 2, 14, 31, 0, tzinfo=timezone.utc)


def test_fixture_sell_timestamp_utc() -> None:
    """Fixture sell at 2026-06-05 15:45 ET (EDT) maps to 19:45 UTC."""
    canon = flex_sections_to_canonical(_sections())
    sell = next(r for r in canon["execution"] if r["Side"] == "SELL")
    ts: datetime = sell["Timestamp"]
    assert ts == datetime(2026, 6, 5, 19, 45, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Compact-format tests — IBKR default CashTransaction/Transfer dateTime forms
# ---------------------------------------------------------------------------


def test_parse_dt_compact_semicolon_datetime() -> None:
    """Compact IBKR default: 20260602;094850 ET (EDT, UTC-4) → 2026-06-02T13:48:50Z."""
    result = _parse_dt("20260602;094850")
    expected = datetime(2026, 6, 2, 13, 48, 50, tzinfo=timezone.utc)
    assert result == expected, f"got {result!r}, want {expected!r}"


def test_parse_dt_compact_date_only() -> None:
    """Compact date-only: 20260608 ET midnight (EDT, UTC-4) → 2026-06-08T04:00Z."""
    result = _parse_dt("20260608")
    expected = datetime(2026, 6, 8, 4, 0, 0, tzinfo=timezone.utc)
    assert result == expected, f"got {result!r}, want {expected!r}"


def test_parse_dt_compact_space_separator() -> None:
    """Compact with space: 20260602 094850 ET (EDT) → 2026-06-02T13:48:50Z."""
    result = _parse_dt("20260602 094850")
    expected = datetime(2026, 6, 2, 13, 48, 50, tzinfo=timezone.utc)
    assert result == expected, f"got {result!r}, want {expected!r}"


def test_parse_dt_compact_not_epoch() -> None:
    """Compact dateTime must NOT silently return the 1970 epoch."""
    result = _parse_dt("20260602;094850")
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    assert result != epoch, "compact dateTime was silently mapped to the 1970 epoch"


def test_parse_dt_dashed_formats_still_parse() -> None:
    """Backward-compat: existing dashed formats continue to work after compact support added."""
    assert _parse_dt("2026-06-02 10:31:00") == datetime(2026, 6, 2, 14, 31, 0, tzinfo=timezone.utc)
    assert _parse_dt("2026-06-03") == datetime(2026, 6, 3, 4, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Integration: compact-date CashTransaction flows through to canonical cash rows
# and external_flows_from_cash keys the flow to the correct date (not 1970).
# ---------------------------------------------------------------------------

_COMPACT_CASH_XML = textwrap.dedent("""\
    <FlexQueryResponse queryName="Activity" type="AF">
    <FlexStatements count="1">
    <FlexStatement accountId="U9999999" fromDate="2026-06-01" toDate="2026-06-08" period="MTD">
    <Trades/>
    <CashTransactions>
    <CashTransaction accountId="U9999999" symbol="" type="Deposits/Withdrawals"
      dateTime="20260602;094850" amount="50000.00" currency="USD"
      description="ACH TRANSFER"/>
    </CashTransactions>
    <EquitySummaryInBase>
    <EquitySummaryByReportDateInBase accountId="U9999999" reportDate="20260601"
      cash="100000.00" stock="0.00" total="100000.00"/>
    <EquitySummaryByReportDateInBase accountId="U9999999" reportDate="20260602"
      cash="150000.00" stock="0.00" total="150000.00"/>
    <EquitySummaryByReportDateInBase accountId="U9999999" reportDate="20260603"
      cash="150000.00" stock="0.00" total="150000.00"/>
    </EquitySummaryInBase>
    </FlexStatement>
    </FlexStatements>
    </FlexQueryResponse>
""")


def test_compact_cash_dateTime_maps_to_correct_date() -> None:
    """End-to-end: compact dateTime on CashTransaction -> Timestamp keyed to 2026-06-02, not 1970."""
    parsed = parse_statement(_COMPACT_CASH_XML)
    assert parsed["status"] == "ok", f"parse failed: {parsed}"
    canon = flex_sections_to_canonical(parsed["sections"])
    cash = canon["cash"]
    assert len(cash) == 1
    ts: datetime = cash[0]["Timestamp"]
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    assert ts != epoch, f"compact dateTime was silently mapped to the 1970 epoch: {ts!r}"
    assert ts.year == 2026 and ts.month == 6 and ts.day == 2, (
        f"expected 2026-06-02, got {ts!r}"
    )


def test_compact_cash_flow_strips_from_performance() -> None:
    """Deposit with compact dateTime is stripped from NAV-based returns (not silently lost on 1970)."""
    parsed = parse_statement(_COMPACT_CASH_XML)
    assert parsed["status"] == "ok"
    canon = flex_sections_to_canonical(parsed["sections"])

    # Build cash Arrow table
    cash_rows = canon["cash"]
    cash_cols: dict[str, pa.Array] = {}
    for col in CASH.columns:
        values = [r.get(col.name) for r in cash_rows]
        cash_cols[col.name] = pa.array(values, type=col.dtype.to_arrow())
    cash_table = pa.table(cash_cols, schema=CASH.to_arrow_schema())

    flows = external_flows_from_cash(cash_table)

    # The flow must be keyed to 2026-06-02, not to 1970-01-01
    assert date(1970, 1, 1) not in flows, "flow landed on epoch date — compact dateTime not parsed"
    assert date(2026, 6, 2) in flows, f"expected flow on 2026-06-02, got keys: {list(flows)}"
    assert flows[date(2026, 6, 2)] == pytest.approx(50_000.0)

    # Build NAV table and verify flow is stripped from the day-2 return
    nav_rows = canon["nav"]
    nav_cols: dict[str, pa.Array] = {}
    for col in NAV.columns:
        values = [r.get(col.name) for r in nav_rows]
        nav_cols[col.name] = pa.array(values, type=col.dtype.to_arrow())
    nav_table = pa.table(nav_cols, schema=NAV.to_arrow_schema())

    perf = compute_performance(nav_table, flows=flows)
    # With 50k deposit stripped, the 100k→150k move is 0% investment return.
    assert perf.flows_applied is True
    assert perf.net_external_flows == pytest.approx(50_000.0)
    assert perf.cumulative_return == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Cash rows with an unusable Timestamp are DROPPED, not anchored to the epoch.
#
# A cash movement attributed to 1970-01-01 falls outside every NAV series, so it
# is silently absent from flow-adjusted returns either way. Dropping it with a
# WARNING makes that loss countable; the epoch anchor made it invisible.
# ---------------------------------------------------------------------------


def test_parse_dt_or_none_returns_none_for_absent_and_unparseable() -> None:
    """The non-substituting parser reports failure instead of inventing a value."""
    assert _parse_dt_or_none(None) is None
    assert _parse_dt_or_none("") is None
    assert _parse_dt_or_none("definitely-not-a-date") is None
    # ...and still parses everything the substituting form does.
    assert _parse_dt_or_none("20260602;094850") == datetime(
        2026, 6, 2, 13, 48, 50, tzinfo=timezone.utc
    )


_BAD_CASH_DATE_XML = textwrap.dedent("""\
    <FlexQueryResponse queryName="Activity" type="AF">
    <FlexStatements count="1">
    <FlexStatement accountId="U9999" fromDate="2026-06-01" toDate="2026-06-02">
    <Trades/>
    <CashTransactions>
    <CashTransaction accountId="U9999" symbol="MSFT" type="Dividends"
      dateTime="not-a-date" amount="42.50" currency="USD" description="MSFT DIVIDEND"/>
    <CashTransaction accountId="U9999" symbol="" type="Broker Interest Received"
      dateTime="2026-06-02" amount="3.21" currency="USD" description="CREDIT INT"/>
    </CashTransactions>
    </FlexStatement>
    </FlexStatements>
    </FlexQueryResponse>
""")


def test_cash_row_with_unparseable_datetime_is_dropped_and_warned(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unparseable dateTime drops the row, logs it, and leaves siblings alone."""
    parsed = parse_statement(_BAD_CASH_DATE_XML)
    assert parsed["status"] == "ok"
    with caplog.at_level(logging.WARNING, logger="ibda.adapters.ibkr.flex.mapping"):
        canon = flex_sections_to_canonical(parsed["sections"])

    cash = canon["cash"]
    assert len(cash) == 1, (
        f"the row with dateTime='not-a-date' must be dropped, not anchored to 1970; got {cash}"
    )
    # The good sibling row survives untouched.
    assert cash[0]["Amount"] == 3.21
    assert cash[0]["Timestamp"] != datetime(1970, 1, 1, tzinfo=timezone.utc)

    warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("not-a-date" in m for m in warnings), (
        f"expected a WARNING naming the raw dateTime; got: {warnings}"
    )
    assert any("unusable Timestamp" in m for m in warnings), (
        f"expected the drop reason to be identifiable in the log; got: {warnings}"
    )


def test_cash_row_with_absent_datetime_is_dropped() -> None:
    """A CashTransaction with no dateTime attribute at all is dropped too."""
    xml = textwrap.dedent("""\
        <FlexQueryResponse queryName="Activity" type="AF">
        <FlexStatements count="1">
        <FlexStatement accountId="U9999" fromDate="2026-06-01" toDate="2026-06-02">
        <Trades/>
        <CashTransactions>
        <CashTransaction accountId="U9999" symbol="MSFT" type="Dividends"
          amount="42.50" currency="USD" description="MSFT DIVIDEND"/>
        </CashTransactions>
        </FlexStatement>
        </FlexStatements>
        </FlexQueryResponse>
    """)
    parsed = parse_statement(xml)
    assert parsed["status"] == "ok"
    canon = flex_sections_to_canonical(parsed["sections"])
    assert canon["cash"] == []


def test_cash_drop_summary_separates_the_two_causes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Missing-Amount and unusable-Timestamp drops are counted separately.

    One log line that says only "N rows skipped" cannot tell a Flex query with a
    missing amount field from one emitting a date format the parser does not know;
    those need different fixes, so the counts stay separable.
    """
    xml = textwrap.dedent("""\
        <FlexQueryResponse queryName="Activity" type="AF">
        <FlexStatements count="1">
        <FlexStatement accountId="U9999" fromDate="2026-06-01" toDate="2026-06-02">
        <Trades/>
        <CashTransactions>
        <CashTransaction accountId="U9999" symbol="MSFT" type="Dividends"
          dateTime="2026-06-02" currency="USD" description="NO AMOUNT"/>
        <CashTransaction accountId="U9999" symbol="AAPL" type="Dividends"
          dateTime="nonsense" amount="1.00" currency="USD" description="BAD DATE"/>
        <CashTransaction accountId="U9999" symbol="GOOG" type="Dividends"
          dateTime="also-nonsense" amount="2.00" currency="USD" description="BAD DATE 2"/>
        </CashTransactions>
        </FlexStatement>
        </FlexStatements>
        </FlexQueryResponse>
    """)
    parsed = parse_statement(xml)
    assert parsed["status"] == "ok"
    with caplog.at_level(logging.WARNING, logger="ibda.adapters.ibkr.flex.mapping"):
        canon = flex_sections_to_canonical(parsed["sections"])

    assert canon["cash"] == []
    summaries = [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING and "Skipped" in r.getMessage()
    ]
    assert len(summaries) == 1, f"expected exactly one summary line; got: {summaries}"
    assert "Skipped 3 cash row(s): 1 with a missing Amount, 2 with an unusable" in summaries[0], (
        f"expected per-cause counts in the summary; got: {summaries[0]!r}"
    )


def test_cash_row_with_a_legitimate_1970_datetime_is_kept() -> None:
    """1970-01-01 is a parseable date, so it must survive — the drop rule keys on
    'unparseable', not on 'looks like the epoch'. A real (if implausible) 1970
    movement must not become collateral damage of the bad-date fix."""
    xml = textwrap.dedent("""\
        <FlexQueryResponse queryName="Activity" type="AF">
        <FlexStatements count="1">
        <FlexStatement accountId="U9999" fromDate="1970-01-01" toDate="1970-01-02">
        <Trades/>
        <CashTransactions>
        <CashTransaction accountId="U9999" symbol="" type="Deposits/Withdrawals"
          dateTime="19700101" amount="100.00" currency="USD" description="ANCIENT"/>
        </CashTransactions>
        </FlexStatement>
        </FlexStatements>
        </FlexQueryResponse>
    """)
    parsed = parse_statement(xml)
    assert parsed["status"] == "ok"
    canon = flex_sections_to_canonical(parsed["sections"])
    assert len(canon["cash"]) == 1
    ts: datetime = canon["cash"][0]["Timestamp"]
    # 1970-01-01 ET midnight (EST, UTC-5) -> 1970-01-01T05:00Z.
    assert ts == datetime(1970, 1, 1, 5, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Multi-currency: fxRateToBase survives parse -> canonical, and Amount/Currency
# stay the LOCAL pair.
# ---------------------------------------------------------------------------

_EUR_CASH_XML = textwrap.dedent("""\
    <FlexQueryResponse queryName="Activity" type="AF">
    <FlexStatements count="1">
    <FlexStatement accountId="U9999" fromDate="2026-06-01" toDate="2026-06-02">
    <Trades/>
    <CashTransactions>
    <CashTransaction accountId="U9999" symbol="" type="Deposits/Withdrawals"
      dateTime="2026-06-02" amount="1000.00" currency="EUR" fxRateToBase="1.08"
      description="EUR WIRE IN"/>
    <CashTransaction accountId="U9999" symbol="" type="Deposits/Withdrawals"
      dateTime="2026-06-02" amount="500.00" currency="USD" fxRateToBase="1"
      description="USD ACH IN"/>
    </CashTransactions>
    </FlexStatement>
    </FlexStatements>
    </FlexQueryResponse>
""")


def test_cash_fx_rate_to_base_survives_parse_to_canonical() -> None:
    """fxRateToBase on a CashTransaction reaches the canonical FxRateToBase column."""
    parsed = parse_statement(_EUR_CASH_XML)
    assert parsed["status"] == "ok"
    assert parsed["sections"]["cash"][0]["fxRateToBase"] == pytest.approx(1.08)

    canon = flex_sections_to_canonical(parsed["sections"])
    eur = next(r for r in canon["cash"] if r["Currency"] == "EUR")
    usd = next(r for r in canon["cash"] if r["Currency"] == "USD")
    assert eur["FxRateToBase"] == pytest.approx(1.08)
    assert usd["FxRateToBase"] == pytest.approx(1.0)
    # Amount and Currency remain the LOCAL pair — the row is not pre-converted.
    assert eur["Amount"] == pytest.approx(1000.0)
    assert set(eur).issubset(set(CASH.column_names))


def test_cash_fx_rate_is_none_when_flex_omits_it() -> None:
    """A query that does not select fxRateToBase yields None, not a fabricated 1.0."""
    canon = flex_sections_to_canonical(_sections())
    assert all(r["FxRateToBase"] is None for r in canon["cash"])


def test_transfer_row_carries_its_fx_rate_to_base() -> None:
    """A mapped transfer exposes the same rate, so the cash table is homogeneous."""
    from ibda.adapters.ibkr.flex.mapping import _map_transfer

    result = _map_transfer({
        "symbol": "SAP",
        "date_time": "2026-06-02",
        "direction": "IN",
        "positionAmountInBase": 10_800.0,
        "fxRateToBase": 1.08,
        "currency": "EUR",
    })
    assert result is not None
    # base 10,800 / 1.08 -> 10,000 EUR local, labelled EUR, with the rate attached.
    assert result["Amount"] == pytest.approx(10_000.0)
    assert result["Currency"] == "EUR"
    assert result["FxRateToBase"] == pytest.approx(1.08)


# ---------------------------------------------------------------------------
# Evening-stamped cash must bucket to the account-local day the NAV series uses.
# ---------------------------------------------------------------------------

_EVENING_CASH_XML = textwrap.dedent("""\
    <FlexQueryResponse queryName="Activity" type="AF">
    <FlexStatements count="1">
    <FlexStatement accountId="U9999999" fromDate="2026-06-01" toDate="2026-06-03" period="Custom">
    <Trades/>
    <CashTransactions>
    <CashTransaction accountId="U9999999" symbol="" type="Deposits/Withdrawals"
      dateTime="20260602;210000" amount="50000.00" currency="USD"
      description="EVENING ACH TRANSFER"/>
    </CashTransactions>
    <EquitySummaryInBase>
    <EquitySummaryByReportDateInBase accountId="U9999999" reportDate="20260601"
      cash="100000.00" stock="0.00" total="100000.00"/>
    <EquitySummaryByReportDateInBase accountId="U9999999" reportDate="20260602"
      cash="150000.00" stock="0.00" total="150000.00"/>
    <EquitySummaryByReportDateInBase accountId="U9999999" reportDate="20260603"
      cash="150000.00" stock="0.00" total="150000.00"/>
    </EquitySummaryInBase>
    </FlexStatement>
    </FlexStatements>
    </FlexQueryResponse>
""")


def test_evening_et_cash_buckets_to_the_same_local_day_as_nav() -> None:
    """A 21:00 local deposit is already the NEXT day in UTC; it must still be
    stripped from the day the NAV series booked it on.

    The deposit is stamped 2026-06-02 21:00 local = 2026-06-03T01:00Z, and the NAV
    series shows the 50,000 arriving on report date 2026-06-02. Bucketing the flow
    by its UTC date keys it to 06-03, leaving 06-02 with a fabricated +50% return.
    """
    parsed = parse_statement(_EVENING_CASH_XML)
    assert parsed["status"] == "ok"
    canon = flex_sections_to_canonical(parsed["sections"])

    cash_table = pa.table(
        {
            col.name: pa.array(
                [r.get(col.name) for r in canon["cash"]], type=col.dtype.to_arrow()
            )
            for col in CASH.columns
        },
        schema=CASH.to_arrow_schema(),
    )
    # The stored Timestamp really is the next UTC day — the roll is present in the
    # data, and the bucketing rule is what has to undo it.
    assert canon["cash"][0]["Timestamp"] == datetime(2026, 6, 3, 1, 0, tzinfo=timezone.utc)

    flows = external_flows_from_cash(cash_table)
    assert date(2026, 6, 2) in flows, (
        f"evening deposit rolled off its NAV day; flow keys: {list(flows)}"
    )
    assert date(2026, 6, 3) not in flows
    assert flows[date(2026, 6, 2)] == pytest.approx(50_000.0)

    nav_table = pa.table(
        {
            col.name: pa.array(
                [r.get(col.name) for r in canon["nav"]], type=col.dtype.to_arrow()
            )
            for col in NAV.columns
        },
        schema=NAV.to_arrow_schema(),
    )
    perf = compute_performance(nav_table, flows=flows)
    # With the deposit stripped from 06-02, the 100k->150k move is a 0% return.
    assert perf.cumulative_return == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# TxnId: cash rows carry an id, so the same movement seen twice is counted once.
# ---------------------------------------------------------------------------

_CASH_WITH_TXN_ID_XML = textwrap.dedent("""\
    <FlexQueryResponse queryName="Activity" type="AF">
    <FlexStatements count="1">
    <FlexStatement accountId="U9999" fromDate="2026-06-01" toDate="2026-06-02">
    <Trades/>
    <CashTransactions>
    <CashTransaction accountId="U9999" symbol="MSFT" type="Dividends"
      dateTime="2026-06-02" amount="42.50" currency="USD" transactionID="7788991"
      description="MSFT CASH DIVIDEND"/>
    <CashTransaction accountId="U9999" symbol="" type="Broker Interest Received"
      dateTime="2026-06-02" amount="3.21" currency="USD" description="CREDIT INT"/>
    </CashTransactions>
    </FlexStatement>
    </FlexStatements>
    </FlexQueryResponse>
""")


def test_cash_txn_id_prefers_the_ibkr_transaction_id() -> None:
    """When IBKR supplies transactionID it is used verbatim, unprefixed."""
    parsed = parse_statement(_CASH_WITH_TXN_ID_XML)
    assert parsed["status"] == "ok"
    canon = flex_sections_to_canonical(parsed["sections"])
    div = next(r for r in canon["cash"] if r["Type"] == "Dividends")
    interest = next(r for r in canon["cash"] if r["Type"] == "Broker Interest Received")

    assert div["TxnId"] == "7788991"
    # The sibling row has no transactionID, so it falls back to a marked synthetic.
    assert interest["TxnId"].startswith("syn-")
    assert set(div).issubset(set(CASH.column_names))


def test_synthetic_cash_id_is_deterministic_across_parses() -> None:
    """Parsing the same report twice yields identical ids — the property that makes
    de-duplicating a re-ingested overlapping window possible at all."""
    first = flex_sections_to_canonical(parse_statement(_COMPACT_CASH_XML)["sections"])
    second = flex_sections_to_canonical(parse_statement(_COMPACT_CASH_XML)["sections"])
    assert [r["TxnId"] for r in first["cash"]] == [r["TxnId"] for r in second["cash"]]
    assert all(r["TxnId"].startswith("syn-") for r in first["cash"])


def test_synthetic_cash_id_distinguishes_rows_that_differ_in_any_field() -> None:
    """Rows differing in exactly one identifying field must not collide."""
    from ibda.adapters.ibkr.flex.mapping import _synthetic_cash_id

    base: dict[str, Any] = {
        "Account": "U9999",
        "Timestamp": datetime(2026, 6, 2, 14, 0, tzinfo=timezone.utc),
        "Type": "Dividends",
        "Sym": "MSFT",
        "Amount": 42.5,
        "Currency": "USD",
    }
    baseline = _synthetic_cash_id(base)
    for field, other in (
        ("Account", "U8888"),
        ("Timestamp", datetime(2026, 6, 3, 14, 0, tzinfo=timezone.utc)),
        ("Type", "Other Fees"),
        ("Sym", "AAPL"),
        ("Amount", 42.51),
        ("Currency", "EUR"),
    ):
        variant = dict(base)
        variant[field] = other
        assert _synthetic_cash_id(variant) != baseline, f"{field} does not affect the id"


_OVERLAPPING_STATEMENTS_XML = textwrap.dedent("""\
    <FlexQueryResponse queryName="Activity" type="AF">
    <FlexStatements count="2">
    <FlexStatement accountId="U9999999" fromDate="2026-06-01" toDate="2026-06-02" period="Custom">
    <Trades/>
    <CashTransactions>
    <CashTransaction accountId="U9999999" symbol="" type="Deposits/Withdrawals"
      dateTime="20260602;094850" amount="50000.00" currency="USD"
      description="ACH TRANSFER"/>
    </CashTransactions>
    </FlexStatement>
    <FlexStatement accountId="U9999999" fromDate="2026-06-02" toDate="2026-06-03" period="Custom">
    <Trades/>
    <CashTransactions>
    <CashTransaction accountId="U9999999" symbol="" type="Deposits/Withdrawals"
      dateTime="20260602;094850" amount="50000.00" currency="USD"
      description="ACH TRANSFER"/>
    </CashTransactions>
    </FlexStatement>
    </FlexStatements>
    </FlexQueryResponse>
""")


def test_cash_rows_are_deduped_across_overlapping_statements(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One movement reported by two overlapping statement windows is counted once.

    Without de-duplication the 50,000 deposit is summed twice and every
    flow-adjusted return built on it is wrong by a full deposit.
    """
    parsed = parse_statement(_OVERLAPPING_STATEMENTS_XML)
    assert parsed["status"] == "ok"
    # The duplication is real at the parse layer — both statements are read.
    assert len(parsed["sections"]["cash"]) == 2

    with caplog.at_level(logging.WARNING, logger="ibda.adapters.ibkr.flex.mapping"):
        canon = flex_sections_to_canonical(parsed["sections"])

    assert len(canon["cash"]) == 1, (
        f"the repeated deposit must survive exactly once; got {canon['cash']}"
    )

    cash_table = pa.table(
        {
            col.name: pa.array(
                [r.get(col.name) for r in canon["cash"]], type=col.dtype.to_arrow()
            )
            for col in CASH.columns
        },
        schema=CASH.to_arrow_schema(),
    )
    flows = external_flows_from_cash(cash_table)
    assert flows[date(2026, 6, 2)] == pytest.approx(50_000.0), (
        "the deposit was counted more than once"
    )

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("Dropped 1 cash row(s) identical in" in m for m in warnings), (
        f"the collapse must be logged, with its limits stated; got: {warnings}"
    )


def test_dedupe_by_ibkr_id_ignores_unrelated_field_differences(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Two rows sharing a real transactionID are the same transaction even if a
    non-identifying detail was re-stated differently between reports."""
    xml = textwrap.dedent("""\
        <FlexQueryResponse queryName="Activity" type="AF">
        <FlexStatements count="1">
        <FlexStatement accountId="U9999" fromDate="2026-06-01" toDate="2026-06-02">
        <Trades/>
        <CashTransactions>
        <CashTransaction accountId="U9999" symbol="MSFT" type="Dividends"
          dateTime="2026-06-02" amount="42.50" currency="USD" transactionID="7788991"
          description="MSFT CASH DIVIDEND"/>
        <CashTransaction accountId="U9999" symbol="MSFT" type="Dividends"
          dateTime="2026-06-02" amount="42.50" currency="USD" transactionID="7788991"
          description="MSFT DIVIDEND (RESTATED)"/>
        </CashTransactions>
        </FlexStatement>
        </FlexStatements>
        </FlexQueryResponse>
    """)
    parsed = parse_statement(xml)
    assert parsed["status"] == "ok"
    with caplog.at_level(logging.WARNING, logger="ibda.adapters.ibkr.flex.mapping"):
        canon = flex_sections_to_canonical(parsed["sections"])

    assert len(canon["cash"]) == 1
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("sharing an IBKR transactionID" in m for m in warnings), (
        f"an IBKR-id collapse must be reported as such, not as a content match; got: {warnings}"
    )


def test_distinct_cash_rows_are_all_kept(caplog: pytest.LogCaptureFixture) -> None:
    """The de-duplication must not touch a report with no repeats.

    Three genuinely different movements on the same day and account stay three
    rows, and nothing is logged.
    """
    parsed = parse_statement(_FIXTURE.read_text())
    assert parsed["status"] == "ok"
    with caplog.at_level(logging.WARNING, logger="ibda.adapters.ibkr.flex.mapping"):
        canon = flex_sections_to_canonical(parsed["sections"])

    assert len(canon["cash"]) == 3
    assert len({r["TxnId"] for r in canon["cash"]}) == 3
    dedupe_warnings = [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING and "Dropped" in r.getMessage()
    ]
    assert not dedupe_warnings, f"unexpected de-duplication on a clean report: {dedupe_warnings}"


def test_transfers_are_not_collapsed_into_one_row() -> None:
    """Transfer rows get their own ids, so several distinct transfers all survive.

    A shared null id would have made the de-duplication pass keep exactly one
    transfer per report — silently deleting real capital movements.
    """
    xml = textwrap.dedent("""\
        <FlexQueryResponse queryName="Test" type="AF">
        <FlexStatements count="1">
        <FlexStatement accountId="U9999999" fromDate="2026-06-01" toDate="2026-06-03">
        <Trades/>
        <CashTransactions/>
        <Transfers>
          <Transfer accountId="U9999999" symbol="TSLA" type="ACATS" date="20260602"
            direction="IN" quantity="100" transferPrice="180.00"
            positionAmount="18000.00" currency="USD" description="IN 1"/>
          <Transfer accountId="U9999999" symbol="AMZN" type="ACATS" date="20260602"
            direction="IN" quantity="50" transferPrice="200.00"
            positionAmount="10000.00" currency="USD" description="IN 2"/>
          <Transfer accountId="U9999999" symbol="NVDA" type="ACATS" date="20260603"
            direction="OUT" quantity="10" transferPrice="100.00"
            positionAmount="1000.00" currency="USD" description="OUT 1"/>
        </Transfers>
        </FlexStatement>
        </FlexStatements>
        </FlexQueryResponse>
    """)
    parsed = parse_statement(xml)
    assert parsed["status"] == "ok"
    canon = flex_sections_to_canonical(parsed["sections"])
    transfers = [r for r in canon["cash"] if r["Type"] == "Transfer"]
    assert len(transfers) == 3, f"transfers were collapsed: {transfers}"
    assert len({r["TxnId"] for r in transfers}) == 3
    assert all(r["TxnId"].startswith("syn-") for r in transfers)


# ---------------------------------------------------------------------------
# Observability tests — silent substitutions now surface as logged warnings
# ---------------------------------------------------------------------------

# Audit item #1: bad/missing date → epoch substitution is observable via logging.


def test_parse_dt_bad_date_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    """_parse_dt emits a WARNING containing the raw value when the date is unparseable.

    The substitution (returning the 1970 epoch) is kept as-is; only the log is
    added to make the silent fallback observable.
    """
    bad_input = "definitely-not-a-date"
    with caplog.at_level(logging.WARNING, logger="ibda.adapters.ibkr.flex.mapping"):
        result = _parse_dt(bad_input)

    assert result == datetime(1970, 1, 1, tzinfo=timezone.utc), (
        "epoch substitution must still happen for unparseable dates"
    )
    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(bad_input in m for m in warning_messages), (
        f"Expected a WARNING containing {bad_input!r}; got messages: {warning_messages}"
    )


def test_parse_dt_empty_does_not_log(caplog: pytest.LogCaptureFixture) -> None:
    """Empty / None input returns epoch WITHOUT logging — absence is expected, not an error."""
    with caplog.at_level(logging.WARNING, logger="ibda.adapters.ibkr.flex.mapping"):
        _parse_dt("")
        _parse_dt(None)

    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert not warning_messages, (
        f"Expected no WARNING for empty/None input; got: {warning_messages}"
    )



# Audit item #3: bad float → None → 0.0 substitution is observable via logging,
# and the warning now includes the field name.


def test_to_float_bad_value_logs_warning_with_field_and_raw_value(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """_to_float logs a WARNING with both the raw value and the field name on parse failure.

    The function still returns None (caller will substitute 0.0), so behavior
    is unchanged — only observability improves.
    """
    with caplog.at_level(logging.WARNING, logger="ibda.adapters.ibkr.flex.parse"):
        result = _to_float("not-a-number", "quantity")

    assert result is None, "_to_float must still return None for unparseable input"
    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("not-a-number" in m for m in warning_messages), (
        f"Expected raw value 'not-a-number' in warning; got: {warning_messages}"
    )
    assert any("quantity" in m for m in warning_messages), (
        f"Expected field name 'quantity' in warning; got: {warning_messages}"
    )


def test_to_float_none_and_empty_do_not_log(caplog: pytest.LogCaptureFixture) -> None:
    """_to_float does NOT log for absent (None) or empty string inputs — those are expected."""
    with caplog.at_level(logging.WARNING, logger="ibda.adapters.ibkr.flex.parse"):
        assert _to_float(None, "amount") is None
        assert _to_float("", "amount") is None

    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert not warning_messages, (
        f"Expected no WARNING for None/empty inputs; got: {warning_messages}"
    )


# Audit item #2: multiplier absent on non-equity logs a WARNING.

# Minimal OPT XML with NO multiplier attribute — exercises the absent-multiplier
# warning path for a non-equity asset category.
_OPT_NO_MULTIPLIER_XML = textwrap.dedent("""\
    <FlexQueryResponse queryName="Activity" type="AF">
    <FlexStatements count="1">
    <FlexStatement accountId="U9999" fromDate="2026-06-01" toDate="2026-06-08">
    <Trades>
      <Trade accountId="U9999" symbol="AAPL  260117C00200000" assetCategory="OPT"
        dateTime="2026-06-02 10:00:00" buySell="BUY" quantity="5" tradePrice="3.20"
        ibCommission="-0.70" currency="USD" conid="678901"
        ibExecID="test-opt-exec-1"/>
    </Trades>
    </FlexStatement>
    </FlexStatements>
    </FlexQueryResponse>
""")


def test_multiplier_absent_on_opt_still_defaults_to_one(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """OPT row with no multiplier attribute still gets Multiplier=1.0 (behavior preserved)."""
    parsed = parse_statement(_OPT_NO_MULTIPLIER_XML)
    assert parsed["status"] == "ok"
    canon = flex_sections_to_canonical(parsed["sections"])
    execs = canon["execution"]
    assert len(execs) == 1
    assert execs[0]["Multiplier"] == 1.0, (
        "Multiplier must still default to 1.0 when absent, even for OPT rows"
    )


def test_multiplier_absent_on_opt_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """OPT row with no multiplier attribute emits a WARNING — absence could mis-price notionals."""
    parsed = parse_statement(_OPT_NO_MULTIPLIER_XML)
    assert parsed["status"] == "ok"
    with caplog.at_level(logging.WARNING, logger="ibda.adapters.ibkr.flex.mapping"):
        flex_sections_to_canonical(parsed["sections"])

    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert warning_messages, "Expected at least one WARNING for OPT row missing multiplier"
    # Warning should mention the asset category so the caller can identify the row.
    assert any("OPT" in m for m in warning_messages), (
        f"Expected 'OPT' in the warning message; got: {warning_messages}"
    )


def test_multiplier_absent_on_stk_does_not_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """STK row with no multiplier attribute (normal) must NOT emit a warning — it's expected."""
    # report_full.xml has STK trades with no multiplier attribute.
    parsed = parse_statement(_FIXTURE.read_text())
    assert parsed["status"] == "ok"
    with caplog.at_level(logging.WARNING, logger="ibda.adapters.ibkr.flex.mapping"):
        flex_sections_to_canonical(parsed["sections"])

    multiplier_warnings = [
        r.message
        for r in caplog.records
        if r.levelno >= logging.WARNING and "multiplier" in r.message.lower()
    ]
    assert not multiplier_warnings, (
        f"Unexpected multiplier WARNING for STK row: {multiplier_warnings}"
    )


# ---------------------------------------------------------------------------
# Audit item #4: sub-second (µs) precision is NOT truncated by the Flex path.
# ---------------------------------------------------------------------------


def test_parse_dt_preserves_microseconds_with_offset() -> None:
    """_parse_dt keeps µs from ISO-8601 strings that include an explicit offset."""
    # 2026-06-02T10:31:00.123456-04:00 (EDT) → 2026-06-02T14:31:00.123456Z
    result = _parse_dt("2026-06-02T10:31:00.123456-04:00")
    assert result.microsecond == 123456, (
        f"Microseconds truncated: expected 123456, got {result.microsecond}"
    )
    assert result == datetime(2026, 6, 2, 14, 31, 0, 123456, tzinfo=timezone.utc)


def test_parse_dt_preserves_microseconds_without_offset() -> None:
    """_parse_dt keeps µs from ISO-8601 strings without an explicit offset (localized to ET)."""
    # 2026-06-02T10:31:00.654321 (no offset) → localized to EDT → 14:31:00.654321Z
    result = _parse_dt("2026-06-02T10:31:00.654321")
    assert result.microsecond == 654321, (
        f"Microseconds truncated: expected 654321, got {result.microsecond}"
    )


def test_arrow_path_preserves_timestamp_microseconds() -> None:
    """Sub-second precision survives the full pure-Python path: _parse_dt → Arrow TIMESTAMP_NS.

    This exercises the same path as builders.py (TIMESTAMP_NS column) without
    requiring a JVM.  The ISO-formatting half of builders._to_j_instant is tested
    separately via _format_instant_iso.
    """
    from ibda.adapters.ibkr.flex.arrow import arrow_table_from_rows

    # OPT XML with sub-second dateTime; multiplier=100 so no warning fires.
    xml = textwrap.dedent("""\
        <FlexQueryResponse queryName="Activity" type="AF">
        <FlexStatements count="1">
        <FlexStatement accountId="U9999" fromDate="2026-06-01" toDate="2026-06-08">
        <Trades>
          <Trade accountId="U9999" symbol="AAPL" assetCategory="STK"
            dateTime="2026-06-02T10:31:00.123456-04:00" buySell="BUY" quantity="10"
            tradePrice="200.00" ibCommission="-1.00" currency="USD"
            ibExecID="test-us-precision" conid="265598"/>
        </Trades>
        </FlexStatement>
        </FlexStatements>
        </FlexQueryResponse>
    """)
    parsed = parse_statement(xml)
    assert parsed["status"] == "ok"
    canon = flex_sections_to_canonical(parsed["sections"])
    execs = canon["execution"]
    assert len(execs) == 1

    # Verify _parse_dt preserved µs in the canonical dict
    ts: datetime = execs[0]["Timestamp"]
    assert ts.microsecond == 123456, (
        f"_parse_dt truncated microseconds in canonical dict: got {ts.microsecond}"
    )

    # Verify pyarrow TIMESTAMP_NS column also preserves µs (no truncation at the schema layer)
    table = arrow_table_from_rows(EXECUTION, execs)
    ts_from_arrow: datetime | None = table.column("Timestamp")[0].as_py()
    assert ts_from_arrow is not None
    assert ts_from_arrow.microsecond == 123456, (
        f"Arrow TIMESTAMP_NS column truncated microseconds: got {ts_from_arrow.microsecond}"
    )


def test_format_instant_iso_preserves_microseconds() -> None:
    """_format_instant_iso includes 6 sub-second digits — the ISO string passed to
    Deephaven's to_j_instant retains µs precision (builders.py pure-Python layer).
    """
    from ibda.adapters.deephaven.builders import _format_instant_iso

    dt = datetime(2026, 6, 2, 14, 31, 0, 123456, tzinfo=timezone.utc)
    iso = _format_instant_iso(dt)

    assert ".123456" in iso, (
        f"Expected 6-digit µs fraction in ISO string; got {iso!r}"
    )
    assert iso.endswith(" UTC"), f"Expected ' UTC' suffix for deephaven compatibility; got {iso!r}"


def test_format_instant_iso_naive_treated_as_utc() -> None:
    """A naive datetime is treated as UTC (project convention) by _format_instant_iso."""
    from ibda.adapters.deephaven.builders import _format_instant_iso

    dt_naive = datetime(2026, 6, 2, 14, 31, 0, 500000)
    iso = _format_instant_iso(dt_naive)

    assert ".500000" in iso
    assert iso.endswith(" UTC")


# ---------------------------------------------------------------------------
# Audit item #2b: unparseable multiplier value also logs a WARNING and defaults to 1.0.
# (Distinct from the absent-multiplier case above: exercises the float() parse-failure
# path in _map_execution, not the attribute-absent branch.)
# ---------------------------------------------------------------------------


def test_bad_multiplier_logs_warning_and_defaults_to_1(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unparseable multiplier value logs WARNING (with raw value) and defaults to 1.0.

    The warning makes the substitution traceable in structured logs, so corrupt
    Flex rows are not silently absorbed as valid contract-size-1 trades.
    This exercises the float() parse-failure path in _map_execution, distinct from
    the absent-multiplier (None) path tested above.
    """
    from ibda.adapters.ibkr.flex.mapping import _map_execution

    trade: dict[str, Any] = {
        "symbol": "AAPL",
        "date_time": "2026-06-02 10:31:00",
        "quantity": 100.0,
        "trade_price": 150.0,
        "buy_sell": "BUY",
        "multiplier": "NOT_A_NUMBER",
        "exec_id": "e1",
    }
    with caplog.at_level(logging.WARNING, logger="ibda.adapters.ibkr.flex.mapping"):
        result = _map_execution(trade)

    assert result is not None
    assert result["Multiplier"] == 1.0, "fallback to 1.0 must still apply"
    matching = [r for r in caplog.records if "NOT_A_NUMBER" in r.message]
    assert matching, (
        f"expected WARNING containing the raw multiplier value; got: {[r.message for r in caplog.records]}"
    )
    assert all(r.levelno >= logging.WARNING for r in matching)


# ---------------------------------------------------------------------------
# NAV Total==None -> row dropped (never 0.0).
# ---------------------------------------------------------------------------


def test_map_nav_returns_none_when_total_missing() -> None:
    from ibda.adapters.ibkr.flex.mapping import _map_nav

    row: dict[str, Any] = {
        "account": "U0000001",
        "report_date": "2026-06-02",
        "total": None,
        "cash": 100.0,
        "stock": 0.0,
    }
    assert _map_nav(row) is None


def test_map_nav_returns_row_when_total_present() -> None:
    from ibda.adapters.ibkr.flex.mapping import _map_nav

    row: dict[str, Any] = {
        "account": "U0000001",
        "report_date": "2026-06-02",
        "total": 100000.0,
        "cash": 50000.0,
        "stock": 50000.0,
    }
    result = _map_nav(row)
    assert result is not None
    assert result["Total"] == 100000.0


def test_flex_sections_to_canonical_drops_nav_row_missing_total(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A NAV row with no total attribute must be dropped, not coerced to 0.0."""
    xml = textwrap.dedent("""\
        <FlexQueryResponse queryName="Activity" type="AF">
        <FlexStatements count="1">
        <FlexStatement accountId="U9999" fromDate="2026-06-01" toDate="2026-06-02">
        <Trades/>
        <CashTransactions/>
        <EquitySummaryInBase>
        <EquitySummaryByReportDateInBase accountId="U9999" reportDate="20260601"
          cash="100000.00" stock="0.00" total="100000.00"/>
        <EquitySummaryByReportDateInBase accountId="U9999" reportDate="20260602"
          cash="150000.00" stock="0.00"/>
        </EquitySummaryInBase>
        </FlexStatement>
        </FlexStatements>
        </FlexQueryResponse>
    """)
    parsed = parse_statement(xml)
    assert parsed["status"] == "ok"
    with caplog.at_level(logging.WARNING, logger="ibda.adapters.ibkr.flex.mapping"):
        canon = flex_sections_to_canonical(parsed["sections"])

    assert len(canon["nav"]) == 1, "the row with no total= attribute must be dropped"
    assert canon["nav"][0]["Total"] == 100000.0
    warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("Total" in m for m in warnings), (
        f"expected a WARNING about the dropped NAV row; got: {warnings}"
    )


# ---------------------------------------------------------------------------
# Cash Amount==None -> row dropped (never 0.0).
# ---------------------------------------------------------------------------


def test_map_cash_returns_none_when_amount_missing() -> None:
    from ibda.adapters.ibkr.flex.mapping import _map_cash

    txn: dict[str, Any] = {
        "account": "U0000001",
        "date_time": "2026-06-02",
        "type": "Dividends",
        "symbol": "MSFT",
        "amount": None,
        "currency": "USD",
    }
    assert _map_cash(txn) is None


def test_map_cash_returns_row_when_amount_present() -> None:
    from ibda.adapters.ibkr.flex.mapping import _map_cash

    txn: dict[str, Any] = {
        "account": "U0000001",
        "date_time": "2026-06-02",
        "type": "Dividends",
        "symbol": "MSFT",
        "amount": 42.50,
        "currency": "USD",
    }
    result = _map_cash(txn)
    assert result is not None
    assert result["Amount"] == 42.50


def test_flex_sections_to_canonical_drops_cash_row_missing_amount(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A CashTransaction row with no amount attribute must be dropped, not coerced to 0.0."""
    xml = textwrap.dedent("""\
        <FlexQueryResponse queryName="Activity" type="AF">
        <FlexStatements count="1">
        <FlexStatement accountId="U9999" fromDate="2026-06-01" toDate="2026-06-02">
        <Trades/>
        <CashTransactions>
        <CashTransaction accountId="U9999" symbol="MSFT" type="Dividends"
          dateTime="2026-06-02" currency="USD" description="MSFT CASH DIVIDEND"/>
        <CashTransaction accountId="U9999" symbol="" type="Broker Interest Received"
          dateTime="2026-06-02" amount="3.21" currency="USD" description="CREDIT INT"/>
        </CashTransactions>
        </FlexStatement>
        </FlexStatements>
        </FlexQueryResponse>
    """)
    parsed = parse_statement(xml)
    assert parsed["status"] == "ok"
    with caplog.at_level(logging.WARNING, logger="ibda.adapters.ibkr.flex.mapping"):
        canon = flex_sections_to_canonical(parsed["sections"])

    assert len(canon["cash"]) == 1, "the row with no amount= attribute must be dropped"
    assert canon["cash"][0]["Amount"] == 3.21
    warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("Amount" in m for m in warnings), (
        f"expected a WARNING about the dropped cash row; got: {warnings}"
    )


# ---------------------------------------------------------------------------
# Execution Sym/Price/Qty missing -> row dropped, never a "ghost fill" (empty
# symbol, $0.00 price, or 0 quantity).
# ---------------------------------------------------------------------------


def test_map_execution_returns_none_when_symbol_missing() -> None:
    from ibda.adapters.ibkr.flex.mapping import _map_execution

    trade: dict[str, Any] = {
        "symbol": None,
        "date_time": "2026-06-02 10:31:00",
        "quantity": 100.0,
        "trade_price": 150.0,
        "buy_sell": "BUY",
    }
    assert _map_execution(trade) is None


def test_map_execution_returns_none_when_price_missing() -> None:
    from ibda.adapters.ibkr.flex.mapping import _map_execution

    trade: dict[str, Any] = {
        "symbol": "AAPL",
        "date_time": "2026-06-02 10:31:00",
        "quantity": 100.0,
        "trade_price": None,
        "buy_sell": "BUY",
    }
    assert _map_execution(trade) is None


def test_map_execution_returns_none_when_quantity_missing() -> None:
    from ibda.adapters.ibkr.flex.mapping import _map_execution

    trade: dict[str, Any] = {
        "symbol": "AAPL",
        "date_time": "2026-06-02 10:31:00",
        "quantity": None,
        "trade_price": 150.0,
        "buy_sell": "BUY",
    }
    assert _map_execution(trade) is None


def test_map_execution_dropped_row_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    from ibda.adapters.ibkr.flex.mapping import _map_execution

    trade: dict[str, Any] = {
        "symbol": None,
        "date_time": "2026-06-02 10:31:00",
        "quantity": 100.0,
        "trade_price": 150.0,
        "buy_sell": "BUY",
        "exec_id": "ghost-1",
    }
    with caplog.at_level(logging.WARNING, logger="ibda.adapters.ibkr.flex.mapping"):
        result = _map_execution(trade)

    assert result is None
    warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("ghost-1" in m for m in warnings), (
        f"expected a WARNING identifying the dropped row; got: {warnings}"
    )


def test_flex_sections_to_canonical_drops_execution_row_missing_symbol() -> None:
    """A Trade element with no symbol attribute must be dropped (no ghost fill)."""
    xml = textwrap.dedent("""\
        <FlexQueryResponse queryName="Activity" type="AF">
        <FlexStatements count="1">
        <FlexStatement accountId="U9999" fromDate="2026-06-01" toDate="2026-06-02">
        <Trades>
        <Trade accountId="U9999" assetCategory="STK" dateTime="2026-06-02 10:00:00"
          buySell="BUY" quantity="100" tradePrice="150.00" ibCommission="-1.00"
          currency="USD"/>
        <Trade accountId="U9999" symbol="AAPL" assetCategory="STK"
          dateTime="2026-06-02 10:01:00" buySell="BUY" quantity="10" tradePrice="150.00"
          ibCommission="-1.00" currency="USD" ibExecID="real-1"/>
        </Trades>
        <CashTransactions/>
        </FlexStatement>
        </FlexStatements>
        </FlexQueryResponse>
    """)
    parsed = parse_statement(xml)
    assert parsed["status"] == "ok"
    canon = flex_sections_to_canonical(parsed["sections"])
    assert len(canon["execution"]) == 1
    assert canon["execution"][0]["ExecId"] == "real-1"


# ---------------------------------------------------------------------------
# OpenClose flows through from Flex's openCloseIndicator.
# ---------------------------------------------------------------------------


def test_execution_open_close_flows_through() -> None:
    canon = flex_sections_to_canonical(_sections())
    execs = canon["execution"]
    sell = next(r for r in execs if r["Side"] == "SELL")
    buy = next(r for r in execs if r["Side"] == "BUY")
    assert sell["OpenClose"] == "C"
    assert buy["OpenClose"] is None


# ---------------------------------------------------------------------------
# Currency flows through from the Trade element.
# ---------------------------------------------------------------------------


def test_execution_currency_flows_through() -> None:
    canon = flex_sections_to_canonical(_sections())
    for row in canon["execution"]:
        assert row["Currency"] == "USD"


# ---------------------------------------------------------------------------
# OrderId flows through from ibOrderID (Flex path).
# ---------------------------------------------------------------------------


def test_execution_order_id_flows_through_from_ib_order_id() -> None:
    """report_full.xml's SELL carries ibOrderID='9988771'; the BUY has none -> None."""
    canon = flex_sections_to_canonical(_sections())
    execs = canon["execution"]
    sell = next(r for r in execs if r["Side"] == "SELL")
    buy = next(r for r in execs if r["Side"] == "BUY")
    assert sell["OrderId"] == 9988771
    assert isinstance(sell["OrderId"], int)
    assert buy["OrderId"] is None


def test_parse_trades_extracts_order_id() -> None:
    """Raw trade dicts carry the raw order_id string (before int coercion)."""
    trades = _sections()["trades"]
    tagged = next(t for t in trades if t["order_reference"] is not None)
    assert tagged["order_id"] == "9988771"


def test_map_execution_order_id_non_digit_yields_none() -> None:
    """A non-numeric order_id string must not raise — falls back to None."""
    from ibda.adapters.ibkr.flex.mapping import _map_execution

    trade: dict[str, Any] = {
        "symbol": "AAPL",
        "date_time": "2026-06-02 10:31:00",
        "quantity": 100.0,
        "trade_price": 150.0,
        "buy_sell": "BUY",
        "order_id": "not-a-number",
    }
    result = _map_execution(trade)
    assert result is not None
    assert result["OrderId"] is None


# ---------------------------------------------------------------------------
# levelOfDetail filtering — keep only EXECUTION-level rows.
# ---------------------------------------------------------------------------

_MULTI_LEVEL_FIXTURE = Path(__file__).parent / "fixtures" / "flex" / "report_multi_level.xml"


def _multi_level_sections() -> dict[str, Any]:
    parsed = parse_statement(_MULTI_LEVEL_FIXTURE.read_text())
    assert parsed["status"] == "ok"
    result: dict[str, Any] = parsed["sections"]
    return result


def test_multi_level_fixture_keeps_only_execution_rows() -> None:
    """report_multi_level.xml has 2 EXECUTION + 1 ORDER + 1 CLOSED_LOT Trade
    elements; only the 2 EXECUTION-level rows must survive."""
    trades = _multi_level_sections()["trades"]
    assert len(trades) == 2
    assert all(t["quantity"] == 60.0 or t["quantity"] == 40.0 for t in trades)
    # The ORDER-level (qty=100) and CLOSED_LOT-level (qty=-100, SELL) rows must
    # be dropped — no SELL side present since only the two BUY fills survive.
    assert all(t["buy_sell"] == "BUY" for t in trades)


def test_multi_level_fixture_canonical_execution_row_count() -> None:
    canon = flex_sections_to_canonical(_multi_level_sections())
    assert len(canon["execution"]) == 2


def test_multi_level_fixture_logs_warning_for_mixed_levels(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="ibda.adapters.ibkr.flex.parse"):
        parsed = parse_statement(_MULTI_LEVEL_FIXTURE.read_text())
    assert parsed["status"] == "ok"
    warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("levelOfDetail" in m for m in warnings), (
        f"expected a WARNING about mixed levelOfDetail values; got: {warnings}"
    )


def test_single_level_fixture_unaffected_by_level_of_detail_filter() -> None:
    """report_full.xml has no levelOfDetail attribute at all (single-level Flex
    statement) — both trades must still be kept (regression guard)."""
    canon = flex_sections_to_canonical(_sections())
    assert len(canon["execution"]) == 2


# ---------------------------------------------------------------------------
# Fix D: levelOfDetail filter hardening — case-insensitivity + the
# single-level-non-EXECUTION hole (an ORDER-only statement silently drops
# every row without ever tripping the mixed-levels warning, since it only
# ever sees ONE distinct level).
# ---------------------------------------------------------------------------

_ORDER_ONLY_XML = textwrap.dedent("""\
    <FlexQueryResponse queryName="Activity" type="AF">
    <FlexStatements count="1">
    <FlexStatement accountId="U0000003" fromDate="2026-06-01" toDate="2026-06-08">
    <Trades>
    <Trade accountId="U0000003" symbol="TSLA" assetCategory="STK" tradeDate="2026-06-02"
      dateTime="2026-06-02 10:00:01" buySell="BUY" quantity="100" tradePrice="250.04"
      proceeds="-25004.00" ibCommission="-2.00" currency="USD"
      ibOrderID="55501" levelOfDetail="order"/>
    </Trades>
    <CashTransactions/>
    </FlexStatement>
    </FlexStatements>
    </FlexQueryResponse>
""")


def test_order_only_statement_drops_all_rows() -> None:
    """A single-level, ORDER-only statement has distinct_levels=={'ORDER'} (len==1),
    so the mixed-levels warning never fires — but every row must still be dropped
    (no EXECUTION-level row exists to keep)."""
    parsed = parse_statement(_ORDER_ONLY_XML)
    assert parsed["status"] == "ok"
    assert parsed["sections"]["trades"] == []


def test_order_only_statement_logs_all_rows_dropped_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Distinct from the mixed-levels warning: fires when levelOfDetail values are
    present but ZERO EXECUTION-level rows survive the filter."""
    with caplog.at_level(logging.WARNING, logger="ibda.adapters.ibkr.flex.parse"):
        parsed = parse_statement(_ORDER_ONLY_XML)
    assert parsed["status"] == "ok"
    warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("none is" in m and "EXECUTION" in m for m in warnings), (
        f"expected a WARNING that all rows were dropped for lack of an EXECUTION "
        f"level; got: {warnings}"
    )
    # The mixed-levels warning (len(distinct_levels) > 1) must NOT fire here —
    # this statement has exactly one distinct level ("ORDER").
    assert not any("mixes multiple" in m for m in warnings), (
        f"mixed-levels warning must not fire for a single-level statement; got: {warnings}"
    )


def test_level_of_detail_match_is_case_insensitive() -> None:
    """levelOfDetail='EXECUTION' in any casing (e.g. lowercase 'execution') must
    be recognized — the comparison upper-cases before matching."""
    xml = textwrap.dedent("""\
        <FlexQueryResponse queryName="Activity" type="AF">
        <FlexStatements count="1">
        <FlexStatement accountId="U0000004" fromDate="2026-06-01" toDate="2026-06-08">
        <Trades>
        <Trade accountId="U0000004" symbol="AAPL" assetCategory="STK"
          dateTime="2026-06-02 10:00:00" buySell="BUY" quantity="10" tradePrice="200.00"
          ibCommission="-1.00" currency="USD" ibExecID="case-insensitive-1"
          levelOfDetail="execution"/>
        </Trades>
        <CashTransactions/>
        </FlexStatement>
        </FlexStatements>
        </FlexQueryResponse>
    """)
    parsed = parse_statement(xml)
    assert parsed["status"] == "ok"
    trades = parsed["sections"]["trades"]
    assert len(trades) == 1
    assert trades[0]["exec_id"] == "case-insensitive-1"


# ---------------------------------------------------------------------------
# Fix #7: Arrow producer TIMESTAMP parity with the Deephaven builder — a
# non-datetime value in a TIMESTAMP_NS column is nulled, not raised.
# ---------------------------------------------------------------------------


def test_arrow_table_from_rows_nulls_non_datetime_timestamp() -> None:
    """A non-datetime value in a TIMESTAMP_NS column becomes a null, mirroring
    ibda.adapters.deephaven.builders.table_from_rows's `isinstance(v, datetime)
    else None` behavior — pa.array(..., type=timestamp) would otherwise raise
    ArrowInvalid on the very first non-datetime value."""
    from ibda.adapters.ibkr.flex.arrow import arrow_table_from_rows

    rows: list[dict[str, Any]] = [
        {
            "ExecId": "e1",
            "Timestamp": "not-a-datetime",  # grammar-valid dict value, wrong type
            "Sym": "AAPL",
            "Side": "BUY",
            "Qty": 10.0,
            "Price": 100.0,
        },
        {
            "ExecId": "e2",
            "Timestamp": datetime(2026, 6, 2, 14, 31, 0, tzinfo=timezone.utc),
            "Sym": "AAPL",
            "Side": "SELL",
            "Qty": 5.0,
            "Price": 105.0,
        },
    ]
    table = arrow_table_from_rows(EXECUTION, rows)
    timestamps = table.column("Timestamp").to_pylist()
    assert timestamps[0] is None, f"expected null for non-datetime Timestamp, got {timestamps[0]!r}"
    assert timestamps[1] is not None


# ---------------------------------------------------------------------------
# Blank Side / transfer-direction defaults are silent — WARNING guards
# make them visible (no behavior change).
# ---------------------------------------------------------------------------


def test_map_execution_blank_buy_sell_logs_warning_and_still_emits_empty_side(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A <Trade> missing/blank buySell logs a WARNING and still emits Side=='' unchanged."""
    from ibda.adapters.ibkr.flex.mapping import _map_execution

    trade: dict[str, Any] = {
        "symbol": "AAPL",
        "date_time": "2026-06-02 10:31:00",
        "quantity": 100.0,
        "trade_price": 150.0,
        "buy_sell": None,
        "exec_id": "e1",
    }
    with caplog.at_level(logging.WARNING, logger="ibda.adapters.ibkr.flex.mapping"):
        result = _map_execution(trade)

    assert result is not None
    assert result["Side"] == "", "Side must remain '' — this test only checks visibility"
    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("buySell" in m for m in warning_messages), (
        f"expected a WARNING naming buySell; got: {warning_messages}"
    )
    assert any("AAPL" in m for m in warning_messages), (
        f"expected the warning to name the symbol; got: {warning_messages}"
    )


def test_map_execution_normal_buy_sell_does_not_log_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A normal BUY/SELL trade must NOT emit the blank-buySell warning."""
    from ibda.adapters.ibkr.flex.mapping import _map_execution

    trade: dict[str, Any] = {
        "symbol": "AAPL",
        "date_time": "2026-06-02 10:31:00",
        "quantity": 100.0,
        "trade_price": 150.0,
        "buy_sell": "BUY",
        "exec_id": "e1",
    }
    with caplog.at_level(logging.WARNING, logger="ibda.adapters.ibkr.flex.mapping"):
        result = _map_execution(trade)

    assert result is not None
    assert result["Side"] == "BUY"
    buy_sell_warnings = [
        r.message
        for r in caplog.records
        if r.levelno >= logging.WARNING and "buySell" in r.message
    ]
    assert not buy_sell_warnings, f"unexpected buySell WARNING: {buy_sell_warnings}"


def test_map_transfer_absent_direction_logs_warning_and_still_defaults_to_in(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A transfer missing 'direction' logs a WARNING and still defaults to IN (+magnitude)."""
    from ibda.adapters.ibkr.flex.mapping import _map_transfer

    transfer: dict[str, Any] = {
        "symbol": "GOOGL",
        "date_time": "2026-06-02",
        "direction": None,
        "positionAmountInBase": 5000.0,
    }
    with caplog.at_level(logging.WARNING, logger="ibda.adapters.ibkr.flex.mapping"):
        result = _map_transfer(transfer)

    assert result is not None
    assert result["Amount"] == 5000.0, "must still default to IN (+magnitude)"
    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("direction" in m for m in warning_messages), (
        f"expected a WARNING naming direction; got: {warning_messages}"
    )
    assert any("GOOGL" in m for m in warning_messages), (
        f"expected the warning to name the symbol; got: {warning_messages}"
    )


def test_map_transfer_normal_in_out_does_not_log_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A transfer with an explicit IN or OUT direction must NOT emit the absent-direction warning."""
    from ibda.adapters.ibkr.flex.mapping import _map_transfer

    with caplog.at_level(logging.WARNING, logger="ibda.adapters.ibkr.flex.mapping"):
        in_result = _map_transfer({
            "symbol": "GOOGL",
            "date_time": "2026-06-02",
            "direction": "IN",
            "positionAmountInBase": 5000.0,
        })
        out_result = _map_transfer({
            "symbol": "GOOGL",
            "date_time": "2026-06-02",
            "direction": "OUT",
            "positionAmountInBase": 5000.0,
        })

    assert in_result is not None and in_result["Amount"] == 5000.0
    assert out_result is not None and out_result["Amount"] == -5000.0
    direction_warnings = [
        r.message
        for r in caplog.records
        if r.levelno >= logging.WARNING and "direction" in r.message
    ]
    assert not direction_warnings, f"unexpected direction WARNING: {direction_warnings}"


# ---------------------------------------------------------------------------
# Overlapping <FlexStatement> blocks
#
# parse_statement deliberately concatenates ALL statement blocks, which makes
# overlapping date ranges reachable — the exact condition _dedupe_cash_rows was written
# for. The identical argument applies to <Trade> elements, and executions were never
# deduped, so every fill in an overlap doubled Qty, Commission and RealizedPnl.
# ---------------------------------------------------------------------------


def test_a_fill_reported_by_two_overlapping_statements_is_counted_once() -> None:
    from ibda.adapters.ibkr.flex.mapping import _dedupe_execution_rows

    def _row(exec_id: str) -> dict[str, object]:
        return {
            "ExecId": exec_id, "Account": "U111", "Sym": "AAPL", "Side": "BUY",
            "Qty": 100.0, "Price": 200.0, "Commission": -1.0, "RealizedPnl": 50.0,
        }

    deduped = _dedupe_execution_rows([_row("0000e0d5.1"), _row("0000e0d5.1")])

    assert len(deduped) == 1, "the same ibExecID reported twice must collapse to one fill"
    assert sum(float(r["Qty"]) for r in deduped) == 100.0
    assert sum(float(r["Commission"]) for r in deduped) == -1.0
    assert sum(float(r["RealizedPnl"]) for r in deduped) == 50.0


def test_the_canonical_mapper_actually_applies_the_execution_dedupe() -> None:
    """End-to-end through flex_sections_to_canonical, not the helper in isolation.

    Written this way deliberately: a test that calls _dedupe_execution_rows directly
    still passes when the call is removed from the return dict, which is exactly the
    regression that matters. The overlap is simulated by duplicating the parsed trades,
    which is what concatenating two overlapping <FlexStatement> blocks produces.
    """
    sections = _sections()
    baseline = len(flex_sections_to_canonical(sections)["execution"])

    overlapped = dict(sections)
    overlapped["trades"] = list(sections["trades"]) + list(sections["trades"])
    canon = flex_sections_to_canonical(overlapped)

    assert len(canon["execution"]) == baseline, (
        "duplicating the trades (an overlapping statement range) changed the canonical "
        f"execution row count from {baseline} to {len(canon['execution'])} — the dedupe "
        "is not wired into flex_sections_to_canonical"
    )
    exec_ids = [r["ExecId"] for r in canon["execution"]]
    assert len(exec_ids) == len(set(exec_ids))


def test_partial_fills_of_one_order_are_not_collapsed() -> None:
    """The safety condition for the dedupe above.

    IBKR gives each partial fill of an order a DISTINCT ibExecID (`…01.01`, `…02.01`),
    and parse.py filters to levelOfDetail="EXECUTION" so ORDER/CLOSED_LOT aggregate
    siblings never reach the mapper. If either were untrue, de-duplicating by ExecId
    would delete real fills.
    """
    from ibda.adapters.ibkr.flex.mapping import _dedupe_execution_rows

    rows = [
        {"ExecId": "0000e0d5.01.01", "Qty": 40.0},
        {"ExecId": "0000e0d5.02.01", "Qty": 60.0},
    ]
    assert len(_dedupe_execution_rows(rows)) == 2
    assert sum(float(r["Qty"]) for r in _dedupe_execution_rows(rows)) == 100.0


# ---------------------------------------------------------------------------
# 1970-epoch anchoring
#
# _parse_dt's own docstring states the rule: a caller "whose Timestamp decides which
# period a money movement belongs to" must use _parse_dt_or_none and drop the row.
# _map_cash follows it; _map_nav and _map_transfer did not.
# ---------------------------------------------------------------------------


def test_a_nav_row_with_an_unparseable_date_is_dropped_not_anchored_to_1970() -> None:
    """NAV is the DENOMINATOR of every return, so an epoch row rewrites the series.

    _nav_series sorts ascending, so the fabricated 1970 row lands first and becomes
    starting_nav; the 56-year gap then counts as one ordinary period.
    """
    from ibda.adapters.ibkr.flex.mapping import _map_nav

    assert _map_nav({"account": "U111", "report_date": "20260602", "total": 1000.0}) is not None
    assert _map_nav({"account": "U111", "report_date": "06/02/2026", "total": 1000.0}) is None
    assert _map_nav({"account": "U111", "report_date": None, "total": 1000.0}) is None


def test_a_transfer_with_an_unparseable_date_is_dropped_not_anchored_to_1970() -> None:
    """A transfer keyed to 1970 matches no NAV point, so it is never stripped from
    returns and the settlement day books the in-kind value as a trading gain."""
    from ibda.adapters.ibkr.flex.mapping import _map_transfer

    base: dict[str, object] = {
        "kind": "transfer", "account": "U111", "symbol": "AAPL", "direction": "IN",
        "quantity": None, "positionAmount": None, "positionAmountInBase": 50000.0,
        "transferPrice": None, "cashTransfer": None, "currency": "USD",
    }
    assert _map_transfer({**base, "date_time": "20260602"}) is not None
    assert _map_transfer({**base, "date_time": "not-a-date"}) is None
    assert _map_transfer({**base, "date_time": None}) is None


# ---------------------------------------------------------------------------
# NAV de-duplication across overlapping <FlexStatement> blocks.
#
# NAV was the only one of the three canonical tables left un-de-duplicated, and it is the
# one whose duplicates do the most damage: a repeated report date fabricates a zero-length
# return period AND re-subtracts that date's external flow a second time.
# ---------------------------------------------------------------------------
_OVERLAPPING_NAV_XML = textwrap.dedent("""\
    <FlexQueryResponse queryName="Activity" type="AF">
    <FlexStatements count="2">
    <FlexStatement accountId="U9999999" fromDate="2026-06-01" toDate="2026-06-03" period="Custom">
    <EquitySummaryByReportDateInBase accountId="U9999999" reportDate="2026-06-01" total="100000"/>
    <EquitySummaryByReportDateInBase accountId="U9999999" reportDate="2026-06-02" total="101000"/>
    <EquitySummaryByReportDateInBase accountId="U9999999" reportDate="2026-06-03" total="102000"/>
    </FlexStatement>
    <FlexStatement accountId="U9999999" fromDate="2026-06-02" toDate="2026-06-04" period="Custom">
    <EquitySummaryByReportDateInBase accountId="U9999999" reportDate="2026-06-02" total="101000"/>
    <EquitySummaryByReportDateInBase accountId="U9999999" reportDate="2026-06-03" total="102000"/>
    <EquitySummaryByReportDateInBase accountId="U9999999" reportDate="2026-06-04" total="103000"/>
    </FlexStatement>
    </FlexStatements>
    </FlexQueryResponse>
""")


def test_nav_rows_are_deduped_across_overlapping_statements(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Two statements sharing report dates must yield one NAV row per date.

    `ibda/schema/nav.py` declares the contract "One row per report date", and
    `analytics.benchmark._returns_by_date` treats a duplicate date as *fatal* while
    `analytics.performance` silently mis-reported it — the two disagreed about whether the
    same table was even valid.
    """
    parsed = parse_statement(_OVERLAPPING_NAV_XML)
    assert parsed["status"] == "ok"
    # The duplication is real at the parse layer: both statements are read in full.
    assert len(parsed["sections"]["nav"]) == 6

    with caplog.at_level(logging.WARNING, logger="ibda.adapters.ibkr.flex.mapping"):
        canon = flex_sections_to_canonical(parsed["sections"])

    dates = [r["Timestamp"].date() for r in canon["nav"]]
    assert len(dates) == 4, f"expected one row per report date, got {dates}"
    assert len(set(dates)) == len(dates), f"duplicate report dates survived: {dates}"
    assert "Dropped 2 duplicate NAV row(s)" in caplog.text


def test_duplicate_nav_dates_no_longer_distort_performance() -> None:
    """The metrics, not just the row count — this is what the defect actually moved.

    Before de-duplication the overlapping report produced extra zero-length periods, so
    num_periods and every metric derived from the return series were wrong with no error
    anywhere. Both statements describe the *same* four days, so the performance computed
    from the overlapping report must equal the clean one exactly.
    """
    clean_xml = textwrap.dedent("""\
        <FlexQueryResponse queryName="Activity" type="AF">
        <FlexStatements count="1">
        <FlexStatement accountId="U9999999" fromDate="2026-06-01" toDate="2026-06-04" period="Custom">
        <EquitySummaryByReportDateInBase accountId="U9999999" reportDate="2026-06-01" total="100000"/>
        <EquitySummaryByReportDateInBase accountId="U9999999" reportDate="2026-06-02" total="101000"/>
        <EquitySummaryByReportDateInBase accountId="U9999999" reportDate="2026-06-03" total="102000"/>
        <EquitySummaryByReportDateInBase accountId="U9999999" reportDate="2026-06-04" total="103000"/>
        </FlexStatement>
        </FlexStatements>
        </FlexQueryResponse>
    """)
    clean_perf = performance_from_sections(parse_statement(clean_xml)["sections"])
    overlap_perf = performance_from_sections(parse_statement(_OVERLAPPING_NAV_XML)["sections"])

    assert overlap_perf.num_periods == clean_perf.num_periods == 3
    assert overlap_perf.cumulative_return == pytest.approx(clean_perf.cumulative_return)
    assert overlap_perf.max_drawdown == pytest.approx(clean_perf.max_drawdown)


# ---------------------------------------------------------------------------
# _map_execution: tradeDate fallback when the Trades section carries no Date/Time.
# ---------------------------------------------------------------------------
_TRADES_WITHOUT_DATETIME_XML = textwrap.dedent("""\
    <FlexQueryResponse queryName="Activity" type="AF">
    <FlexStatements count="1">
    <FlexStatement accountId="U9999999" fromDate="2026-06-02" toDate="2026-06-03" period="Custom">
    <Trades>
      <Trade accountId="U9999999" symbol="AAPL" tradeDate="2026-06-02" quantity="100"
             tradePrice="200" ibCommission="-1" buySell="BUY" currency="USD" assetCategory="STK"/>
      <Trade accountId="U9999999" symbol="AAPL" tradeDate="2026-06-02" quantity="100"
             tradePrice="200" ibCommission="-1" buySell="BUY" currency="USD" assetCategory="STK"/>
      <Trade accountId="U9999999" symbol="AAPL" tradeDate="2026-06-03" quantity="-200"
             tradePrice="210" ibCommission="-2" buySell="SELL" currency="USD" assetCategory="STK"/>
    </Trades>
    </FlexStatement>
    </FlexStatements>
    </FlexQueryResponse>
""")


def test_trade_date_is_used_when_the_trades_section_has_no_datetime(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An absent dateTime must not epoch-anchor the fill, and must warn.

    `_parse_dt` warns only when its input is TRUTHY, so an empty dateTime substituted the
    1970 epoch in total silence. `parse.py` had already extracted tradeDate; the mapper
    simply never read it.
    """
    parsed = parse_statement(_TRADES_WITHOUT_DATETIME_XML)
    assert parsed["status"] == "ok"

    with caplog.at_level(logging.WARNING, logger="ibda.adapters.ibkr.flex.mapping"):
        canon = flex_sections_to_canonical(parsed["sections"])

    years = {r["Timestamp"].year for r in canon["execution"]}
    assert years == {2026}, f"fills were epoch-anchored instead of dated: {years}"
    assert "falling back to tradeDate" in caplog.text


def test_trade_date_fallback_restores_ordering_but_not_identity() -> None:
    """What the fallback does and does not fix — stated precisely, because it matters.

    It DOES restore day resolution, so fills sort correctly and round-trip reconstruction
    sees BUY-then-SELL rather than three fills stacked on 1970-01-01.

    It does NOT make two genuinely identical same-day fills distinguishable: with no
    ibExecID, and identical Account/Sym/Date/Qty/Price, the synthetic id is by construction
    the same and `_dedupe_execution_rows` collapses them — warning that it cannot tell a
    duplicate from two real fills and naming ibExecID as the resolution. That is a
    documented, warned limitation of the Flex query shape, not something a date fallback
    can repair. Asserting otherwise would encode a fix that does not exist.
    """
    canon = flex_sections_to_canonical(parse_statement(_TRADES_WITHOUT_DATETIME_XML)["sections"])
    rows = sorted(canon["execution"], key=lambda r: r["Timestamp"])

    # Ordering is restored: the buy precedes the sell on a real calendar.
    assert [r["Side"] for r in rows] == ["BUY", "SELL"]
    assert rows[0]["Timestamp"].date() < rows[1]["Timestamp"].date()
    assert all(r["Timestamp"].year == 2026 for r in rows)


def test_distinct_same_day_fills_are_kept_once_dated() -> None:
    """Two same-day fills that differ in any visible field both survive.

    This is the half the date fallback genuinely rescues. Before it every fill collapsed
    onto the 1970 epoch, so the synthetic fingerprint carried no date at all and fills on
    DIFFERENT days could collide with each other too. With the date restored, only rows
    that are identical in every visible field remain ambiguous.
    """
    xml = textwrap.dedent("""\
        <FlexQueryResponse queryName="Activity" type="AF">
        <FlexStatements count="1">
        <FlexStatement accountId="U9999999" fromDate="2026-06-02" toDate="2026-06-03" period="Custom">
        <Trades>
          <Trade accountId="U9999999" symbol="AAPL" tradeDate="2026-06-02" quantity="100"
                 tradePrice="200" ibCommission="-1" buySell="BUY" currency="USD" assetCategory="STK"/>
          <Trade accountId="U9999999" symbol="AAPL" tradeDate="2026-06-02" quantity="100"
                 tradePrice="201" ibCommission="-1" buySell="BUY" currency="USD" assetCategory="STK"/>
          <Trade accountId="U9999999" symbol="AAPL" tradeDate="2026-06-03" quantity="-200"
                 tradePrice="210" ibCommission="-2" buySell="SELL" currency="USD" assetCategory="STK"/>
        </Trades>
        </FlexStatement>
        </FlexStatements>
        </FlexQueryResponse>
    """)
    canon = flex_sections_to_canonical(parse_statement(xml)["sections"])
    buys = [r for r in canon["execution"] if r["Side"] == "BUY"]
    assert len(buys) == 2, f"two distinguishable buys must both survive; got {buys}"
    assert sum(r["Qty"] for r in buys) == pytest.approx(200.0)
