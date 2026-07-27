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

from ibda.adapters.ibkr.flex.mapping import _parse_dt, flex_sections_to_canonical
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
    """Fixture has no ibExecID/tradeID, so a deterministic synthetic id is used."""
    canon = flex_sections_to_canonical(_sections())
    for row in canon["execution"]:
        assert row["ExecId"]  # non-empty
        # Synthetic id should embed the symbol so it's recognisable.
        assert "AAPL" in row["ExecId"]


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
