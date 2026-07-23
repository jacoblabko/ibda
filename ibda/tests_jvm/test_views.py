"""JVM test: apply_canonical_view — builds conformant canonical tables.

For each IBKR spec: constructs a raw-shaped in-process table, applies
apply_canonical_view, snapshots through the port, and asserts schema.validate
passes and that renamed / null-filled columns have the correct values/nullness.

Run with:
    uv run pytest ibda/tests_jvm/test_views.py -q
"""
from __future__ import annotations

from typing import Any

import pytest

import ibda
from ibda.adapters.deephaven.views import apply_canonical_view
from ibda.adapters.ibkr.specs import CANONICAL_SPECS
from ibda.schema import BAR, COMMISSION, DEFINITION, EXECUTION, ORDER, POSITION, QUOTE, TRADE


# ---------------------------------------------------------------------------
# Helpers to build raw-shaped tables for each spec
# ---------------------------------------------------------------------------

def _raw_position_table() -> Any:
    from deephaven import new_table
    from deephaven.column import double_col, long_col, string_col

    return new_table([
        string_col("Account", ["DU1"]),
        long_col("ContractId", [12345]),
        string_col("Symbol", ["AAPL"]),
        string_col("SecType", ["STK"]),
        double_col("Position", [100.0]),
        double_col("AvgCost", [150.0]),
        # Option-identity columns — accounts_positions carries these natively
        # per position (deephaven-ib's logger_contract logs the full Contract
        # on every position update); null for a plain STK row. Multiplier
        # arrives as a double here (unlike execution/definition's Multiplier).
        string_col("Right", [None]),
        double_col("Strike", [None]),
        string_col("LastTradeDateOrContractMonth", [None]),
        string_col("LocalSymbol", ["AAPL"]),
        double_col("Multiplier", [None]),
        string_col("Currency", ["USD"]),
        string_col("Exchange", ["SMART"]),
        # Raw table does NOT have MarketPrice / MarketValue / UnrealizedPnl
    ])


def _raw_position_table_option() -> Any:
    """An OPT position row — Right/Strike/LastTradeDateOrContractMonth/
    LocalSymbol/Multiplier all populated, mirroring a live accounts_positions
    row for a held option contract."""
    from deephaven import new_table
    from deephaven.column import double_col, long_col, string_col

    return new_table([
        string_col("Account", ["DU1"]),
        long_col("ContractId", [67890]),
        string_col("Symbol", ["AAPL"]),
        string_col("SecType", ["OPT"]),
        double_col("Position", [-2.0]),
        double_col("AvgCost", [530.0]),
        string_col("Right", ["C"]),
        double_col("Strike", [200.0]),
        string_col("LastTradeDateOrContractMonth", ["20260717"]),
        string_col("LocalSymbol", ["AAPL  260717C00200000"]),
        double_col("Multiplier", [100.0]),
        string_col("Currency", ["USD"]),
        string_col("Exchange", ["SMART"]),
    ])


def _raw_execution_table(exec_id: str = "exec001", side: str = "BUY") -> Any:
    from deephaven import new_table
    from deephaven.column import datetime_col, double_col, long_col, string_col
    from deephaven.time import to_j_instant

    t0 = to_j_instant("2026-06-10T15:00:00 UTC")
    return new_table([
        string_col("ExecId", [exec_id]),
        datetime_col("Timestamp", [t0]),
        string_col("Account", ["DU1"]),
        long_col("ContractId", [12345]),
        long_col("OrderId", [987]),
        string_col("OrderRef", ["STRAT-A-AAPL"]),
        string_col("Symbol", ["AAPL"]),
        string_col("SecType", ["STK"]),
        string_col("Side", [side]),
        double_col("Shares", [100.0]),
        double_col("Price", [155.0]),
        # Multiplier arrives as string in orders_exec_details; cast to double in view.
        string_col("Multiplier", ["1"]),
        string_col("ExecutionExchange", ["NASDAQ"]),
        string_col("Currency", ["USD"]),
        # Commission/Liquidity are NOT raw orders_exec_details columns — Commission
        # now comes from the join to orders_exec_commission_report (see
        # _raw_commission_table below); Liquidity has no source anywhere and stays
        # null-filled by apply_canonical_view. OpenClose has no source anywhere
        # either (Flex-only field) and also stays null-filled.
    ])


def _raw_commission_table(
    exec_id: str = "exec001",
    commission: float = 1.25,
    realized_pnl: float = 0.0,
    receive_time: str = "2026-06-10T15:00:05 UTC",
) -> Any:
    """A raw ``orders_exec_commission_report``-shaped table — one row.

    Mirrors ``ibda/tests_jvm/test_commission_news_errors_views.py``'s
    ``_raw_commission_table`` (kept local here so this file's execution-spec
    tests don't cross-import from another test module).
    """
    from deephaven import new_table
    from deephaven.column import datetime_col, double_col, string_col
    from deephaven.time import to_j_instant

    t0 = to_j_instant(receive_time)
    return new_table([
        string_col("ExecId", [exec_id]),
        datetime_col("ReceiveTime", [t0]),
        double_col("Commission", [commission]),
        string_col("Currency", ["USD"]),
        double_col("RealizedPNL", [realized_pnl]),
        double_col("Yield", [0.0]),
        string_col("YieldRedemptionDate", [""]),
    ])


def _raw_definition_table() -> Any:
    from deephaven import new_table
    from deephaven.column import double_col, long_col, string_col

    return new_table([
        long_col("ContractId", [12345]),
        string_col("Symbol", ["AAPL"]),
        string_col("SecType", ["STK"]),
        string_col("PrimaryExchange", ["NASDAQ"]),
        string_col("Currency", ["USD"]),
        # Multiplier is a STRING in contracts_details (verified against a live session)
        string_col("Multiplier", ["1"]),
        # Option/future identity columns — null for a plain STK contract.
        string_col("Right", [None]),
        double_col("Strike", [None]),
        string_col("LastTradeDateOrContractMonth", [None]),
        string_col("LocalSymbol", ["AAPL"]),
    ])


def _raw_definition_table_option() -> Any:
    """An OPT contract row — Right/Strike/LastTradeDateOrContractMonth/
    LocalSymbol all populated, mirroring a live contracts_details row for an
    option contract."""
    from deephaven import new_table
    from deephaven.column import double_col, long_col, string_col

    return new_table([
        long_col("ContractId", [67890]),
        string_col("Symbol", ["AAPL"]),
        string_col("SecType", ["OPT"]),
        string_col("PrimaryExchange", ["CBOE"]),
        string_col("Currency", ["USD"]),
        string_col("Multiplier", ["100"]),
        string_col("Right", ["C"]),
        double_col("Strike", [200.0]),
        string_col("LastTradeDateOrContractMonth", ["20260717"]),
        string_col("LocalSymbol", ["AAPL  260717C00200000"]),
    ])


def _raw_definition_table_null_multiplier() -> Any:
    from deephaven import new_table
    from deephaven.column import double_col, long_col, string_col

    return new_table([
        long_col("ContractId", [99999]),
        string_col("Symbol", ["MSFT"]),
        string_col("SecType", ["STK"]),
        string_col("PrimaryExchange", ["NASDAQ"]),
        string_col("Currency", ["USD"]),
        # Empty string Multiplier → NULL_DOUBLE after cast
        string_col("Multiplier", [""]),
        string_col("Right", [None]),
        double_col("Strike", [None]),
        string_col("LastTradeDateOrContractMonth", [None]),
        string_col("LocalSymbol", ["MSFT"]),
    ])


def _raw_definition_table_double_multiplier() -> Any:
    """Mirror live contracts_details: Multiplier is already a double column.

    The live IBKR account's contracts_details table delivers Multiplier as
    a numeric double, not a string.  apply_canonical_view must detect this
    via meta_table and skip the String-parse path entirely — otherwise
    Groovy throws FormulaCompilationException (no .isEmpty()/.parseDouble on
    a double primitive).
    """
    from deephaven import new_table
    from deephaven.column import double_col, long_col, string_col

    return new_table([
        long_col("ContractId", [54321]),
        string_col("Symbol", ["SPY"]),
        string_col("SecType", ["STK"]),
        string_col("PrimaryExchange", ["ARCA"]),
        string_col("Currency", ["USD"]),
        # Already a double — as seen in live contracts_details
        double_col("Multiplier", [1.0]),
        string_col("Right", [None]),
        double_col("Strike", [None]),
        string_col("LastTradeDateOrContractMonth", [None]),
        string_col("LocalSymbol", ["SPY"]),
    ])


def _raw_quote_table() -> Any:
    from deephaven import new_table
    from deephaven.column import datetime_col, double_col, string_col
    from deephaven.time import to_j_instant

    t0 = to_j_instant("2026-06-10T15:00:00 UTC")
    return new_table([
        string_col("Symbol", ["AAPL"]),
        datetime_col("Timestamp", [t0]),
        double_col("BidPrice", [154.9]),
        double_col("AskPrice", [155.1]),
        double_col("BidSize", [10.0]),
        double_col("AskSize", [12.0]),
        # Last absent — comes from ticks_price, not ticks_bid_ask
    ])


def _raw_bar_table() -> Any:
    from deephaven import new_table
    from deephaven.column import datetime_col, double_col, string_col
    from deephaven.time import to_j_instant

    t0 = to_j_instant("2026-06-10T15:00:00 UTC")
    return new_table([
        string_col("Symbol", ["AAPL"]),
        datetime_col("Timestamp", [t0]),
        double_col("Open", [150.0]),
        double_col("High", [158.0]),
        double_col("Low", [149.0]),
        double_col("Close", [155.0]),
        double_col("Volume", [1000000.0]),
    ])


# ---------------------------------------------------------------------------
# position spec
# ---------------------------------------------------------------------------

def test_position_view_conforms_to_schema() -> None:
    spec = CANONICAL_SPECS["position"]
    viewed = apply_canonical_view(_raw_position_table(), spec, POSITION)
    port = ibda.connect({"position": viewed})
    arrow = port.table("position").snapshot()
    POSITION.validate(arrow)
    assert arrow.num_rows == 1


def test_position_view_renames_columns() -> None:
    spec = CANONICAL_SPECS["position"]
    viewed = apply_canonical_view(_raw_position_table(), spec, POSITION)
    port = ibda.connect({"position": viewed})
    arrow = port.table("position").snapshot()
    row = arrow.to_pylist()[0]
    assert row["Account"] == "DU1"
    assert row["ConId"] == 12345
    assert row["Sym"] == "AAPL"
    assert row["SecType"] == "STK"
    assert row["Qty"] == 100.0


def test_position_view_null_fills_market_columns() -> None:
    """MarketPrice, MarketValue, UnrealizedPnl must exist and be null."""
    spec = CANONICAL_SPECS["position"]
    viewed = apply_canonical_view(_raw_position_table(), spec, POSITION)
    port = ibda.connect({"position": viewed})
    arrow = port.table("position").snapshot()
    row = arrow.to_pylist()[0]
    assert row["MarketPrice"] is None
    assert row["MarketValue"] is None
    assert row["UnrealizedPnl"] is None


def test_position_view_option_identity_columns_null_for_stock() -> None:
    """Right/Strike/LastTradeDateOrContractMonth/Multiplier are null for a
    plain STK row; LocalSymbol is populated (present for every SecType)."""
    spec = CANONICAL_SPECS["position"]
    viewed = apply_canonical_view(_raw_position_table(), spec, POSITION)
    port = ibda.connect({"position": viewed})
    row = port.table("position").snapshot().to_pylist()[0]
    assert row["Right"] is None
    assert row["Strike"] is None
    assert row["LastTradeDateOrContractMonth"] is None
    assert row["Multiplier"] is None
    assert row["LocalSymbol"] == "AAPL"


def test_position_view_option_identity_columns_populated_for_option() -> None:
    """Right/Strike/LastTradeDateOrContractMonth/LocalSymbol/Multiplier survive
    for an OPT position — sourced directly from accounts_positions (NOT via a
    join to `definition`, which is sparse for a full book; see the POSITION
    schema module docstring)."""
    spec = CANONICAL_SPECS["position"]
    viewed = apply_canonical_view(_raw_position_table_option(), spec, POSITION)
    port = ibda.connect({"position": viewed})
    arrow = port.table("position").snapshot()
    POSITION.validate(arrow)
    row = arrow.to_pylist()[0]
    assert row["Right"] == "C"
    assert row["Strike"] == pytest.approx(200.0)
    assert row["LastTradeDateOrContractMonth"] == "20260717"
    assert row["LocalSymbol"] == "AAPL  260717C00200000"
    assert row["Multiplier"] == pytest.approx(100.0)


def test_position_view_currency_and_exchange_populated() -> None:
    """Currency/Exchange survive from accounts_positions — needed by
    subscribe.py's Contract reconstruction."""
    spec = CANONICAL_SPECS["position"]
    viewed = apply_canonical_view(_raw_position_table(), spec, POSITION)
    port = ibda.connect({"position": viewed})
    row = port.table("position").snapshot().to_pylist()[0]
    assert row["Currency"] == "USD"
    assert row["Exchange"] == "SMART"


# ---------------------------------------------------------------------------
# execution spec
# ---------------------------------------------------------------------------

def test_execution_view_conforms_to_schema() -> None:
    spec = CANONICAL_SPECS["execution"]
    viewed = apply_canonical_view(
        _raw_execution_table(), spec, EXECUTION, join_raw=_raw_commission_table()
    )
    port = ibda.connect({"execution": viewed})
    arrow = port.table("execution").snapshot()
    EXECUTION.validate(arrow)
    assert arrow.num_rows == 1


def test_execution_view_renames_shares_to_qty() -> None:
    spec = CANONICAL_SPECS["execution"]
    viewed = apply_canonical_view(
        _raw_execution_table(), spec, EXECUTION, join_raw=_raw_commission_table()
    )
    port = ibda.connect({"execution": viewed})
    arrow = port.table("execution").snapshot()
    row = arrow.to_pylist()[0]
    assert row["Qty"] == 100.0
    assert row["Venue"] == "NASDAQ"
    assert row["ExecId"] == "exec001"


def test_execution_view_liquidity_always_null_fills() -> None:
    """Liquidity has no source anywhere in deephaven-ib — always null, join or not."""
    spec = CANONICAL_SPECS["execution"]
    viewed = apply_canonical_view(
        _raw_execution_table(), spec, EXECUTION, join_raw=_raw_commission_table()
    )
    port = ibda.connect({"execution": viewed})
    row = port.table("execution").snapshot().to_pylist()[0]
    assert row["Liquidity"] is None


def test_execution_view_matched_exec_id_populates_commission_and_realized_pnl() -> None:
    """A fill whose ExecId has a commission report gets real Commission/RealizedPnl.

    Commission sign: commissionReport.commission arrives as a positive
    magnitude on the live path — confirm against a live snapshot — and the
    canonical view negates it to match the project's negative=cost convention
    (the same convention Flex's already-negative ibCommission already satisfies).
    """
    spec = CANONICAL_SPECS["execution"]
    viewed = apply_canonical_view(
        _raw_execution_table(exec_id="exec001"),
        spec,
        EXECUTION,
        join_raw=_raw_commission_table(exec_id="exec001", commission=1.25, realized_pnl=12.5),
    )
    port = ibda.connect({"execution": viewed})
    row = port.table("execution").snapshot().to_pylist()[0]
    assert row["Commission"] == pytest.approx(-1.25), (
        "raw commissionReport.commission (1.25, a positive magnitude) must be "
        "negated to -1.25 to match the negative=cost convention"
    )
    assert row["RealizedPnl"] == pytest.approx(12.5)


def test_execution_and_commission_views_negate_the_same_raw_value_once_each() -> None:
    """Double-flip guard: execution.Commission and commission.Commission are
    built by INDEPENDENT reads of the same raw orders_exec_commission_report
    row (execution via its natural_join, commission via its own raw_table —
    see specs.py) — each is negated exactly ONCE by apply_canonical_view's
    _COMMISSION_SIGN_FLIP_COLS registry. For a single raw-positive
    commissionReport input, both canonical columns must come out negative AND
    equal to each other for the same ExecId; neither may double-negate the
    other's already-negative output back to positive.
    """
    raw_commission = _raw_commission_table(exec_id="exec001", commission=1.25, realized_pnl=12.5)

    exec_spec = CANONICAL_SPECS["execution"]
    exec_viewed = apply_canonical_view(
        _raw_execution_table(exec_id="exec001"),
        exec_spec,
        EXECUTION,
        join_raw=raw_commission,
    )

    commission_spec = CANONICAL_SPECS["commission"]
    commission_viewed = apply_canonical_view(raw_commission, commission_spec, COMMISSION)

    port = ibda.connect({"execution": exec_viewed, "commission": commission_viewed})
    exec_row = port.table("execution").snapshot().to_pylist()[0]
    commission_row = port.table("commission").snapshot().to_pylist()[0]

    assert exec_row["Commission"] == pytest.approx(-1.25)
    assert commission_row["Commission"] == pytest.approx(-1.25)
    assert exec_row["Commission"] == pytest.approx(commission_row["Commission"])


def test_execution_view_unmatched_exec_id_yields_null_commission_not_zero() -> None:
    """A fill with NO matching commission report gets NULL Commission — never 0.0.

    This is the regression guard for the original bug: a declared-but-always-null
    Commission column is indistinguishable from "genuinely $0.00"; a consumer
    must be able to tell "not yet known" (None) apart from a real zero.
    """
    spec = CANONICAL_SPECS["execution"]
    viewed = apply_canonical_view(
        _raw_execution_table(exec_id="exec001"),
        spec,
        EXECUTION,
        # Commission report is for a DIFFERENT ExecId — exec001 has no match.
        join_raw=_raw_commission_table(exec_id="exec999", commission=1.25),
    )
    port = ibda.connect({"execution": viewed})
    arrow = port.table("execution").snapshot()
    # natural_join is left-outer: the execution row survives even unmatched.
    assert arrow.num_rows == 1
    row = arrow.to_pylist()[0]
    assert row["Commission"] is None
    assert row["RealizedPnl"] is None


def test_execution_view_commission_correction_resend_yields_latest() -> None:
    """A corrected commission report (same ExecId, resent) — the LATEST value wins.

    orders_exec_commission_report can re-emit a correction for the same ExecId;
    the join must dedupe (last_by) the right side first so natural_join doesn't
    error on >1 right-side match, and the surviving row must be the most recent.
    """
    from deephaven import new_table
    from deephaven.column import datetime_col, double_col, string_col
    from deephaven.time import to_j_instant

    t0 = to_j_instant("2026-06-10T15:00:05 UTC")
    t1 = to_j_instant("2026-06-10T15:05:00 UTC")
    commission_raw = new_table([
        string_col("ExecId", ["exec001", "exec001"]),
        datetime_col("ReceiveTime", [t0, t1]),
        double_col("Commission", [1.25, 1.10]),
        string_col("Currency", ["USD", "USD"]),
        double_col("RealizedPNL", [12.5, 13.0]),
        double_col("Yield", [0.0, 0.0]),
        string_col("YieldRedemptionDate", ["", ""]),
    ])
    spec = CANONICAL_SPECS["execution"]
    viewed = apply_canonical_view(
        _raw_execution_table(exec_id="exec001"), spec, EXECUTION, join_raw=commission_raw
    )
    port = ibda.connect({"execution": viewed})
    arrow = port.table("execution").snapshot()
    assert arrow.num_rows == 1  # the correction collapsed to one row, not two
    row = arrow.to_pylist()[0]
    # 1.10 (positive magnitude, the latest correction) negated to -1.10.
    assert row["Commission"] == pytest.approx(-1.10)
    assert row["RealizedPnl"] == pytest.approx(13.0)


def test_execution_view_sec_type_and_multiplier_populated() -> None:
    """SecType and Multiplier from raw table must survive the canonical view."""
    spec = CANONICAL_SPECS["execution"]
    viewed = apply_canonical_view(
        _raw_execution_table(), spec, EXECUTION, join_raw=_raw_commission_table()
    )
    port = ibda.connect({"execution": viewed})
    arrow = port.table("execution").snapshot()
    row = arrow.to_pylist()[0]
    assert row["SecType"] == "STK"
    # Multiplier string "1" must be cast to double 1.0.
    assert row["Multiplier"] == pytest.approx(1.0)


def test_execution_view_currency_passes_through() -> None:
    """Currency comes straight from orders_exec_details.Currency (no join needed)."""
    spec = CANONICAL_SPECS["execution"]
    viewed = apply_canonical_view(
        _raw_execution_table(), spec, EXECUTION, join_raw=_raw_commission_table()
    )
    port = ibda.connect({"execution": viewed})
    row = port.table("execution").snapshot().to_pylist()[0]
    assert row["Currency"] == "USD"


def test_execution_view_open_close_always_null_fills() -> None:
    """OpenClose has no source anywhere in deephaven-ib (Flex-only field) —
    always null on the live path, same as Liquidity."""
    spec = CANONICAL_SPECS["execution"]
    viewed = apply_canonical_view(
        _raw_execution_table(), spec, EXECUTION, join_raw=_raw_commission_table()
    )
    port = ibda.connect({"execution": viewed})
    row = port.table("execution").snapshot().to_pylist()[0]
    assert row["OpenClose"] is None


# ---------------------------------------------------------------------------
# Side vocabulary normalization — IB execDetails.side is BOT/SLD; canonical
# vocabulary (matching the Flex mapper) is BUY/SELL.
# ---------------------------------------------------------------------------


def test_execution_view_side_normalizes_bot_to_buy() -> None:
    """IB execDetails.side 'BOT' -> canonical 'BUY' — confirm against a live snapshot."""
    spec = CANONICAL_SPECS["execution"]
    viewed = apply_canonical_view(
        _raw_execution_table(side="BOT"), spec, EXECUTION, join_raw=_raw_commission_table()
    )
    port = ibda.connect({"execution": viewed})
    row = port.table("execution").snapshot().to_pylist()[0]
    assert row["Side"] == "BUY"


def test_execution_view_side_normalizes_sld_to_sell() -> None:
    """IB execDetails.side 'SLD' -> canonical 'SELL' — confirm against a live snapshot."""
    spec = CANONICAL_SPECS["execution"]
    viewed = apply_canonical_view(
        _raw_execution_table(exec_id="exec002", side="SLD"),
        spec,
        EXECUTION,
        join_raw=_raw_commission_table(exec_id="exec002"),
    )
    port = ibda.connect({"execution": viewed})
    row = port.table("execution").snapshot().to_pylist()[0]
    assert row["Side"] == "SELL"


def test_execution_view_side_passes_through_when_already_canonical() -> None:
    """Side values already in canonical vocabulary (BUY/SELL) pass through unchanged."""
    spec = CANONICAL_SPECS["execution"]
    viewed = apply_canonical_view(
        _raw_execution_table(side="BUY"), spec, EXECUTION, join_raw=_raw_commission_table()
    )
    port = ibda.connect({"execution": viewed})
    row = port.table("execution").snapshot().to_pylist()[0]
    assert row["Side"] == "BUY"


# ---------------------------------------------------------------------------
# order spec — orders_submitted → canonical order (Action→Side, etc.)
# ---------------------------------------------------------------------------

def _raw_order_table() -> Any:
    from deephaven import new_table
    from deephaven.column import datetime_col, double_col, long_col, string_col
    from deephaven.time import to_j_instant

    t0 = to_j_instant("2026-06-12T14:30:00 UTC")
    return new_table([
        long_col("OrderId", [101]),
        long_col("PermId", [987654321]),
        string_col("Account", ["DU1"]),
        string_col("Symbol", ["AAPL"]),
        string_col("Action", ["BUY"]),
        double_col("TotalQuantity", [100.0]),
        double_col("FilledQuantity", [40.0]),
        double_col("LmtPrice", [150.25]),
        string_col("Status", ["Submitted"]),
        datetime_col("ReceiveTime", [t0]),
    ])


def test_order_view_conforms_to_schema() -> None:
    spec = CANONICAL_SPECS["order"]
    viewed = apply_canonical_view(_raw_order_table(), spec, ORDER)
    port = ibda.connect({"order": viewed})
    arrow = port.table("order").snapshot()
    ORDER.validate(arrow)
    assert arrow.num_rows == 1


def test_order_view_renames_action_to_side() -> None:
    spec = CANONICAL_SPECS["order"]
    viewed = apply_canonical_view(_raw_order_table(), spec, ORDER)
    port = ibda.connect({"order": viewed})
    row = port.table("order").snapshot().to_pylist()[0]
    assert row["Side"] == "BUY"
    assert row["Qty"] == 100.0
    assert row["FilledQty"] == 40.0
    assert row["LimitPrice"] == pytest.approx(150.25)


def test_order_view_status_and_ids_preserved() -> None:
    spec = CANONICAL_SPECS["order"]
    viewed = apply_canonical_view(_raw_order_table(), spec, ORDER)
    port = ibda.connect({"order": viewed})
    row = port.table("order").snapshot().to_pylist()[0]
    assert row["OrderId"] == 101
    assert row["PermId"] == 987654321
    assert row["Status"] == "Submitted"


# ---------------------------------------------------------------------------
# trade spec — ticks_trade → canonical trade
# ---------------------------------------------------------------------------

def _raw_trade_table() -> Any:
    from deephaven import new_table
    from deephaven.column import datetime_col, double_col, string_col
    from deephaven.time import to_j_instant

    t0 = to_j_instant("2026-06-12T15:30:00 UTC")
    return new_table([
        string_col("Symbol", ["AAPL"]),
        datetime_col("Timestamp", [t0]),
        double_col("Price", [291.55]),
        double_col("Size", [100.0]),
    ])


def test_trade_view_conforms_to_schema() -> None:
    spec = CANONICAL_SPECS["trade"]
    viewed = apply_canonical_view(_raw_trade_table(), spec, TRADE)
    port = ibda.connect({"trade": viewed})
    arrow = port.table("trade").snapshot()
    TRADE.validate(arrow)
    assert arrow.num_rows == 1


def test_trade_view_renames_and_values() -> None:
    spec = CANONICAL_SPECS["trade"]
    viewed = apply_canonical_view(_raw_trade_table(), spec, TRADE)
    port = ibda.connect({"trade": viewed})
    row = port.table("trade").snapshot().to_pylist()[0]
    assert row["Sym"] == "AAPL"
    assert row["Price"] == pytest.approx(291.55)
    assert row["Size"] == 100.0


# ---------------------------------------------------------------------------
# definition spec — including Multiplier string→double cast
# ---------------------------------------------------------------------------

def test_definition_view_conforms_to_schema() -> None:
    spec = CANONICAL_SPECS["definition"]
    viewed = apply_canonical_view(_raw_definition_table(), spec, DEFINITION)
    port = ibda.connect({"definition": viewed})
    arrow = port.table("definition").snapshot()
    DEFINITION.validate(arrow)
    assert arrow.num_rows == 1


def test_definition_view_multiplier_cast_to_float() -> None:
    """Multiplier string '1' must arrive as float 1.0."""
    spec = CANONICAL_SPECS["definition"]
    viewed = apply_canonical_view(_raw_definition_table(), spec, DEFINITION)
    port = ibda.connect({"definition": viewed})
    arrow = port.table("definition").snapshot()
    row = arrow.to_pylist()[0]
    assert row["Multiplier"] == pytest.approx(1.0)
    assert row["Exchange"] == "NASDAQ"


def test_definition_view_empty_multiplier_is_null() -> None:
    """Empty-string Multiplier must produce NULL_DOUBLE (not an error)."""
    spec = CANONICAL_SPECS["definition"]
    viewed = apply_canonical_view(_raw_definition_table_null_multiplier(), spec, DEFINITION)
    port = ibda.connect({"definition": viewed})
    arrow = port.table("definition").snapshot()
    DEFINITION.validate(arrow)
    row = arrow.to_pylist()[0]
    assert row["Multiplier"] is None


def test_definition_view_double_multiplier_conforms_to_schema() -> None:
    """Live contracts_details has Multiplier as double — must not raise.

    Regression guard: before the type-aware fix, apply_canonical_view would
    emit a Groovy expression using .isEmpty() and Double.parseDouble() on a
    double primitive, causing FormulaCompilationException at snapshot time.
    """
    spec = CANONICAL_SPECS["definition"]
    viewed = apply_canonical_view(_raw_definition_table_double_multiplier(), spec, DEFINITION)
    port = ibda.connect({"definition": viewed})
    arrow = port.table("definition").snapshot()
    DEFINITION.validate(arrow)
    assert arrow.num_rows == 1


def test_definition_view_double_multiplier_value_preserved() -> None:
    """When Multiplier is already a double column its value must survive."""
    spec = CANONICAL_SPECS["definition"]
    viewed = apply_canonical_view(_raw_definition_table_double_multiplier(), spec, DEFINITION)
    port = ibda.connect({"definition": viewed})
    arrow = port.table("definition").snapshot()
    row = arrow.to_pylist()[0]
    assert row["Multiplier"] == pytest.approx(1.0)
    assert row["Sym"] == "SPY"
    assert row["Exchange"] == "ARCA"


def test_definition_view_option_identity_columns_null_for_stock() -> None:
    """Right/Strike/LastTradeDateOrContractMonth are null for a plain STK row;
    LocalSymbol is populated (IBKR reports it for every SecType)."""
    spec = CANONICAL_SPECS["definition"]
    viewed = apply_canonical_view(_raw_definition_table(), spec, DEFINITION)
    port = ibda.connect({"definition": viewed})
    row = port.table("definition").snapshot().to_pylist()[0]
    assert row["Right"] is None
    assert row["Strike"] is None
    assert row["LastTradeDateOrContractMonth"] is None
    assert row["LocalSymbol"] == "AAPL"


def test_definition_view_option_identity_columns_populated_for_option() -> None:
    """Right/Strike/LastTradeDateOrContractMonth/LocalSymbol survive for an
    OPT contract — the columns a downstream positions view needs to name a
    specific option contract."""
    spec = CANONICAL_SPECS["definition"]
    viewed = apply_canonical_view(_raw_definition_table_option(), spec, DEFINITION)
    port = ibda.connect({"definition": viewed})
    arrow = port.table("definition").snapshot()
    DEFINITION.validate(arrow)
    row = arrow.to_pylist()[0]
    assert row["Right"] == "C"
    assert row["Strike"] == pytest.approx(200.0)
    assert row["LastTradeDateOrContractMonth"] == "20260717"
    assert row["LocalSymbol"] == "AAPL  260717C00200000"


# ---------------------------------------------------------------------------
# quote spec
# ---------------------------------------------------------------------------

def test_quote_view_conforms_to_schema() -> None:
    spec = CANONICAL_SPECS["quote"]
    viewed = apply_canonical_view(_raw_quote_table(), spec, QUOTE)
    port = ibda.connect({"quote": viewed})
    arrow = port.table("quote").snapshot()
    QUOTE.validate(arrow)
    assert arrow.num_rows == 1


def test_quote_view_renames_bid_ask() -> None:
    spec = CANONICAL_SPECS["quote"]
    viewed = apply_canonical_view(_raw_quote_table(), spec, QUOTE)
    port = ibda.connect({"quote": viewed})
    arrow = port.table("quote").snapshot()
    row = arrow.to_pylist()[0]
    assert row["Sym"] == "AAPL"
    assert row["Bid"] == pytest.approx(154.9)
    assert row["Ask"] == pytest.approx(155.1)
    assert row["Last"] is None


# ---------------------------------------------------------------------------
# bar spec
# ---------------------------------------------------------------------------

def test_bar_view_conforms_to_schema() -> None:
    spec = CANONICAL_SPECS["bar"]
    viewed = apply_canonical_view(_raw_bar_table(), spec, BAR)
    port = ibda.connect({"bar": viewed})
    arrow = port.table("bar").snapshot()
    BAR.validate(arrow)
    assert arrow.num_rows == 1


def test_bar_view_all_columns_present() -> None:
    spec = CANONICAL_SPECS["bar"]
    viewed = apply_canonical_view(_raw_bar_table(), spec, BAR)
    port = ibda.connect({"bar": viewed})
    arrow = port.table("bar").snapshot()
    row = arrow.to_pylist()[0]
    assert row["Sym"] == "AAPL"
    assert row["Open"] == pytest.approx(150.0)
    assert row["Close"] == pytest.approx(155.0)
    assert row["Volume"] == pytest.approx(1000000.0)


# ---------------------------------------------------------------------------
# dedupe_keys: position deduplication (accounts_positions double-subscription)
# ---------------------------------------------------------------------------

def _raw_position_table_doubled() -> Any:
    """Two identical rows per (Account, ConId) — mirrors the double-subscription bug.

    accounts_positions is append-only; deephaven-ib issues reqPositionsMulti("All")
    and then reqPositionsMulti(account) from the managedAccounts callback, writing
    each position twice at startup.  The canonical position view must reduce to one
    row per (Account, ConId) via last_by.
    """
    from deephaven import new_table
    from deephaven.column import double_col, long_col, string_col

    return new_table([
        # DU1 / AAPL — appears twice (identical rows from duplicate subscription)
        string_col("Account",    ["DU1",     "DU1"]),
        long_col("ContractId",   [12345,     12345]),
        string_col("Symbol",     ["AAPL",    "AAPL"]),
        string_col("SecType",    ["STK",     "STK"]),
        double_col("Position",   [100.0,     100.0]),
        double_col("AvgCost",    [150.0,     150.0]),
        string_col("Right",      [None, None]),
        double_col("Strike",     [None, None]),
        string_col("LastTradeDateOrContractMonth", [None, None]),
        string_col("LocalSymbol", ["AAPL", "AAPL"]),
        double_col("Multiplier", [None, None]),
        string_col("Currency", ["USD", "USD"]),
        string_col("Exchange", ["SMART", "SMART"]),
    ])


def _raw_position_table_two_positions() -> Any:
    """Two distinct positions — must each survive the last_by dedupe."""
    from deephaven import new_table
    from deephaven.column import double_col, long_col, string_col

    return new_table([
        string_col("Account",    ["DU1",    "DU1"]),
        long_col("ContractId",   [12345,    67890]),
        string_col("Symbol",     ["AAPL",   "MSFT"]),
        string_col("SecType",    ["STK",    "STK"]),
        double_col("Position",   [100.0,    200.0]),
        double_col("AvgCost",    [150.0,    250.0]),
        string_col("Right",      [None, None]),
        double_col("Strike",     [None, None]),
        string_col("LastTradeDateOrContractMonth", [None, None]),
        string_col("LocalSymbol", ["AAPL", "MSFT"]),
        double_col("Multiplier", [None, None]),
        string_col("Currency", ["USD", "USD"]),
        string_col("Exchange", ["SMART", "SMART"]),
    ])


def test_position_dedupe_collapses_duplicate_rows_to_one() -> None:
    """Duplicate (Account, ConId) rows from double-subscription collapse to one row.

    The canonical position spec has dedupe_keys=["Account", "ConId"].  When
    accounts_positions contains two identical rows for the same position (the live bug),
    apply_canonical_view must return exactly one row per position.

    Exercises apply_canonical_view directly (not via port) to avoid the schema-name
    constraint in DeephavenPort.table(); the deduplication happens inside the view fn.
    """
    from deephaven.pandas import to_pandas

    spec = CANONICAL_SPECS["position"]
    viewed = apply_canonical_view(_raw_position_table_doubled(), spec, POSITION)
    df = to_pandas(viewed)
    assert len(df) == 1, (
        f"Expected 1 row after deduplication of 2 identical (Account, ConId) rows; "
        f"got {len(df)}"
    )
    row = df.iloc[0]
    assert row["Account"] == "DU1"
    assert row["ConId"] == 12345
    assert row["Qty"] == pytest.approx(100.0)


def test_position_dedupe_keeps_both_distinct_positions() -> None:
    """Two positions with different ConIds must both survive last_by dedupe."""
    from deephaven.pandas import to_pandas

    spec = CANONICAL_SPECS["position"]
    viewed = apply_canonical_view(_raw_position_table_two_positions(), spec, POSITION)
    df = to_pandas(viewed)
    assert len(df) == 2, (
        f"Expected 2 rows (two distinct positions); got {len(df)}"
    )
    df_sorted = df.sort_values("ConId").reset_index(drop=True)
    assert df_sorted.iloc[0]["ConId"] == 12345
    assert df_sorted.iloc[0]["Sym"] == "AAPL"
    assert df_sorted.iloc[1]["ConId"] == 67890
    assert df_sorted.iloc[1]["Sym"] == "MSFT"


def test_position_dedupe_schema_validates_after_dedupe() -> None:
    """Schema validation passes on deduped position view."""

    spec = CANONICAL_SPECS["position"]
    viewed = apply_canonical_view(_raw_position_table_doubled(), spec, POSITION)
    # Use the port path (canonical schema name "position") for schema validation.
    port = ibda.connect({"position": viewed})
    arrow = port.table("position").snapshot()
    POSITION.validate(arrow)
    assert arrow.num_rows == 1


def test_execution_no_dedupe_retains_all_rows() -> None:
    """execution spec has no dedupe_keys — event-log rows must all be kept.

    This guards against accidentally enabling deduplication on the execution table,
    which would hide fills that share the same (Account, ConId) pair.
    """
    spec = CANONICAL_SPECS["execution"]
    assert spec.dedupe_keys is None, (
        "execution spec must NOT have dedupe_keys — it is an event log"
    )
    # Build a table with two fills for the same ConId (normal for a split fill)
    from deephaven import new_table
    from deephaven.column import datetime_col, double_col, long_col, string_col
    from deephaven.pandas import to_pandas
    from deephaven.time import to_j_instant

    t0 = to_j_instant("2026-06-10T15:00:00 UTC")
    t1 = to_j_instant("2026-06-10T15:01:00 UTC")
    raw = new_table([
        string_col("ExecId",           ["exec001",      "exec002"]),
        datetime_col("Timestamp",      [t0,             t1]),
        string_col("Account",          ["DU1",          "DU1"]),
        long_col("ContractId",         [12345,          12345]),
        long_col("OrderId",            [987,            987]),
        string_col("OrderRef",         ["STRAT-A-AAPL", "STRAT-A-AAPL"]),
        string_col("Symbol",           ["AAPL",         "AAPL"]),
        string_col("SecType",          ["STK",          "STK"]),
        string_col("Side",             ["BUY",          "BUY"]),
        double_col("Shares",           [50.0,           50.0]),
        double_col("Price",            [155.0,          155.5]),
        string_col("Multiplier",       ["1",            "1"]),
        string_col("ExecutionExchange", ["NASDAQ",      "NASDAQ"]),
        string_col("Currency",         ["USD",          "USD"]),
    ])
    # execution now declares a join_table (orders_exec_commission_report) — pass
    # a commission table (neither row matches either ExecId here; irrelevant to
    # this test's row-count assertion, but required so the join step has a
    # right-hand table to natural_join against).
    viewed = apply_canonical_view(
        raw, spec, EXECUTION, join_raw=_raw_commission_table(exec_id="unrelated")
    )
    df = to_pandas(viewed)
    assert len(df) == 2, (
        f"execution view must retain all fills; got {len(df)} (expected 2)"
    )
