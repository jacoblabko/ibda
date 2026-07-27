"""Contract tests for DeephavenPort — JVM-backed.

These tests boot an in-process Deephaven Server (see conftest.py) and verify
that every DataPort operation produces Arrow tables conforming to the canonical
schemas. The contract (assertions) is fixed; run with:

    uv run pytest ibda/tests_jvm
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

from ibda.schema import POSITION


def _build_tables() -> dict[str, Any]:
    from deephaven import new_table
    from deephaven.column import datetime_col, double_col, long_col, string_col
    from deephaven.time import to_j_instant

    positions = new_table([
        string_col("Account", ["DU1"]),
        long_col("ConId", [1]),
        string_col("Sym", ["AAPL"]),
        string_col("SecType", ["STK"]),
        double_col("Qty", [100.0]),
        double_col("AvgCost", [150.0]),
        string_col("Right", [None]),
        double_col("Strike", [None]),
        string_col("LastTradeDateOrContractMonth", [None]),
        string_col("LocalSymbol", ["AAPL"]),
        double_col("Multiplier", [None]),
        string_col("Currency", ["USD"]),
        string_col("Exchange", ["SMART"]),
        double_col("MarketPrice", [155.0]),
        double_col("MarketValue", [15500.0]),
        double_col("UnrealizedPnl", [500.0]),
    ])
    t0 = to_j_instant("2026-06-10T15:00:00 UTC")
    t1 = to_j_instant("2026-06-10T15:00:01 UTC")
    fills = new_table([
        string_col("ExecId", ["e1"]),
        datetime_col("Timestamp", [t1]),
        string_col("Account", ["DU1"]),
        long_col("ConId", [1]),
        string_col("Sym", ["AAPL"]),
        string_col("Side", ["BUY"]),
        double_col("Qty", [100.0]),
        double_col("Price", [155.0]),
        string_col("Venue", ["NASDAQ"]),
        string_col("Liquidity", ["taker"]),
        double_col("Commission", [1.0]),
    ])
    quotes = new_table([
        string_col("Sym", ["AAPL"]),
        datetime_col("Timestamp", [t0]),
        double_col("Bid", [154.9]),
        double_col("Ask", [155.1]),
        double_col("BidSize", [10.0]),
        double_col("AskSize", [12.0]),
        double_col("Last", [155.0]),
    ])
    return {"position": positions, "execution": fills, "quote": quotes}


def _adapter() -> Any:
    from ibda.adapters.deephaven.adapter import DeephavenPort
    return DeephavenPort(_build_tables())


def test_table_snapshot_conforms_to_schema() -> None:
    port = _adapter()
    tbl = port.table("position").snapshot()
    assert isinstance(tbl, pa.Table)
    POSITION.validate(tbl)
    assert tbl.num_rows == 1


def test_unknown_table_raises() -> None:
    from ibda.errors import UnknownTable
    port = _adapter()
    with pytest.raises(UnknownTable):
        port.table("does_not_exist")


def test_filter() -> None:
    port = _adapter()
    empty = port.filter(port.table("position"), "Qty < 0").snapshot()
    assert empty.num_rows == 0


def test_filter_type_invalid_predicate_raises_valueerror_not_raw_engine_error() -> None:
    """Fix #3: `validate_expression` is not even in `filter`'s path (it hands the
    predicate to `.where()` as-is), so a grammar-valid-Deephaven-syntax but
    type-invalid predicate (comparing a String column to a double column) must
    surface as a port-level ValueError, not a raw `deephaven.DHError`."""
    from deephaven.dherror import DHError  # noqa: PLC0415 — JVM-gated

    port = _adapter()
    position = port.table("position")

    with pytest.raises(ValueError) as exc_info:
        port.filter(position, "Sym > Qty")  # String > double: type-invalid

    assert not isinstance(exc_info.value, DHError), (
        "a raw deephaven.DHError leaked through filter() instead of ValueError"
    )
    assert "Sym > Qty" in str(exc_info.value)


def test_time_window_non_timestamp_column_raises_valueerror_not_raw_engine_error() -> None:
    """Fix #3: time_window's generated predicate compares `column` against
    parseInstant(...) — naming a non-timestamp column (e.g. the double `Qty`)
    is grammar-valid but type-invalid, and must surface as ValueError."""
    from deephaven.dherror import DHError  # noqa: PLC0415 — JVM-gated

    port = _adapter()
    position = port.table("position")

    with pytest.raises(ValueError) as exc_info:
        port.time_window(
            position,
            column="Qty",  # double, not a timestamp column
            start="2026-01-01T00:00:00Z",
            end="2026-12-31T00:00:00Z",
        )

    assert not isinstance(exc_info.value, DHError), (
        "a raw deephaven.DHError leaked through time_window() instead of ValueError"
    )
    assert "Qty" in str(exc_info.value)


def test_as_of_join_fills_to_quotes() -> None:
    port = _adapter()
    fills = port.table("execution")
    quotes = port.table("quote")
    joined = port.as_of_join(fills, quotes, on=["Sym", "Timestamp"], joins=["Bid", "Ask"])
    out = joined.snapshot()
    assert out.num_rows == 1
    row = out.to_pylist()[0]
    assert row["Bid"] == 154.9 and row["Ask"] == 155.1


def test_group_by_count_uses_string_not_list() -> None:
    """Regression (Fix 1): agg.count_ takes a plain string, not a list.

    Before the fix, agg.count_(["ConId"]) raised a runtime error.  After the
    fix, agg.count_("ConId") produces a per-group row with the count column
    named "ConId".
    """
    port = _adapter()
    result = port.group_by(
        port.table("position"),
        by=["Sym"],
        aggs={"ConId": "count"},
    ).snapshot()
    assert result.num_rows >= 1, "group_by count should produce at least one row"
    assert "ConId" in result.schema.names, "count output column 'ConId' should be present"
    # The count of a single-row table grouped by Sym should be 1.
    row = result.to_pylist()[0]
    assert row["ConId"] == 1


def test_group_by_type_invalid_agg_raises_valueerror_not_raw_engine_error() -> None:
    """FIX 2 (query.py validate() gap + adapter defense-in-depth): summing a
    non-numeric (String) column is grammar-valid-looking but type-invalid --
    query.validate() now catches the common "unknown column" case, but a
    real-but-wrong-dtype column (e.g. "Sym", a String, summed) still reaches
    `agg_by` and must surface as a port-level ValueError, not a raw
    `deephaven.DHError` -- mirroring `filter`/`time_window`/`derive`'s existing
    mitigation, which `group_by` previously lacked."""
    from deephaven.dherror import DHError  # noqa: PLC0415 — JVM-gated

    port = _adapter()
    position = port.table("position")

    with pytest.raises(ValueError) as exc_info:
        port.group_by(position, by=["ConId"], aggs={"Sym": "sum"})  # String column, sum

    assert not isinstance(exc_info.value, DHError), (
        "a raw deephaven.DHError leaked through group_by() instead of ValueError"
    )
    assert "Sym" in str(exc_info.value)


def test_sort_ascending() -> None:
    """sort() with descending=False produces ascending order."""
    port = _adapter()
    # Build a two-row table via filter (both rows present) then sort Sym ascending.
    result = port.sort(
        port.table("position"),
        keys=[("Sym", False)],
    ).snapshot()
    assert result.num_rows == 1  # fixture has only one position row (AAPL)


def test_sort_descending() -> None:
    """sort() with descending=True must produce a non-empty result."""
    port = _adapter()
    result = port.sort(
        port.table("position"),
        keys=[("Qty", True)],
    ).snapshot()
    assert result.num_rows >= 1


# ---------------------------------------------------------------------------
# Result.schema must reflect the POST-operation column set
#
# Before the fix, group_by and as_of_join returned the INPUT schema unchanged,
# so consumers validating a grouped/joined snapshot got a spurious SchemaMismatch
# because the schema listed columns that had been removed (group_by) or missing
# ones that had been added (as_of_join).  After the fix, _schema_from_dh_handle
# derives the schema from meta_table so it always matches the actual output.
# ---------------------------------------------------------------------------


def test_group_by_schema_reflects_output_columns_not_input() -> None:
    """result.schema.column_names after group_by == actual grouped columns.

    Before the fix, result.schema was the full POSITION input schema (all 9 columns).
    After the fix, result.schema is derived from meta_table: only the by + agg columns
    (SecType, Qty) are present — not ConId or any other input-only column.
    """
    port = _adapter()
    result = port.group_by(
        port.table("position"),
        by=["SecType"],
        aggs={"Qty": "sum"},
    )
    snapshot = result.snapshot()

    # Core assertion: schema.column_names must exactly match the actual snapshot columns
    assert set(result.schema.column_names) == set(snapshot.schema.names), (
        f"Result.schema column names {result.schema.column_names!r} do not match "
        f"actual grouped output columns {tuple(snapshot.schema.names)!r}. "
        "The stale-schema bug was not fixed."
    )
    # Output must contain only the by + agg columns
    assert "SecType" in result.schema.column_names
    assert "Qty" in result.schema.column_names
    # Input-only columns must NOT appear in the grouped schema
    assert "ConId" not in result.schema.column_names, (
        "ConId (input-only column) must not appear in the derived group_by schema — "
        "group_by strips non-by/non-agg columns from the output"
    )
    assert "Account" not in result.schema.column_names
    assert "Sym" not in result.schema.column_names


def test_as_of_join_schema_includes_joined_columns() -> None:
    """result.schema.column_names after as_of_join includes right-side columns.

    Before the fix, result.schema was the LEFT schema (execution columns only); Bid
    and Ask (joined from the right) were absent, so any consumer calling
    result.schema.column_names to plan field access found them missing.
    After the fix, the schema is derived from meta_table and includes both.
    """
    port = _adapter()
    fills = port.table("execution")
    quotes = port.table("quote")
    joined = port.as_of_join(fills, quotes, on=["Sym", "Timestamp"], joins=["Bid", "Ask"])
    snapshot = joined.snapshot()

    # Core assertion: schema.column_names must match the actual snapshot columns
    assert set(joined.schema.column_names) == set(snapshot.schema.names), (
        f"Result.schema column names {joined.schema.column_names!r} do not match "
        f"actual joined output columns {tuple(snapshot.schema.names)!r}. "
        "The stale-schema bug was not fixed."
    )
    # The joined right-side columns must be present in the schema
    assert "Bid" in joined.schema.column_names, (
        "Bid (joined from right) not in result.schema after as_of_join"
    )
    assert "Ask" in joined.schema.column_names, (
        "Ask (joined from right) not in result.schema after as_of_join"
    )
    # Original left-side columns must still be present
    assert "ExecId" in joined.schema.column_names
    assert "Sym" in joined.schema.column_names


# ---------------------------------------------------------------------------
# NULL_LONG round-trip: nullable long column must be int64 in Arrow
#
# Deephaven NULL_LONG (Java Long.MIN_VALUE) becomes NaN in pandas, forcing the
# column to float64.  _normalize_arrow must detect float64 columns declared INT64
# in the schema and cast them back to nullable int64.
# ---------------------------------------------------------------------------


def _build_position_with_null_conid() -> dict[str, Any]:
    """Build a position-schema table where one row has a null ConId (NULL_LONG).

    Uses empty_table().update() with the DH formula constant NULL_LONG so the
    resulting table has a proper Java Long null in row 1.  When read via
    to_pandas() this becomes NaN (float64), exercising the _normalize_arrow cast.
    """
    from deephaven import empty_table

    # NULL_LONG is the Deephaven built-in formula constant (= Long.MIN_VALUE).
    positions = empty_table(2).update([
        "Account = `DU1`",
        "ConId = i == 0 ? (long) 12345 : NULL_LONG",
        "Sym = i == 0 ? `AAPL` : `MSFT`",
        "SecType = `STK`",
        "Qty = (double) 100.0",
        "AvgCost = (double) 150.0",
        "MarketPrice = (double) 155.0",
        "MarketValue = (double) 15500.0",
        "UnrealizedPnl = (double) 500.0",
    ])
    return {"position": positions}


def test_nullable_long_snapshots_as_int64_not_float64() -> None:
    """NULL_LONG in a Deephaven long column must round-trip as nullable int64.

    Without _normalize_arrow's INT64 cast, a Deephaven long column containing
    NULL_LONG would arrive as float64 (NULL_LONG → pandas NaN → float64 Arrow column),
    and any consumer that canonically expects int64/int|None (e.g. ConId) would see
    float64, causing schema.validate() to raise SchemaMismatch.

    After the fix, _normalize_arrow detects float64 columns declared INT64 in the
    schema and casts them back to nullable int64, preserving null sentinels.
    """
    import pyarrow as pa

    from ibda.adapters.deephaven.adapter import DeephavenPort

    port = DeephavenPort(_build_position_with_null_conid())
    arrow = port.table("position").snapshot()

    conid_field = arrow.schema.field("ConId")
    assert conid_field.type == pa.int64(), (
        f"ConId column type should be int64 (nullable), got {conid_field.type!r}. "
        "NULL_LONG round-trip is broken: float64 was not cast back to int64 "
        "by _normalize_arrow (a prior regression)."
    )

    conid_vals = arrow.column("ConId").to_pylist()
    assert conid_vals[0] == 12345, (
        f"Non-null ConId row should be 12345, got {conid_vals[0]!r}"
    )
    assert conid_vals[1] is None, (
        f"NULL_LONG row should be None in Arrow int64, got {conid_vals[1]!r}"
    )
