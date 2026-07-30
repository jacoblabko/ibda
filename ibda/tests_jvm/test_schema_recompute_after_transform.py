"""JVM regression test: schema recompute after group_by / as_of_join.

DeephavenPort.group_by() and as_of_join() once returned the *input* schema
unchanged.  The actual output table has a different column set, so result.schema
was stale: schema.validate(result.snapshot()) raised SchemaMismatch.  Both now
derive the output schema from the handle's meta_table.

This file is the regression guard.  Each test asserts:

  1. result.schema.column_names exactly matches the ACTUAL Arrow output columns
     (i.e. the schema was re-derived from the real table, not copied from the input).
  2. result.snapshot() validates against result.schema with no SchemaMismatch
     (the definitive correctness check — this is what would have blown up in
     production before the fix).

If the fix is reverted, assertion (1) will catch the mismatch in column names and
assertion (2) will catch it via schema.validate().

Run with:
    uv run pytest ibda/tests_jvm/test_schema_recompute_after_transform.py -v
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa


# ---------------------------------------------------------------------------
# Table builders — all deephaven imports deferred inside function bodies
# so they only execute when the JVM is live.
# ---------------------------------------------------------------------------

def _build_tables() -> dict[str, Any]:
    """Build fixture tables matching the shapes used in test_adapter_contract.py."""
    from deephaven import new_table  # noqa: PLC0415 — JVM-gated
    from deephaven.column import datetime_col, double_col, long_col, string_col  # noqa: PLC0415
    from deephaven.time import to_j_instant  # noqa: PLC0415

    positions = new_table([
        string_col("Account", ["DU1"]),
        long_col("ConId", [1]),
        string_col("Sym", ["AAPL"]),
        string_col("SecType", ["STK"]),
        double_col("Qty", [100.0]),
        double_col("AvgCost", [150.0]),
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
    from ibda.adapters.deephaven.adapter import DeephavenPort  # noqa: PLC0415
    return DeephavenPort(_build_tables())


# ---------------------------------------------------------------------------
# Regression tests
# ---------------------------------------------------------------------------

def test_group_by_schema_matches_output_columns() -> None:
    """Regression: group_by result.schema reflects the OUTPUT columns, not the input.

    The position table has 9 columns.  After group_by(by=["Sym"], aggs={"ConId":
    "count"}), the output has exactly 2 columns (Sym, ConId).

    Before the fix, group_by returned the full 9-column input schema.
    schema.validate() would then raise SchemaMismatch because it expected columns
    that Deephaven's agg_by dropped.

    Assertions:
    - result.schema.column_names == actual Arrow output column names (recomputed,
      not copied from the 9-column input).
    - "Account" is NOT in result.schema.column_names (it is an input-only column
      dropped by the aggregation — its presence indicates the stale schema bug).
    - result.schema.validate(snapshot) passes without SchemaMismatch.
    """
    port = _adapter()
    result = port.group_by(
        port.table("position"),
        by=["Sym"],
        aggs={"ConId": "count"},
    )
    snapshot: pa.Table = result.snapshot()

    actual_columns: tuple[str, ...] = tuple(snapshot.schema.names)
    assert result.schema.column_names == actual_columns, (
        f"group_by schema is stale: result.schema.column_names="
        f"{result.schema.column_names!r} but snapshot has columns "
        f"{actual_columns!r}. DeephavenPort.group_by() must call "
        "_schema_from_dh_handle() to recompute the schema from the actual output."
    )

    assert "Account" not in result.schema.column_names, (
        "'Account' is an input-only column dropped by the aggregation. "
        "Its presence in result.schema.column_names means the stale 9-column "
        "input schema was returned unchanged — the #5 regression is back."
    )

    # Definitive check: schema.validate() against the snapshot must not raise.
    result.schema.validate(snapshot)


def test_as_of_join_schema_includes_joined_columns() -> None:
    """Regression: as_of_join result.schema includes right-table join columns.

    The fills table has 11 columns.  After as_of_join(..., joins=["Bid", "Ask"]),
    the output gains Bid and Ask from the right (quotes) table.

    Before the fix, as_of_join returned the LEFT (fills) schema unchanged.
    result.schema.column_names would be missing Bid and Ask, meaning any downstream
    code inspecting the schema would have an incomplete view of the table.

    Assertions:
    - result.schema.column_names == actual Arrow output column names (Bid and Ask
      are present because the schema was re-derived from the joined output).
    - "Bid" and "Ask" are in result.schema.column_names (the primary sentinel: if
      the left schema was returned unchanged, these would be absent).
    - result.schema.validate(snapshot) passes without SchemaMismatch.
    """
    port = _adapter()
    fills = port.table("execution")
    quotes = port.table("quote")
    result = port.as_of_join(fills, quotes, on=["Sym", "Timestamp"], joins=["Bid", "Ask"])
    snapshot: pa.Table = result.snapshot()

    actual_columns: tuple[str, ...] = tuple(snapshot.schema.names)
    assert result.schema.column_names == actual_columns, (
        f"as_of_join schema is stale: result.schema.column_names="
        f"{result.schema.column_names!r} but snapshot has columns "
        f"{actual_columns!r}. DeephavenPort.as_of_join() must call "
        "_schema_from_dh_handle() to recompute the schema from the actual output."
    )

    assert "Bid" in result.schema.column_names, (
        "'Bid' must be in result.schema.column_names after as_of_join with "
        "joins=['Bid', 'Ask']. If missing, the left-table (fills) schema was "
        "returned unchanged — the #5 regression is back."
    )
    assert "Ask" in result.schema.column_names, (
        "'Ask' must be in result.schema.column_names after as_of_join with "
        "joins=['Bid', 'Ask']. If missing, the left-table (fills) schema was "
        "returned unchanged — the #5 regression is back."
    )

    # Definitive check: schema.validate() against the snapshot must not raise.
    result.schema.validate(snapshot)
