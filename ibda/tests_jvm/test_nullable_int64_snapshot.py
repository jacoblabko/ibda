"""JVM regression test: NULL_LONG long column → int64 after snapshot.

Characterises the nullable INT64 round-trip: whether a Deephaven ``long`` column
containing NULL_LONG deserialises as float64 (the failure mode anticipated ahead
of time) or as int64 (the current Deephaven behaviour).

## What the JVM test found (run 2026-06-30)

The bug does NOT reproduce.  Deephaven's ``to_pandas()`` returns
``pd.Int64Dtype()`` (nullable pandas integer) for a ``long`` column with a null
row — NOT float64.  That means:

  1. ``to_pandas()``   → pd.Int64Dtype Series; null row is pd.NA (not NaN)
  2. ``from_pandas()`` → Arrow int64 with null bitmap set (no float64 round-trip)
  3. ``_normalize_arrow``'s float64→int64 cast guard is NEVER triggered
  4. ``schema.validate()`` passes regardless — the column is already int64

The docstring in ``adapter._normalize_arrow`` ("round-trips through to_pandas as
NaN, which forces the column to float64") describes the OLD behaviour from an
earlier Deephaven version that predated ``pd.Int64Dtype()`` support in
``deephaven-server``.  The guard code (lines 56-69 of adapter.py) is now a dead
code path for this Deephaven version, but is kept as forward-compatibility
protection in case an older Deephaven version is ever used.

## Tests in this file

Steps 1–3 document the ACTUAL JVM round-trip behaviour.
Steps 4–5b unit-test the float64→int64 guard in ``_normalize_arrow`` using a
  synthetic Arrow table (float64 column, no JVM involvement beyond what the
  session fixture provides).  These tests exercise the dead-code path directly
  so that if Deephaven ever regresses to float64 for nullable longs the guard
  still has test coverage.
Step 6 is the end-to-end smoke test through ``DeephavenPort._snapshot``.

Run with:
    uv run pytest ibda/tests_jvm/test_nullable_int64_snapshot.py -v
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pyarrow as pa
import pytest

from ibda.adapters.deephaven.adapter import DeephavenPort, _normalize_arrow
from ibda.errors import SchemaMismatch
from ibda.schema._core import Column, DType, Schema
from ibda.schema.execution import EXECUTION

# ---------------------------------------------------------------------------
# A minimal two-column schema for the intermediate-step tests.
# Using a local Schema avoids any dependency on live IBKR schema constants.
# ---------------------------------------------------------------------------
_SYNTH_SCHEMA: Schema = Schema(
    name="synth",
    columns=(
        Column("Sym", DType.STRING, nullable=False),
        Column("ConId", DType.INT64, nullable=True),
    ),
)


# ---------------------------------------------------------------------------
# Table builders — all deephaven imports deferred inside function bodies
# so they only execute when the JVM is live.
# ---------------------------------------------------------------------------

def _nullable_long_table() -> Any:
    """Two-row DH table: ConId = [12345, NULL_LONG].

    Row 0 is a valid long; row 1 is NULL_LONG (passed as Python ``None``).
    """
    from deephaven import new_table  # noqa: PLC0415 — JVM-gated
    from deephaven.column import long_col, string_col  # noqa: PLC0415

    return new_table([
        string_col("Sym",   ["AAPL", "MSFT"]),
        long_col("ConId",   [12345,  None]),  # None → NULL_LONG
    ])


def _canonical_execution_table_with_null_conid() -> Any:
    """Execution-shaped table with canonical column names; row 1 has NULL_LONG ConId.

    All columns match the EXECUTION schema exactly (canonical names, canonical
    types).  Commission and Liquidity are set to real values here — they are
    null-filled in the live path but having them non-null avoids confounding
    the ConId null assertion.
    """
    from deephaven import new_table  # noqa: PLC0415
    from deephaven.column import (  # noqa: PLC0415
        datetime_col,
        double_col,
        long_col,
        string_col,
    )
    from deephaven.time import to_j_instant  # noqa: PLC0415

    t0 = to_j_instant("2026-06-10T15:00:00 UTC")
    t1 = to_j_instant("2026-06-10T15:01:00 UTC")
    return new_table([
        string_col("ExecId",       ["e1",     "e2"]),
        datetime_col("Timestamp",  [t0,        t1]),
        string_col("Account",      ["DU1",    "DU1"]),
        long_col("ConId",          [12345,    None]),   # row 1: NULL_LONG
        long_col("OrderId",        [987,       988]),
        string_col("OrderRef",     ["TAG1",   "TAG2"]),
        string_col("Sym",          ["AAPL",   "MSFT"]),
        string_col("SecType",      ["STK",    "STK"]),
        string_col("Side",         ["BUY",    "SELL"]),
        double_col("Qty",          [100.0,    200.0]),
        double_col("Price",        [155.0,    250.0]),
        double_col("Multiplier",   [1.0,      1.0]),
        string_col("Venue",        ["NYSE",   "NASDAQ"]),
        string_col("Liquidity",    ["taker",  "maker"]),
        string_col("Currency",     ["USD",    "USD"]),
        string_col("OpenClose",    [None,     None]),
        double_col("Commission",   [1.0,      2.0]),
        double_col("RealizedPnl",  [0.0,      0.5]),
    ])


def _raw_arrow(dh_table: Any) -> pa.Table:
    """Run to_pandas → from_pandas without _normalize_arrow."""
    from deephaven.pandas import to_pandas  # noqa: PLC0415

    df: pd.DataFrame = to_pandas(dh_table)
    return pa.Table.from_pandas(df, preserve_index=False)


def _synthetic_float64_arrow() -> pa.Table:
    """Build a float64 Arrow table with a null, WITHOUT involving Deephaven.

    Represents what older Deephaven versions used to return (NULL_LONG → NaN →
    float64), used to unit-test the float64→int64 guard in _normalize_arrow
    even though that path is not hit by the current Deephaven to_pandas().
    """

    sym_array = pa.array(["AAPL", "MSFT"], type=pa.large_string())
    # float64 with a null at index 1 — mimics old Deephaven NaN round-trip after
    # pa.Table.from_pandas converts pandas NaN → Arrow null.
    conid_array = pa.array([12345.0, None], type=pa.float64())
    return pa.table(
        {"Sym": sym_array, "ConId": conid_array},
        schema=pa.schema([
            pa.field("Sym", pa.large_string()),
            pa.field("ConId", pa.float64()),
        ]),
    )


# =========================================================================
# Steps 1–3: ACTUAL Deephaven JVM behaviour (as observed 2026-06-30)
# =========================================================================

def test_step1_to_pandas_null_long_is_nullable_int64_not_float() -> None:
    """Deephaven to_pandas() returns pd.Int64Dtype() for nullable long columns.

    This is the key finding of the JVM investigation.  The anticipated failure mode
    ("NULL_LONG → NaN → float64") does NOT reproduce on the current Deephaven
    version.  Deephaven now uses pandas nullable integer (Int64Dtype), so
    NULL_LONG becomes pd.NA rather than NaN, and the column stays integer.

    If this assertion fails with dtype float64, Deephaven has reverted to the
    old NaN-based approach.  In that case:
    - from_pandas() would produce Arrow float64 (not int64)
    - _normalize_arrow's float64→int64 guard (adapter.py lines 56-69) would
      become the live fix again
    - this would need reopening and steps 4–5b below test the fix path
    """
    from deephaven.pandas import to_pandas  # noqa: PLC0415

    df: pd.DataFrame = to_pandas(_nullable_long_table())
    dtype = df["ConId"].dtype
    assert isinstance(dtype, pd.Int64Dtype), (
        f"Expected pd.Int64Dtype() (nullable pandas integer); got {dtype!r}. "
        "Deephaven may have reverted to float64 for nullable longs — see docstring."
    )


def test_step2_from_pandas_int64dtype_yields_arrow_int64_with_null() -> None:
    """pa.Table.from_pandas(Int64Dtype Series) → Arrow int64 with null bitmap.

    Because Deephaven returns pd.Int64Dtype (not float64), from_pandas produces
    Arrow int64 directly.  No float64 intermediate exists.  The null at index 1
    is an Arrow null (is_valid=False) without any NaN → null conversion.
    """
    raw = _raw_arrow(_nullable_long_table())

    # Arrow type is already int64 (NOT float64) at this stage.
    assert raw.schema.field("ConId").type == pa.int64(), (
        f"Expected int64 Arrow type (from pd.Int64Dtype); "
        f"got {raw.schema.field('ConId').type}. "
        "If this is float64, Deephaven has reverted to the old NaN path."
    )

    # The null position must be a proper Arrow null.
    col_arr = raw.column("ConId")
    assert col_arr[1].is_valid is False, (
        "Index 1 (NULL_LONG) must be Arrow null (is_valid=False)"
    )
    assert col_arr[1].as_py() is None, (
        f"Arrow null at index 1 must deserialise to Python None; "
        f"got {col_arr[1].as_py()!r}"
    )


def test_step3_no_float64_intermediate_normalize_is_passthrough() -> None:
    """_normalize_arrow does NOT need to fix int64 columns — they arrive correct.

    Because the current Deephaven returns int64 from to_pandas, _normalize_arrow
    has nothing to repair.  The column is already int64 in the raw Arrow table,
    and _normalize_arrow leaves it unchanged regardless of whether schema is
    provided or not (the float64 guard's `pa.types.is_floating(int64)` is False).
    """
    raw = _raw_arrow(_nullable_long_table())

    # Both paths (with and without schema) leave ConId as int64.
    normalized_no_schema = _normalize_arrow(raw, schema=None)
    normalized_with_schema = _normalize_arrow(raw, schema=_SYNTH_SCHEMA)

    assert normalized_no_schema.schema.field("ConId").type == pa.int64(), (
        "ConId must be int64 after _normalize_arrow(schema=None)"
    )
    assert normalized_with_schema.schema.field("ConId").type == pa.int64(), (
        "ConId must be int64 after _normalize_arrow(schema=_SYNTH_SCHEMA)"
    )

    # schema.validate() passes without any float64 correction.
    _SYNTH_SCHEMA.validate(normalized_with_schema)


# =========================================================================
# Steps 4–5b: unit-test the float64→int64 guard in _normalize_arrow.
#
# These tests exercise adapter.py lines 56-69 directly using a synthetic
# float64 Arrow table.  This path is not triggered by current Deephaven
# (which produces int64 directly), but the guard must still work correctly
# in case Deephaven reverts or an older version is used.
# =========================================================================

def test_step4_normalize_float64_with_schema_casts_to_int64() -> None:
    """_normalize_arrow float64→int64 guard: float64 column declared INT64 is cast.

    Directly exercises adapter.py lines 56-69 using a synthetic float64 Arrow
    table (not produced by Deephaven — built manually to test the dead code path).
    When the schema declares ConId as INT64 and the actual Arrow type is float64,
    _normalize_arrow must cast it back to int64.
    """
    raw = _synthetic_float64_arrow()
    # Verify our synthetic table IS float64 before normalization.
    assert raw.schema.field("ConId").type == pa.float64()

    normalized = _normalize_arrow(raw, schema=_SYNTH_SCHEMA)

    field_type = normalized.schema.field("ConId").type
    assert field_type == pa.int64(), (
        f"_normalize_arrow must cast float64 → int64 when schema declares INT64; "
        f"got {field_type}"
    )


def test_step5_float64_null_preserved_after_int64_cast() -> None:
    """float64 null survives as Arrow null (not sentinel int) after float64→int64 cast.

    Arrow's cast from float64-with-null to int64 preserves the null bitmap.
    The resulting column must have Arrow null at index 1, not -9223372036854775808.
    """
    raw = _synthetic_float64_arrow()
    normalized = _normalize_arrow(raw, schema=_SYNTH_SCHEMA)
    col_arr = normalized.column("ConId")

    # Row 0: valid integer 12345 survives the cast.
    assert col_arr[0].as_py() == 12345, (
        f"Non-null ConId must survive the float64→int64 cast as 12345; "
        f"got {col_arr[0].as_py()!r}"
    )

    # Row 1: must be Arrow null, NOT the integer sentinel -9223372036854775808.
    assert col_arr[1].is_valid is False, (
        "After float64→int64 cast, null must remain Arrow null (is_valid=False)"
    )
    assert col_arr[1].as_py() is None, (
        f"Arrow null must deserialise to Python None after cast; "
        f"got {col_arr[1].as_py()!r}"
    )


def test_step5b_schema_validate_passes_after_float64_cast() -> None:
    """schema.validate() passes after _normalize_arrow's float64→int64 correction."""
    raw = _synthetic_float64_arrow()
    normalized = _normalize_arrow(raw, schema=_SYNTH_SCHEMA)
    _SYNTH_SCHEMA.validate(normalized)  # must not raise


def test_step5c_schema_validate_fails_without_schema_on_float64() -> None:
    """Without schema, float64 stays float64 and schema.validate() raises SchemaMismatch.

    Confirms the guard only fires when schema is provided, and that the validate()
    check correctly catches the unconverted float64 column.
    """
    raw = _synthetic_float64_arrow()
    normalized_no_schema = _normalize_arrow(raw, schema=None)

    # Still float64 (no correction).
    assert normalized_no_schema.schema.field("ConId").type == pa.float64()

    with pytest.raises(SchemaMismatch, match="ConId"):
        _SYNTH_SCHEMA.validate(normalized_no_schema)


# =========================================================================
# Step 6: End-to-end through DeephavenPort._snapshot → EXECUTION.validate()
# =========================================================================

def test_step6_end_to_end_null_conid_via_port_snapshot() -> None:
    """Full canonical path: DeephavenPort._snapshot → EXECUTION.validate().

    This is the production-identical code path:
      DeephavenPort.table("execution") wraps the DH table with EXECUTION schema.
      result.snapshot() calls _snapshot → to_pandas → from_pandas → _normalize_arrow.
      The returned Arrow table must pass EXECUTION.validate() with ConId as int64.

    Specifically asserts:
    - ConId is int64 (not float64) in the output Arrow table
    - Row 0 ConId = 12345 (valid value preserved)
    - Row 1 ConId = None (Arrow null for NULL_LONG)
    - EXECUTION.validate() raises no SchemaMismatch
    """
    tbl = _canonical_execution_table_with_null_conid()
    port = DeephavenPort({"execution": tbl})
    arrow_out = port.table("execution").snapshot()

    # ConId must be int64 after snapshot.
    conid_type = arrow_out.schema.field("ConId").type
    assert conid_type == pa.int64(), (
        f"End-to-end: ConId must be int64 in Arrow output; got {conid_type}. "
        "NULL_LONG has leaked as float64 — check _normalize_arrow or Deephaven's "
        "to_pandas() behaviour (did Deephaven revert from Int64Dtype to float64?)."
    )

    # Null must survive as Arrow null, not garbage integer.
    rows = arrow_out.to_pylist()
    assert rows[0]["ConId"] == 12345, (
        f"Row 0 ConId must be 12345; got {rows[0]['ConId']!r}"
    )
    assert rows[1]["ConId"] is None, (
        f"Row 1 ConId (NULL_LONG) must be None in output; got {rows[1]['ConId']!r}"
    )

    # Schema validation — the definitive correctness check.
    EXECUTION.validate(arrow_out)
