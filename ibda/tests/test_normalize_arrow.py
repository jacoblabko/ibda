"""Pure unit tests for _normalize_arrow — nullable INT64 round-trip.

No JVM required.  These tests verify that ``_normalize_arrow`` correctly casts
float64 columns — produced by Deephaven when NULL_LONG values round-trip through
pandas — back to int64 when the canonical Schema declares the column as INT64.

Deephaven → Python round-trip for a NULL_LONG value:
  1. Deephaven ``to_pandas()``     → pandas float64 column with NaN
  2. ``pa.Table.from_pandas()``    → Arrow float64 column with null sentinel (NaN→null)
  3. ``_normalize_arrow()``        → Arrow int64 column with null sentinel (fix here)

The JVM-backed end-to-end test lives in ``tests_jvm/test_adapter_contract.py``
(``test_nullable_long_snapshots_as_int64_not_float64``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa

from ibda.adapters.deephaven.adapter import _normalize_arrow
from ibda.schema._core import Column, DType, Schema


def _schema(*cols: tuple[str, DType]) -> Schema:
    """Build a minimal Schema with the given (name, dtype) pairs."""
    return Schema(
        name="<test>",
        columns=tuple(Column(name=n, dtype=d, nullable=True) for n, d in cols),
    )


class TestNormalizeArrowNullableInt:
    """_normalize_arrow casts float64 → int64 when the Schema declares INT64."""

    def test_non_null_float64_cast_to_int64(self) -> None:
        """Float64 column with no nulls, declared INT64 → all values cast to int64."""
        tbl = pa.table({"ConId": pa.array([1.0, 2.0, 3.0], type=pa.float64())})
        out = _normalize_arrow(tbl, _schema(("ConId", DType.INT64)))
        assert out.schema.field("ConId").type == pa.int64()
        assert out.column("ConId").to_pylist() == [1, 2, 3]

    def test_null_float64_casts_to_nullable_int64(self) -> None:
        """Float64 with a null sentinel, declared INT64 → nullable int64 with null preserved.

        This is the core NULL_LONG round-trip test.  Deephaven NULL_LONG becomes NaN
        in pandas; ``pa.Table.from_pandas`` converts NaN to a null sentinel in the
        float64 Arrow column.  ``_normalize_arrow`` must then cast to int64 (not fail)
        and preserve the null sentinel.
        """
        # Simulate the Deephaven → pandas → Arrow round-trip for NULL_LONG:
        df = pd.DataFrame({"ConId": [1.0, np.nan, 3.0]})
        # pa.Table.from_pandas converts NaN → null sentinel in the float64 array.
        raw = pa.Table.from_pandas(df, preserve_index=False)

        # Verify the fixture: float64 column with exactly one null
        assert raw.schema.field("ConId").type == pa.float64()
        assert raw.column("ConId").null_count == 1

        out = _normalize_arrow(raw, _schema(("ConId", DType.INT64)))

        assert out.schema.field("ConId").type == pa.int64(), (
            "NULL_LONG round-trip failed: float64 (with null sentinel) "
            "was not cast to int64 by _normalize_arrow"
        )
        vals = out.column("ConId").to_pylist()
        assert vals[0] == 1
        assert vals[1] is None   # null sentinel preserved through the cast
        assert vals[2] == 3

    def test_all_null_float64_casts_to_all_null_int64(self) -> None:
        """A fully-null float64 column declared INT64 casts to a fully-null int64 column."""
        df = pd.DataFrame({"ConId": [np.nan, np.nan]})
        raw = pa.Table.from_pandas(df, preserve_index=False)
        out = _normalize_arrow(raw, _schema(("ConId", DType.INT64)))
        assert out.schema.field("ConId").type == pa.int64()
        vals = out.column("ConId").to_pylist()
        assert vals == [None, None]

    def test_fractional_float64_stays_float64_on_cast_failure(self) -> None:
        """Float64 with genuinely non-integer values, declared INT64 → cast fails, stays float64.

        The ``except`` clause in ``_normalize_arrow`` leaves the column as float64 so
        the caller's ``schema.validate()`` can surface the type mismatch explicitly,
        rather than silently truncating valid non-integer data.
        """
        tbl = pa.table({"Qty": pa.array([1.5, 2.7], type=pa.float64())})
        out = _normalize_arrow(tbl, _schema(("Qty", DType.INT64)))
        # Cast fails; column stays float64 so schema.validate() catches the mismatch
        assert out.schema.field("Qty").type == pa.float64()

    def test_float64_column_absent_from_schema_is_untouched(self) -> None:
        """A float64 column not declared in the schema must remain float64."""
        tbl = pa.table({"Qty": pa.array([1.0, 2.0], type=pa.float64())})
        # Schema knows only about ConId (INT64), not Qty
        out = _normalize_arrow(tbl, _schema(("ConId", DType.INT64)))
        assert out.schema.field("Qty").type == pa.float64()

    def test_float64_declared_float64_in_schema_stays_float64(self) -> None:
        """A float64 column explicitly declared FLOAT64 in the schema must not be cast."""
        tbl = pa.table({"Qty": pa.array([1.0, 2.0], type=pa.float64())})
        out = _normalize_arrow(tbl, _schema(("Qty", DType.FLOAT64)))
        assert out.schema.field("Qty").type == pa.float64()

    def test_no_schema_float64_is_not_touched(self) -> None:
        """When no schema is provided, float64 columns are left as-is."""
        tbl = pa.table({"ConId": pa.array([1.0, 2.0], type=pa.float64())})
        out = _normalize_arrow(tbl, None)
        assert out.schema.field("ConId").type == pa.float64()

    def test_large_string_cast_and_int64_cast_fire_together(self) -> None:
        """large_string → string and float64 → int64 both apply correctly in one pass."""
        tbl = pa.table({
            "Sym": pa.array(["AAPL", "MSFT"], type=pa.large_string()),
            "ConId": pa.array([1.0, 2.0], type=pa.float64()),
        })
        out = _normalize_arrow(
            tbl, _schema(("Sym", DType.STRING), ("ConId", DType.INT64))
        )
        assert out.schema.field("Sym").type == pa.string()
        assert out.schema.field("ConId").type == pa.int64()

    def test_multiple_null_rows_all_preserved(self) -> None:
        """Multiple null sentinels in a nullable INT64 column are all preserved after cast."""
        df = pd.DataFrame({"ConId": [np.nan, 42.0, np.nan]})
        raw = pa.Table.from_pandas(df, preserve_index=False)
        out = _normalize_arrow(raw, _schema(("ConId", DType.INT64)))
        assert out.schema.field("ConId").type == pa.int64()
        vals = out.column("ConId").to_pylist()
        assert vals[0] is None
        assert vals[1] == 42
        assert vals[2] is None
