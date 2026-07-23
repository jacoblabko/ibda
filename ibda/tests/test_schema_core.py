from __future__ import annotations

import pyarrow as pa
import pytest

from ibda.errors import SchemaMismatch
from ibda.schema._core import Column, DType, Schema


def _sample_schema() -> Schema:
    return Schema(
        name="sample",
        columns=(
            Column("Sym", DType.STRING, nullable=False, doc="instrument symbol"),
            Column("Price", DType.FLOAT64, nullable=False, doc="trade price"),
        ),
        doc="a sample table",
    )


def test_column_names_unique_enforced() -> None:
    with pytest.raises(ValueError, match="duplicate column"):
        Schema(name="bad", columns=(Column("A", DType.INT64), Column("A", DType.INT64)), doc="x")


def test_to_arrow_schema_maps_dtypes() -> None:
    arrow = _sample_schema().to_arrow_schema()
    assert arrow.field("Sym").type == pa.string()
    assert arrow.field("Price").type == pa.float64()


def test_validate_accepts_conforming_table() -> None:
    s = _sample_schema()
    tbl = pa.table({"Sym": ["AAPL"], "Price": [1.0]}, schema=s.to_arrow_schema())
    s.validate(tbl)  # must not raise


def test_validate_rejects_missing_column() -> None:
    s = _sample_schema()
    tbl = pa.table({"Sym": ["AAPL"]})
    with pytest.raises(SchemaMismatch, match="Price"):
        s.validate(tbl)
