"""ibda.schema._core — the pure schema vocabulary primitives.

A Schema is a declarative description of a canonical table: ordered columns,
each with a canonical DType, nullability, and a documentation string. It maps
to an Apache Arrow schema (the API's wire type) and can validate a produced
Arrow table against itself. No engine, no vendor imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import cast

import pyarrow as pa

from ibda.errors import SchemaMismatch


def _ts_ns() -> pa.DataType:
    """Return the canonical UTC nanosecond timestamp Arrow type."""
    return cast(pa.DataType, pa.timestamp("ns", tz="UTC"))


class DType(Enum):
    """Canonical, vendor-neutral column types."""

    STRING = "STRING"
    INT64 = "INT64"
    FLOAT64 = "FLOAT64"
    BOOL = "BOOL"
    TIMESTAMP_NS = "TIMESTAMP_NS"

    def to_arrow(self) -> pa.DataType:
        """Return the Arrow DataType corresponding to this DType."""
        return cast(pa.DataType, _DTYPE_TO_ARROW[self])


# Mapping from each DType member to its Arrow DataType.
# Defined after the class so DType members are fully constructed.
_DTYPE_TO_ARROW: dict[DType, pa.DataType] = {
    DType.STRING: cast(pa.DataType, pa.string()),
    DType.INT64: cast(pa.DataType, pa.int64()),
    DType.FLOAT64: cast(pa.DataType, pa.float64()),
    DType.BOOL: cast(pa.DataType, pa.bool_()),
    DType.TIMESTAMP_NS: _ts_ns(),
}


@dataclass(frozen=True)
class Column:
    """One column of a canonical table."""

    name: str
    dtype: DType
    nullable: bool = True
    doc: str = ""


@dataclass(frozen=True)
class Schema:
    """An ordered, named set of columns describing one canonical table."""

    name: str
    columns: tuple[Column, ...]
    doc: str = ""
    # True when this table's rows represent a settled/reconciled snapshot (e.g.
    # Flex-sourced) rather than a live, still-mutable in-session view.
    settled_flag: bool = field(default=False)

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for col in self.columns:
            if col.name in seen:
                raise ValueError(
                    f"duplicate column {col.name!r} in schema {self.name!r}"
                )
            seen.add(col.name)

    @property
    def column_names(self) -> tuple[str, ...]:
        """Return an ordered tuple of column names."""
        return tuple(c.name for c in self.columns)

    def to_arrow_schema(self) -> pa.Schema:
        """Build an Arrow schema from this Schema's columns."""
        return cast(
            pa.Schema,
            pa.schema(
                [
                    pa.field(c.name, c.dtype.to_arrow(), nullable=c.nullable)
                    for c in self.columns
                ]
            ),
        )

    def validate(self, table: pa.Table) -> None:
        """Raise SchemaMismatch if *table* lacks a declared column or mistypes one."""
        present: dict[str, pa.DataType] = {
            f.name: cast(pa.DataType, f.type) for f in table.schema
        }
        for col in self.columns:
            if col.name not in present:
                raise SchemaMismatch(
                    f"schema {self.name!r}: missing column {col.name!r}"
                )
            expected = col.dtype.to_arrow()
            if present[col.name] != expected:
                raise SchemaMismatch(
                    f"schema {self.name!r}: column {col.name!r} is "
                    f"{present[col.name]}, expected {expected}"
                )
