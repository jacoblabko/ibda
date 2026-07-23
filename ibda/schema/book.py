"""Canonical `book` table - L2/L3 depth.

Vocabulary defined now so the schema cannot fork later; the adapter that
populates it is deferred (microstructure is a Databento-era concern).
"""

from __future__ import annotations

from ibda.schema._core import Column, DType, Schema

BOOK = Schema(
    name="book",
    doc="Order-book depth levels (L2/L3). Adapter deferred to a later sub-project.",
    columns=(
        Column("Sym", DType.STRING, nullable=False, doc="instrument symbol"),
        Column("Timestamp", DType.TIMESTAMP_NS, nullable=False, doc="update time (UTC)"),
        Column("Side", DType.STRING, nullable=False, doc="BID or ASK"),
        Column("Level", DType.INT64, nullable=False, doc="depth level (0 = top)"),
        Column("Price", DType.FLOAT64, doc="price at this level"),
        Column("Size", DType.FLOAT64, doc="size at this level"),
    ),
)
