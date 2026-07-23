"""Canonical `trade` table - tick-by-tick trade prints."""

from __future__ import annotations

from ibda.schema._core import Column, DType, Schema

TRADE = Schema(
    name="trade",
    doc="Tick-by-tick last-trade prints.",
    columns=(
        Column("Sym", DType.STRING, nullable=False, doc="instrument symbol"),
        Column("Timestamp", DType.TIMESTAMP_NS, nullable=False, doc="trade time (UTC)"),
        Column("Price", DType.FLOAT64, nullable=False, doc="trade price"),
        Column("Size", DType.FLOAT64, doc="trade size"),
    ),
)
