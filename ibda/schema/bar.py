"""Canonical `bar` table - OHLCV."""

from __future__ import annotations

from ibda.schema._core import Column, DType, Schema

BAR = Schema(
    name="bar",
    doc="OHLCV bars at a fixed interval.",
    columns=(
        Column("Sym", DType.STRING, nullable=False, doc="instrument symbol"),
        Column("Timestamp", DType.TIMESTAMP_NS, nullable=False, doc="bar close time (UTC)"),
        Column("Open", DType.FLOAT64, doc="open price"),
        Column("High", DType.FLOAT64, doc="high price"),
        Column("Low", DType.FLOAT64, doc="low price"),
        Column("Close", DType.FLOAT64, doc="close price"),
        Column("Volume", DType.FLOAT64, doc="volume"),
    ),
)
