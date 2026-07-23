"""Canonical `quote` table - L1 top-of-book."""

from __future__ import annotations

from ibda.schema._core import Column, DType, Schema

QUOTE = Schema(
    name="quote",
    doc="Top-of-book L1 quotes (bid/ask/last with sizes).",
    columns=(
        Column("Sym", DType.STRING, nullable=False, doc="instrument symbol"),
        Column("Timestamp", DType.TIMESTAMP_NS, nullable=False, doc="quote time (UTC)"),
        Column("Bid", DType.FLOAT64, doc="best bid price"),
        Column("Ask", DType.FLOAT64, doc="best ask price"),
        Column("BidSize", DType.FLOAT64, doc="best bid size"),
        Column("AskSize", DType.FLOAT64, doc="best ask size"),
        Column("Last", DType.FLOAT64, doc="last trade price"),
    ),
)
