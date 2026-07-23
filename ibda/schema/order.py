"""Canonical `order` table - live order state and lifecycle."""

from __future__ import annotations

from ibda.schema._core import Column, DType, Schema

ORDER = Schema(
    name="order",
    doc="Live order state and lifecycle status.",
    columns=(
        Column("OrderId", DType.INT64, nullable=False, doc="broker order id"),
        Column("PermId", DType.INT64, doc="IBKR permanent id"),
        Column("Account", DType.STRING, doc="IBKR account id"),
        Column("Sym", DType.STRING, nullable=False, doc="instrument symbol"),
        Column("Side", DType.STRING, nullable=False, doc="BUY or SELL"),
        Column("Qty", DType.FLOAT64, nullable=False, doc="ordered quantity"),
        Column("FilledQty", DType.FLOAT64, doc="cumulative filled quantity"),
        Column("LimitPrice", DType.FLOAT64, doc="limit price if any"),
        Column("Status", DType.STRING, doc="order status (Submitted/Filled/Cancelled/...)"),
        Column("Timestamp", DType.TIMESTAMP_NS, doc="last status update time (UTC)"),
    ),
)
