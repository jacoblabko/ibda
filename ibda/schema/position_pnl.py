"""Canonical ``position_pnl`` table — per-position daily P&L from the broker.

Populated only after ``request_single_pnl(conids)`` is called on the IBKR adapter
(deephaven-ib ``reqPnLSingle``).  Each row corresponds to one subscribed position.
``Sym`` is resolved from ``ConId`` via the position table at subscription time and
stored here; the raw deephaven-ib ``ConId`` column arrives as a string and is cast
to int64 in the adapter.
"""

from __future__ import annotations

from ibda.schema._core import Column, DType, Schema

POSITION_PNL = Schema(
    name="position_pnl",
    doc=(
        "Per-position daily P&L.  Populated only after request_single_pnl() is"
        " called for each contract.  DailyPnl is the broker-reported daily"
        " mark-to-market P&L for the individual position."
    ),
    columns=(
        Column("Account", DType.STRING, nullable=False, doc="IBKR account id"),
        Column("ConId", DType.INT64, nullable=False, doc="IBKR contract id"),
        Column("Sym", DType.STRING, doc="instrument symbol (resolved from ConId)"),
        Column("DailyPnl", DType.FLOAT64, doc="per-position daily P&L (mark-to-market)"),
        Column("UnrealizedPnl", DType.FLOAT64, doc="unrealized P&L for this position"),
        Column("RealizedPnl", DType.FLOAT64, doc="realized P&L for this position today"),
        Column("MarketValue", DType.FLOAT64, doc="current market value of the position"),
    ),
)
