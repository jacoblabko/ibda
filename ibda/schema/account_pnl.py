"""Canonical ``account_pnl`` table — account-level daily P&L from the broker.

Auto-populated on connect (deephaven-ib calls ``reqPnL`` for each managed account
in the ``managedAccounts`` callback).  One row per account at any time; refreshes
intraday as IBKR sends updates.
"""

from __future__ import annotations

from ibda.schema._core import Column, DType, Schema

ACCOUNT_PNL = Schema(
    name="account_pnl",
    doc=(
        "Account-level daily P&L snapshot.  Auto-populated; one row per account."
        "  DailyPnl is the intraday daily P&L from the broker (mark-to-market)."
    ),
    columns=(
        Column("Account", DType.STRING, nullable=False, doc="IBKR account id"),
        Column(
            "Timestamp",
            DType.TIMESTAMP_NS,
            nullable=True,
            doc="receive-time of the last broker update (UTC)",
        ),
        Column("DailyPnl", DType.FLOAT64, doc="account daily P&L (mark-to-market)"),
        Column("UnrealizedPnl", DType.FLOAT64, doc="unrealized P&L across all positions"),
        Column("RealizedPnl", DType.FLOAT64, doc="realized P&L for the day"),
    ),
)
