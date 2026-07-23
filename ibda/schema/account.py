"""Canonical `account` table - balances / NAV / margin snapshots over time."""

from __future__ import annotations

from ibda.schema._core import Column, DType, Schema

ACCOUNT = Schema(
    name="account",
    doc="Point-in-time account balances and margin metrics.",
    columns=(
        Column("Account", DType.STRING, nullable=False, doc="IBKR account id"),
        Column("Timestamp", DType.TIMESTAMP_NS, nullable=False, doc="snapshot time (UTC)"),
        Column("NetLiquidation", DType.FLOAT64, doc="total net liquidation value"),
        Column("BuyingPower", DType.FLOAT64, doc="available buying power"),
        Column("MaintMargin", DType.FLOAT64, doc="maintenance margin requirement"),
        Column("GrossPositionValue", DType.FLOAT64, doc="gross market value of positions"),
        Column("Currency", DType.STRING, doc="reporting currency"),
        Column(
            "TotalCashValue",
            DType.FLOAT64,
            doc=(
                "account-level cash summary, base-currency-converted and "
                "summed across all currencies held (accounts_overview's "
                "TotalCashValue tag)"
            ),
        ),
    ),
)
