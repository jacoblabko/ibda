"""Canonical `nav` table — daily account net-asset-value (NAV) time series.

This is the foundation for historic performance analytics (returns, Sharpe,
Sortino, drawdown). Its authoritative source is the Flex
``EquitySummaryByReportDateInBase`` section — one row per report date giving the
account's total NAV in base currency, broken into cash and stock components.

A daily NAV series is what turns raw P&L into *returns*: ``r_t`` is derived from
``Total_t`` relative to ``Total_{t-1}`` (adjusted for external cash flows), and a
stream of returns is what every risk-adjusted metric is built on. P&L alone is
not enough — see ``ibda.analytics.performance``.
"""

from __future__ import annotations

from ibda.schema._core import Column, DType, Schema

NAV = Schema(
    name="nav",
    doc=(
        "Daily account NAV in base currency. One row per report date. "
        "Authoritative source is Flex EquitySummaryByReportDateInBase. "
        "Basis for historic return and risk-adjusted performance metrics."
    ),
    settled_flag=True,
    columns=(
        Column("Account", DType.STRING, doc="IBKR account id"),
        Column(
            "Timestamp",
            DType.TIMESTAMP_NS,
            nullable=False,
            doc="report date at ET midnight, converted to UTC (one per day)",
        ),
        Column(
            "Total",
            DType.FLOAT64,
            nullable=False,
            doc="total account NAV in base currency at end of report date",
        ),
        Column("Cash", DType.FLOAT64, doc="cash component of NAV (base currency)"),
        Column("Stock", DType.FLOAT64, doc="stock/position component of NAV (base currency)"),
    ),
)
