"""Canonical `execution` table - fills."""

from __future__ import annotations

from ibda.schema._core import Column, DType, Schema

EXECUTION = Schema(
    name="execution",
    doc="Individual fills with venue, liquidity flag, and commission.",
    # Reflects the live execDetails path (in-session, not yet reconciled), which is
    # this schema's default settled_flag. The same schema is reused by the
    # Flex-sourced adapter (adapters/ibkr/flex/arrow.py) to build the settled,
    # reconciled execution table — those rows ARE settled despite this flag.
    columns=(
        Column("ExecId", DType.STRING, nullable=False, doc="unique execution id"),
        Column("Timestamp", DType.TIMESTAMP_NS, nullable=False, doc="execution time (UTC)"),
        Column("Account", DType.STRING, doc="IBKR account id"),
        Column("ConId", DType.INT64, doc="IBKR contract id"),
        Column("OrderId", DType.INT64, nullable=True, doc="IBKR order id this fill belongs to"),
        Column(
            "OrderRef",
            DType.STRING,
            nullable=True,
            doc="Free-form order tag (e.g. STRAT-A-GOOGL); strategy attribution",
        ),
        Column("Sym", DType.STRING, nullable=False, doc="instrument symbol"),
        Column(
            "SecType",
            DType.STRING,
            nullable=True,
            doc="IBKR security type (STK/OPT/FUT/CASH/…)",
        ),
        Column("Side", DType.STRING, nullable=False, doc="BUY or SELL"),
        Column("Qty", DType.FLOAT64, nullable=False, doc="filled quantity"),
        Column("Price", DType.FLOAT64, nullable=False, doc="fill price"),
        Column(
            "Multiplier",
            DType.FLOAT64,
            nullable=True,
            doc="contract multiplier; 1.0 for stocks/FX, 100 for equity options, etc.",
        ),
        Column("Venue", DType.STRING, doc="execution exchange/venue"),
        Column("Liquidity", DType.STRING, doc="maker/taker/unknown"),
        Column(
            "Currency",
            DType.STRING,
            nullable=True,
            doc=(
                "trade/contract currency. Flex: passthrough of the Trade "
                "element's currency attribute. Live: orders_exec_details.Currency "
                "(populated per fill); null if absent."
            ),
        ),
        Column(
            "OpenClose",
            DType.STRING,
            nullable=True,
            doc=(
                "open/close indicator for the fill (e.g. 'O'=open, 'C'=close). "
                "Flex: passthrough of the Trade element's openCloseIndicator "
                "attribute. Live: orders_exec_details carries no equivalent "
                "field, so this is always null on the live path."
            ),
        ),
        Column(
            "Commission",
            DType.FLOAT64,
            doc=(
                "commission charged for this fill, joined from the broker's "
                "commissionReport callback (see ibda.schema.commission.COMMISSION) "
                "keyed by ExecId. Null means the commission report has not yet "
                "arrived (or predates this session) — never a synonym for $0.00. "
                "Sign convention: negative = cost (a fee reduces P&L). Flex's "
                "ibCommission is already negative and passes through unchanged; "
                "the live commissionReport.commission callback reports a "
                "positive magnitude and is negated by the canonical view to "
                "match."
            ),
        ),
        Column(
            "RealizedPnl",
            DType.FLOAT64,
            doc=(
                "IBKR's own broker-computed realized P&L for this fill, joined "
                "from the commissionReport callback (0.0 on opening trades that "
                "have no realized P&L yet). Null under the same unmatched-ExecId "
                "condition as Commission."
            ),
        ),
    ),
)
