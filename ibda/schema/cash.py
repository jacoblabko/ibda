"""Canonical `cash` table - cash movements (dividends/interest/fees/deposits/withdrawals).

Flex is the authoritative source for this table: it is the settled, reconciled
record of account cash activity, as opposed to a live in-session view.
"""

from __future__ import annotations

from ibda.schema._core import Column, DType, Schema

CASH = Schema(
    name="cash",
    doc="Cash movements: dividends, interest, fees, deposits, withdrawals, transfers. "
    "Authoritative source is Flex (the settled/reconciled record of account activity).",
    settled_flag=True,
    columns=(
        Column("Account", DType.STRING, doc="IBKR account id"),
        Column("Timestamp", DType.TIMESTAMP_NS, nullable=False, doc="value time (UTC)"),
        Column(
            "Type",
            DType.STRING,
            nullable=False,
            doc=(
                "movement category: raw Flex category strings, mixed case, as emitted "
                "by IBKR (e.g. 'Dividends', 'Broker Interest Received', 'Withholding "
                "Tax', 'Transfer') — not a normalized lowercase enum. Downstream "
                "consumers (ibda.analytics.performance.external_flows_from_cash) match "
                "case-insensitively on substrings such as 'deposit', 'withdraw', and "
                "'transfer' to identify external capital flows to strip from returns "
                "when flow-adjusting."
            ),
        ),
        Column("Sym", DType.STRING, doc="related instrument symbol if any"),
        Column("Amount", DType.FLOAT64, nullable=False, doc="signed cash amount"),
        Column("Currency", DType.STRING, doc="currency"),
    ),
)
