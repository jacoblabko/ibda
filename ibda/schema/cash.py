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
        Column(
            "TxnId",
            DType.STRING,
            doc=(
                "row identity, for de-duplicating the same movement seen twice (e.g. two "
                "report windows that overlap). Prefers IBKR's own transactionID, which is "
                "a true stable id but is emitted only when the Flex query definition "
                "selects that field. Otherwise a deterministic content hash of (Account, "
                "Timestamp, Type, Sym, Amount, Currency) prefixed 'syn-', so the prefix "
                "always tells the two apart. GUARANTEE: the same movement reported twice "
                "gets the same id, so re-ingesting an overlapping window cannot "
                "double-count it. NOT GUARANTEED for a 'syn-' id: two genuinely distinct "
                "movements identical in all six fields share an id and are "
                "indistinguishable. Select transactionID in the Flex query to remove that "
                "ambiguity."
            ),
        ),
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
        Column(
            "Amount",
            DType.FLOAT64,
            nullable=False,
            doc="signed cash amount, in the row's own Currency (not converted to base)",
        ),
        Column("Currency", DType.STRING, doc="currency of Amount (the row's local currency)"),
        Column(
            "FxRateToBase",
            DType.FLOAT64,
            doc=(
                "local->base conversion rate for this row: base_amount = Amount * "
                "FxRateToBase. None when the Flex query does not select the field, "
                "which is the normal case for a single-currency query. Consumers that "
                "must sum in base currency (e.g. "
                "ibda.analytics.performance.external_flows_from_cash) treat a row with "
                "a non-base Currency and no positive rate as unconvertible."
            ),
        ),
    ),
)
