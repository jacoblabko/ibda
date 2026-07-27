"""Canonical `cash_balance` table — per-currency ledger balances, LIVE.

Populated from deephaven-ib's ``accounts_overview`` key-value table. IBKR
reports the per-currency ledger under ``$LEDGER-``-prefixed ``Key`` values
(e.g. ``$LEDGER-CashBalance``, ``$LEDGER-NetLiquidationByCurrency``), one row
per (Account, Currency) pair — distinct from the account-level summary keys
the canonical ``account`` table pivots (those never carry more than one row
and have no ``Currency`` dimension; verified live against a real IBKR paper
session).

Built by ``ibda.adapters.deephaven.views.build_cash_balance_view`` (a
hand-written key-value pivot, not a plain ``apply_canonical_view`` rename —
the source is key-value, not columnar, so it cannot be expressed as an
``IbkrTableSpec``). Includes every currency row reported, including the
home-currency ``BASE`` mirror row; live, ticking as TWS pushes ledger updates.
"""

from __future__ import annotations

from ibda.schema._core import Column, DType, Schema

CASH_BALANCE = Schema(
    name="cash_balance",
    doc=(
        "Per-currency ledger balances (cash + net-liq-by-currency), live from "
        "accounts_overview's $LEDGER-* tags. One row per (Account, Currency), "
        "including the BASE mirror row."
    ),
    columns=(
        Column("Account", DType.STRING, nullable=False, doc="IBKR account id"),
        Column(
            "Currency",
            DType.STRING,
            nullable=False,
            doc="ledger currency code (includes the home-currency BASE row)",
        ),
        Column("CashBalance", DType.FLOAT64, doc="cash balance held in this currency"),
        Column(
            "NetLiquidation",
            DType.FLOAT64,
            doc="net liquidation value attributable to this currency",
        ),
        Column(
            "ExchangeRate",
            DType.FLOAT64,
            doc=(
                "FX rate vs. the account base currency, from the "
                "$LEDGER-ExchangeRate tag (1.0 for the BASE row)"
            ),
        ),
    ),
)
