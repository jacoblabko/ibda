"""Canonical `commission` table — broker commission reports, keyed by ExecId.

Populated from deephaven-ib's ``orders_exec_commission_report`` table, which
arrives on IB's ``commissionReport`` callback — a separate, asynchronous
callback from ``execDetails`` (the source of the canonical ``execution``
table).  This table carries fields the canonical ``execution`` table's
``Commission`` column does not: ``RealizedPnl``, ``Yield``, and
``YieldRedemptionDate`` (bond executions).  Keyed by ``ExecId`` so consumers
can ``natural_join`` this onto ``execution``.

Auto-populated alongside the execution stream — no explicit subscription is
needed.
"""

from __future__ import annotations

from ibda.schema._core import Column, DType, Schema

COMMISSION = Schema(
    name="commission",
    doc=(
        "Broker commission report per execution, keyed by ExecId. Carries "
        "RealizedPnl/Yield fields the execution table's Commission column "
        "does not. Auto-populated alongside the execution stream (no "
        "explicit subscription needed)."
    ),
    columns=(
        Column(
            "ExecId",
            DType.STRING,
            nullable=False,
            doc="execution id — join key onto the canonical `execution` table",
        ),
        Column(
            "Timestamp",
            DType.TIMESTAMP_NS,
            nullable=False,
            doc="receive time of the commission report (UTC)",
        ),
        Column(
            "Commission",
            DType.FLOAT64,
            doc=(
                "commission charged for this execution. Sign convention: "
                "negative = cost (a fee reduces P&L) — the same unified "
                "convention as the canonical execution.Commission column. "
                "IB's commissionReport callback reports a positive magnitude "
                "on the live path; the canonical view negates it to match. "
                "For a given ExecId, this column and execution.Commission "
                "are equal."
            ),
        ),
        Column("Currency", DType.STRING, doc="commission currency"),
        Column(
            "RealizedPnl",
            DType.FLOAT64,
            doc="realized P&L attributed to this execution (0.0 on opening trades)",
        ),
        Column(
            "Yield",
            DType.FLOAT64,
            doc="bond yield at execution; null for non-bond instruments",
        ),
        Column(
            "YieldRedemptionDate",
            DType.STRING,
            doc="bond yield redemption date string; null for non-bond instruments",
        ),
    ),
)
