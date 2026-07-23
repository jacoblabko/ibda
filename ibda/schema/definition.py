"""Canonical `definition` table - contract/instrument reference."""

from __future__ import annotations

from ibda.schema._core import Column, DType, Schema

DEFINITION = Schema(
    name="definition",
    doc="Instrument reference data keyed by contract id.",
    columns=(
        Column("ConId", DType.INT64, nullable=False, doc="IBKR contract id"),
        Column("Sym", DType.STRING, nullable=False, doc="instrument symbol"),
        Column("SecType", DType.STRING, doc="STK/OPT/FUT/CASH/..."),
        Column("Exchange", DType.STRING, doc="primary exchange"),
        Column("Currency", DType.STRING, doc="trading currency"),
        Column("Multiplier", DType.FLOAT64, doc="contract multiplier"),
        Column(
            "Right",
            DType.STRING,
            doc="option right: C(all)/P(ut); null for non-options",
        ),
        Column(
            "Strike",
            DType.FLOAT64,
            doc="option/future strike price; null for non-derivative contracts",
        ),
        Column(
            "LastTradeDateOrContractMonth",
            DType.STRING,
            doc="option/future expiry (YYYYMMDD) or contract month (YYYYMM)",
        ),
        Column(
            "LocalSymbol",
            DType.STRING,
            doc="exchange-native symbol (e.g. OCC-formatted option symbol)",
        ),
    ),
)
