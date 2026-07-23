"""Canonical `position` table - per-instrument holdings with cost basis and live valuation.

Right/Strike/LastTradeDateOrContractMonth/LocalSymbol/Multiplier/Currency/
Exchange are sourced directly from the position/portfolio feeds, not from a
join to the `definition` table, so these contract-identifying fields are
always populated for every held contract (see the provenance note below for
why).
"""

from __future__ import annotations

from ibda.schema._core import Column, DType, Schema

# Provenance: Right/Strike/LastTradeDateOrContractMonth/LocalSymbol/Multiplier/
# Currency/Exchange come directly from `accounts_positions`/`accounts_portfolio`
# (NOT via a join to `definition`/`contracts_details`). deephaven-ib's
# updatePortfolio/position callbacks both log the full Contract object (see
# `logger_contract` in the vendored deephaven-ib fork), so these fields are
# always in sync with the position, for every held contract. `contracts_details`
# (the source of the canonical `definition` table) is populated only per
# explicit reqContractDetails request — sparse in practice (measured: 12 rows for a
# large live book, >2,000 positions) — so it is NOT a reliable source for a
# full-book positions view. This was confirmed by direct investigation while
# migrating a downstream positions view off `contracts_details` as its source.

POSITION = Schema(
    name="position",
    doc="Current holdings per instrument with cost basis and live valuation.",
    columns=(
        Column("Account", DType.STRING, nullable=False, doc="IBKR account id"),
        Column("ConId", DType.INT64, nullable=False, doc="IBKR contract id"),
        Column("Sym", DType.STRING, nullable=False, doc="instrument symbol"),
        Column("SecType", DType.STRING, doc="IBKR security type (e.g. STK, OPT, FUT)"),
        Column("Qty", DType.FLOAT64, nullable=False, doc="signed position quantity"),
        Column("AvgCost", DType.FLOAT64, doc="average cost per unit"),
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
        Column("Multiplier", DType.FLOAT64, doc="contract multiplier"),
        Column("Currency", DType.STRING, doc="trading currency"),
        Column(
            "Exchange",
            DType.STRING,
            doc="routing/primary exchange as reported on the position's Contract",
        ),
        # MarketPrice/MarketValue/UnrealizedPnl MUST remain the last 3 columns:
        # ibda.adapters.deephaven.views.enrich_position_with_marks drops and
        # re-appends exactly these 3 (in this order) when joining live marks
        # from accounts_portfolio onto a position view built from
        # accounts_positions — see that function's docstring.
        Column("MarketPrice", DType.FLOAT64, doc="last/mark price"),
        Column("MarketValue", DType.FLOAT64, doc="Qty * MarketPrice * multiplier"),
        Column("UnrealizedPnl", DType.FLOAT64, doc="unrealized P&L"),
    ),
)
