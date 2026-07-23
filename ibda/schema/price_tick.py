"""Canonical `price_tick` table — streaming LAST/CLOSE/DELAYED_* price ticks.

Populated from deephaven-ib's ``ticks_price`` table: streaming-until-requested
(empty until ``request_market_data()`` is called for a contract — the same
lifecycle as ``quote``/``trade``, which come from separate raw feeds
(``ticks_bid_ask`` / tick-by-tick LAST subscription respectively). Unlike
those two, ``ticks_price`` multiplexes several *tick types* per contract onto
ONE stream — ``LAST``, ``CLOSE``, ``DELAYED_LAST``, ``DELAYED_CLOSE`` (and
others) — distinguished by the ``TickType`` column. This canonical table
preserves ``TickType`` as a dimension rather than collapsing it, so a
consumer can distinguish a live last-trade tick from a delayed/close price.

``Sym`` is populated directly from ``ticks_price.Symbol`` — the raw table
already carries the symbol per row, so no join to ``definition`` is needed.

Append-only event log; no dedupe — every tick is retained.
"""

from __future__ import annotations

from ibda.schema._core import Column, DType, Schema

PRICE_TICK = Schema(
    name="price_tick",
    doc=(
        "Streaming price ticks from ticks_price (LAST/CLOSE/DELAYED_LAST/"
        "DELAYED_CLOSE/...), keyed by TickType. Empty until "
        "request_market_data() is called for a contract; append-only "
        "thereafter."
    ),
    columns=(
        Column("ConId", DType.INT64, nullable=False, doc="IBKR contract id"),
        Column("Sym", DType.STRING, doc="instrument symbol"),
        Column(
            "TickType",
            DType.STRING,
            nullable=False,
            doc="LAST | CLOSE | DELAYED_LAST | DELAYED_CLOSE | ...",
        ),
        Column("Price", DType.FLOAT64, doc="tick price"),
        Column("Timestamp", DType.TIMESTAMP_NS, nullable=False, doc="tick receive time (UTC)"),
    ),
)
