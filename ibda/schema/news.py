"""Canonical `news` table — historical news headlines by contract.

Populated from deephaven-ib's ``news_historical`` table, which is
**streaming-until-requested**: empty until ``request_news_historical()`` is
called for a contract (see ``ibda.adapters.ibkr.marketdata.subscribe_news``),
identical in spirit to the ``quote``/``bar`` tables and ``ticks_bid_ask``/
``bars_realtime``.  Joined against the auto-populated ``news_providers``
dimension table (provider code → display name) so ``ProviderName`` is
resolved without a second round trip.

Time-ordered event log; no dedupe — a contract may have many historical
headlines and every one is retained.
"""

from __future__ import annotations

from ibda.schema._core import Column, DType, Schema

NEWS = Schema(
    name="news",
    doc=(
        "Historical news headlines by contract. Empty until subscribe_news() "
        "/ request_news_historical() is called for a contract; append-only "
        "thereafter."
    ),
    columns=(
        Column(
            "Timestamp",
            DType.TIMESTAMP_NS,
            nullable=False,
            doc="headline publish time (UTC)",
        ),
        Column("Sym", DType.STRING, doc="instrument symbol the headline is about"),
        Column("ConId", DType.INT64, doc="IBKR contract id"),
        Column("ProviderCode", DType.STRING, doc="news provider short code (e.g. FLY, BRFG)"),
        Column(
            "ProviderName",
            DType.STRING,
            doc="news provider display name, resolved via join to news_providers",
        ),
        Column(
            "ArticleId",
            DType.STRING,
            doc="provider article id; pass to request_news_article() for full text",
        ),
        Column("Headline", DType.STRING, nullable=False, doc="headline text"),
    ),
)
