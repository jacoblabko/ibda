"""Canonical `news_provider` table — news provider code -> display name.

Populated from deephaven-ib's ``news_providers`` table: a small, auto-populated
reference/dimension table (available news feed subscriptions), listed on
connect with no explicit request needed. The canonical ``news`` table already
joins this in-place to resolve ``ProviderName`` inline; this table exposes the
same dimension directly, so a consumer can browse/join it independently
(e.g. to render a provider picker) without depending on the ``news`` join.

Deduplicated by ``Code`` (``last_by``) in case of a re-emitted row; in
practice this table rarely if ever changes after connect.
"""

from __future__ import annotations

from ibda.schema._core import Column, DType, Schema

NEWS_PROVIDER = Schema(
    name="news_provider",
    doc="News provider code -> display name dimension table, auto-populated on connect.",
    columns=(
        Column(
            "Code",
            DType.STRING,
            nullable=False,
            doc="news provider short code (e.g. FLY, BRFG)",
        ),
        Column("Name", DType.STRING, nullable=False, doc="news provider display name"),
    ),
)
