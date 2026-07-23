"""JVM test: apply_canonical_view for the price_tick and news_provider specs.

Mirrors the pattern in test_commission_news_errors_views.py: constructs
raw-shaped in-process tables, applies apply_canonical_view, snapshots through
the port, and asserts schema.validate passes and renamed columns carry the
right values.

Run with:
    uv run pytest ibda/tests_jvm/test_price_tick_news_provider_views.py -q
"""
from __future__ import annotations

from typing import Any

import pytest

import ibda
from ibda.adapters.deephaven.views import apply_canonical_view
from ibda.adapters.ibkr.specs import CANONICAL_SPECS
from ibda.schema import NEWS_PROVIDER, PRICE_TICK


# ---------------------------------------------------------------------------
# price_tick spec — ticks_price -> canonical price_tick
# ---------------------------------------------------------------------------

def _raw_price_tick_table() -> Any:
    from deephaven import new_table
    from deephaven.column import datetime_col, double_col, long_col, string_col
    from deephaven.time import to_j_instant

    t0 = to_j_instant("2026-06-10T15:00:00 UTC")
    return new_table([
        datetime_col("ReceiveTime", [t0]),
        long_col("ContractId", [12345]),
        string_col("Symbol", ["AAPL"]),
        string_col("TickType", ["LAST"]),
        double_col("Price", [291.55]),
    ])


def test_price_tick_view_conforms_to_schema() -> None:
    spec = CANONICAL_SPECS["price_tick"]
    viewed = apply_canonical_view(_raw_price_tick_table(), spec, PRICE_TICK)
    port = ibda.connect({"price_tick": viewed})
    arrow = port.table("price_tick").snapshot()
    PRICE_TICK.validate(arrow)
    assert arrow.num_rows == 1


def test_price_tick_view_renames_columns() -> None:
    spec = CANONICAL_SPECS["price_tick"]
    viewed = apply_canonical_view(_raw_price_tick_table(), spec, PRICE_TICK)
    port = ibda.connect({"price_tick": viewed})
    row = port.table("price_tick").snapshot().to_pylist()[0]
    assert row["ConId"] == 12345
    assert row["Sym"] == "AAPL"
    assert row["TickType"] == "LAST"
    assert row["Price"] == pytest.approx(291.55)


def test_price_tick_no_dedupe_keeps_every_tick_type() -> None:
    """price_tick has no dedupe_keys — LAST and CLOSE ticks for the same
    contract must both be retained (TickType is a preserved dimension, not
    collapsed)."""
    from deephaven import new_table
    from deephaven.column import datetime_col, double_col, long_col, string_col
    from deephaven.pandas import to_pandas
    from deephaven.time import to_j_instant

    t0 = to_j_instant("2026-06-10T15:00:00 UTC")
    t1 = to_j_instant("2026-06-10T15:00:01 UTC")
    raw = new_table([
        datetime_col("ReceiveTime", [t0, t1]),
        long_col("ContractId", [12345, 12345]),
        string_col("Symbol", ["AAPL", "AAPL"]),
        string_col("TickType", ["LAST", "CLOSE"]),
        double_col("Price", [291.55, 289.00]),
    ])
    spec = CANONICAL_SPECS["price_tick"]
    viewed = apply_canonical_view(raw, spec, PRICE_TICK)
    df = to_pandas(viewed)
    assert len(df) == 2
    assert set(df["TickType"]) == {"LAST", "CLOSE"}


# ---------------------------------------------------------------------------
# news_provider spec — news_providers -> canonical news_provider
# ---------------------------------------------------------------------------

def _raw_news_providers_table() -> Any:
    from deephaven import new_table
    from deephaven.column import string_col

    return new_table([
        string_col("Code", ["BRFG"]),
        string_col("Name", ["Briefing.com"]),
    ])


def test_news_provider_view_conforms_to_schema() -> None:
    spec = CANONICAL_SPECS["news_provider"]
    viewed = apply_canonical_view(_raw_news_providers_table(), spec, NEWS_PROVIDER)
    port = ibda.connect({"news_provider": viewed})
    arrow = port.table("news_provider").snapshot()
    NEWS_PROVIDER.validate(arrow)
    assert arrow.num_rows == 1


def test_news_provider_view_values() -> None:
    spec = CANONICAL_SPECS["news_provider"]
    viewed = apply_canonical_view(_raw_news_providers_table(), spec, NEWS_PROVIDER)
    port = ibda.connect({"news_provider": viewed})
    row = port.table("news_provider").snapshot().to_pylist()[0]
    assert row["Code"] == "BRFG"
    assert row["Name"] == "Briefing.com"


def test_news_provider_dedupes_by_code() -> None:
    """A re-emitted row for the same Code must collapse to one row (last wins)."""
    from deephaven import new_table
    from deephaven.column import string_col
    from deephaven.pandas import to_pandas

    raw = new_table([
        string_col("Code", ["BRFG", "BRFG"]),
        string_col("Name", ["Briefing.com", "Briefing.com General Market Columns"]),
    ])
    spec = CANONICAL_SPECS["news_provider"]
    viewed = apply_canonical_view(raw, spec, NEWS_PROVIDER)
    df = to_pandas(viewed)
    assert len(df) == 1
    assert df.iloc[0]["Name"] == "Briefing.com General Market Columns"
