"""JVM test: DeephavenPort.live_table() returns a genuine, usable live Table.

The pure-Python test (ibda/tests/test_live_table_accessor.py) covers the
resolution logic with a plain sentinel object. This test proves the
production contract: the object returned by live_table() is a real deephaven
Table that ticks and supports Deephaven table operations directly (not just
an opaque handle) — the whole point of the in-engine escape hatch.

Uses ``DeephavenPort`` directly (not ``ibda.connect``) — same pattern as
test_register_derived.py — because ``ibda.connect`` returns the abstract
``DataPort`` (which does not, and must not, declare ``live_table``); calling
an adapter-only method needs the concrete type.

Run with:
    uv run pytest ibda/tests_jvm/test_live_table_accessor_jvm.py -q
"""
from __future__ import annotations

from typing import Any

import pytest

from ibda.errors import UnknownTable


def _raw_quote_table() -> Any:
    from deephaven import new_table
    from deephaven.column import datetime_col, double_col, string_col
    from deephaven.time import to_j_instant

    t0 = to_j_instant("2026-06-10T15:00:00 UTC")
    return new_table([
        string_col("Symbol", ["AAPL"]),
        datetime_col("Timestamp", [t0]),
        double_col("BidPrice", [154.9]),
        double_col("AskPrice", [155.1]),
        double_col("BidSize", [10.0]),
        double_col("AskSize", [12.0]),
    ])


def _quote_port() -> Any:
    from ibda.adapters.deephaven.adapter import DeephavenPort
    from ibda.adapters.deephaven.views import apply_canonical_view
    from ibda.adapters.ibkr.specs import CANONICAL_SPECS
    from ibda.schema import QUOTE

    spec = CANONICAL_SPECS["quote"]
    viewed = apply_canonical_view(_raw_quote_table(), spec, QUOTE)
    return DeephavenPort({"quote": viewed})


def test_live_table_returns_a_real_deephaven_table() -> None:
    port = _quote_port()

    live = port.live_table("quote")

    # A genuine deephaven Table exposes .size; a stray Arrow table or plain
    # object would not.
    assert live.size == 1


def test_live_table_and_table_dot_handle_are_the_same_object() -> None:
    port = _quote_port()

    assert port.live_table("quote") is port.table("quote").handle


def test_live_table_raises_unknown_table_on_bad_name() -> None:
    port = _quote_port()

    with pytest.raises(UnknownTable):
        port.live_table("not_a_real_table")


def test_live_table_resolves_a_registered_derived_table() -> None:
    """live_table must resolve derived (register_derived) names too, per the ADR."""
    port = _quote_port()

    base = port.table("quote")
    port.register_derived("quote_mid", base=base, columns={"Mid": "(Bid + Ask) / 2"})

    live = port.live_table("quote_mid")
    assert live.size == 1
