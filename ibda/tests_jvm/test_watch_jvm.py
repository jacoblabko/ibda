"""JVM integration smoke — watch() fires callbacks on a real ticking table.

Verifies that when rows matching a predicate are appended to a deephaven
DynamicTableWriter-backed ticking table, the watch() callback is invoked with
ONLY the matching rows (predicate filtering works end-to-end through the
Deephaven update graph).

Run with:
    uv run pytest ibda/tests_jvm/test_watch_jvm.py -q
"""

from __future__ import annotations

import threading
import time

import pyarrow as pa

import ibda
from ibda.watch import watch


def test_watch_fires_on_matching_rows() -> None:
    from deephaven import DynamicTableWriter, dtypes
    from deephaven.time import to_j_instant

    # Create a quote-shaped ticking table via DynamicTableWriter.
    # Using a subset of QUOTE columns — Sym, Timestamp, Last — which is enough
    # for the predicate "Last > 150".  The filter operates on the deephaven table
    # before it ever reaches our callback, so we don't need full QUOTE conformance
    # here; we just need the predicate column to exist.
    dtw = DynamicTableWriter(
        {
            "Sym": dtypes.string,
            "Timestamp": dtypes.Instant,
            "Last": dtypes.double,
        }
    )

    port = ibda.connect({"quote": dtw.table})

    received: list[pa.Table] = []
    lock = threading.Lock()

    def cb(tbl: pa.Table) -> None:
        with lock:
            received.append(tbl)

    # Watch only rows where Last > 150.
    watch(port.table("quote"), cb, predicate="Last > 150")

    t0 = to_j_instant("2026-06-10T15:00:00 UTC")
    # Write one matching row (NVDA, Last=155) and one non-matching row (AAPL, Last=140).
    dtw.write_row("NVDA", t0, 155.0)
    dtw.write_row("AAPL", t0, 140.0)

    # The deephaven update graph cycles asynchronously; poll for up to 20 s.
    deadline = time.time() + 20.0
    while time.time() < deadline:
        with lock:
            if received:
                break
        time.sleep(0.2)

    with lock:
        assert received, "watch callback never fired within 20 s"
        rows = [r for t in received for r in t.to_pylist()]

    syms = {r["Sym"] for r in rows}
    assert "NVDA" in syms, f"matching row (NVDA) not delivered; syms seen: {syms}"
    assert "AAPL" not in syms, f"non-matching row (AAPL) leaked through predicate; syms seen: {syms}"
