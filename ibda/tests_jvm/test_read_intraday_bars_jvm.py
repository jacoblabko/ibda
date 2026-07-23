"""JVM test: read_intraday_bars — engine-side WHERE reduces materialized rows.

Proves two things against a real Deephaven server:
1. **Equivalence** — the engine-filtered result matches a full-snapshot-then-Python-filter.
2. **Bounded materialization** — the WHERE predicate limits what crosses into Python;
   decoy rows (for untracked RequestIds) are excluded at the engine level.

Run with:
    uv run pytest ibda/tests_jvm/test_read_intraday_bars_jvm.py -q
"""
from __future__ import annotations

import types
from typing import Any

from ibda.adapters.ibkr.marketdata import read_intraday_bars
from ibda.adapters.ibkr.supervisor import IbkrSupervisor


# ---------------------------------------------------------------------------
# Table factory
# ---------------------------------------------------------------------------

def _bars_historical_table() -> Any:
    """Build a bars_historical-shaped Deephaven table with 4 rows across 3 RequestIds.

    Layout:
    - RequestId 1 (AAPL-proxy): 2 rows — closes 150.0, 151.0
    - RequestId 2 (MSFT-proxy): 1 row  — close  300.0
    - RequestId 99 (decoy, not tracked): 1 row — close 500.0

    The test tracks only RequestIds 1 and 2 so the decoy row is the signal
    that proves the engine filter excludes non-tracked rows.
    """
    from deephaven import new_table
    from deephaven.column import datetime_col, double_col, long_col
    from deephaven.time import to_j_instant

    t0 = to_j_instant("2026-06-15T14:30:00 UTC")
    t1 = to_j_instant("2026-06-15T14:31:00 UTC")

    return new_table([
        long_col("RequestId",  [1,     1,     2,     99]),
        long_col("ContractId", [100,   100,   200,   999]),
        datetime_col("Timestamp", [t0, t1,   t0,    t0]),
        double_col("Open",  [150.0, 151.0, 300.0, 500.0]),
        double_col("High",  [150.0, 151.0, 300.0, 500.0]),
        double_col("Low",   [150.0, 151.0, 300.0, 500.0]),
        double_col("Close", [150.0, 151.0, 300.0, 500.0]),
        double_col("Volume", [100.0, 100.0, 200.0, 999.0]),
    ])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _supervisor(dh_table: Any) -> IbkrSupervisor:
    """Wrap *dh_table* in a supervisor using the real snapshot_rows_where path."""
    session = types.SimpleNamespace(tables={"bars_historical": dh_table})
    return IbkrSupervisor.from_session(session)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_read_intraday_bars_jvm_equivalence() -> None:
    """Engine-filtered result equals full-snapshot-then-Python-filter.

    Runs read_intraday_bars on a real Deephaven table (engine WHERE path) and
    compares against the naïve full-snapshot + Python filter to prove they
    produce identical output.
    """
    from ibda.adapters.deephaven.views import snapshot_rows

    dh_table = _bars_historical_table()
    sup = _supervisor(dh_table)

    # Engine-filtered path (the fix under test).
    out = read_intraday_bars(sup, {"AAPL": 1, "MSFT": 2})

    # Reference: full snapshot + Python filter (the old behaviour).
    all_rows = snapshot_rows(dh_table)
    aapl_ref = [r for r in all_rows if r.get("RequestId") == 1]
    msft_ref = [r for r in all_rows if r.get("RequestId") == 2]

    assert set(out) == {"AAPL", "MSFT"}

    # Row counts must match reference.
    assert out["AAPL"].num_rows == len(aapl_ref) == 2
    assert out["MSFT"].num_rows == len(msft_ref) == 1

    # Values must match reference (ordering preserved — append-only table).
    assert out["AAPL"].column("Close").to_pylist() == [150.0, 151.0]
    assert out["MSFT"].column("Close").to_pylist() == [300.0]

    # Symbol tag must be applied correctly.
    assert out["AAPL"].column("Sym").to_pylist() == ["AAPL", "AAPL"]
    assert out["MSFT"].column("Sym").to_pylist() == ["MSFT"]


def test_read_intraday_bars_jvm_bounded_materialization() -> None:
    """Engine WHERE limits materialized rows to only tracked RequestIds.

    Injects a counting wrapper around the real snapshot_rows_where to measure
    how many rows cross from the engine into Python.  Tracking RequestIds 1 and 2
    in a 4-row table (RequestId 99 is a decoy) must materialise exactly 3 rows —
    not 4 — proving the engine filter excludes the decoy at source.
    """
    from ibda.adapters.deephaven.views import snapshot_rows_where as _real_where

    dh_table = _bars_historical_table()
    materialized: list[int] = []

    def counting_where(raw_table: Any, predicate: str) -> list[dict[str, Any]]:
        result = _real_where(raw_table, predicate)
        materialized.append(len(result))
        return result

    session = types.SimpleNamespace(tables={"bars_historical": dh_table})
    sup = IbkrSupervisor.from_session(session, snapshot_rows_where_fn=counting_where)

    read_intraday_bars(sup, {"AAPL": 1, "MSFT": 2})

    # Exactly one combined snapshot call (not one per symbol).
    assert len(materialized) == 1, (
        f"expected 1 snapshot call; got {len(materialized)}"
    )

    # 3 matching rows (RequestId 1: 2 rows + RequestId 2: 1 row),
    # not 4 (the RequestId=99 decoy row must be excluded by the engine filter).
    assert materialized[0] == 3, (
        f"engine should materialise 3 rows (tracking 1+2), got {materialized[0]}"
    )


def test_read_intraday_bars_jvm_untracked_id_excluded() -> None:
    """A symbol not in request_ids (RequestId=99) must produce no output row.

    Verifies correctness at the symbol level: the caller only gets entries for
    the symbols it asked about.
    """
    dh_table = _bars_historical_table()
    sup = _supervisor(dh_table)

    out = read_intraday_bars(sup, {"AAPL": 1})

    assert "AAPL" in out
    assert out["AAPL"].num_rows == 2
    # RequestId=99 was in the table but not requested — must not leak into output.
    assert len(out) == 1


def test_read_intraday_bars_jvm_empty_returns_early() -> None:
    """Empty request_ids must return an empty dict with no snapshot issued."""
    from ibda.adapters.deephaven.views import snapshot_rows_where as _real_where

    dh_table = _bars_historical_table()
    snapshots: list[int] = []

    def counting_where(raw_table: Any, predicate: str) -> list[dict[str, Any]]:
        snapshots.append(1)
        return _real_where(raw_table, predicate)

    session = types.SimpleNamespace(tables={"bars_historical": dh_table})
    sup = IbkrSupervisor.from_session(session, snapshot_rows_where_fn=counting_where)

    out = read_intraday_bars(sup, {})

    assert out == {}
    assert len(snapshots) == 0, "no snapshot should be issued for empty request_ids"
