"""Proof that reading positions THROUGH ibda's port (consumer -> ibda -> Arrow) produces
identical rows to reading the raw live table directly (the pre-ibda path).

``_positions_snapshot_via_port`` below stands in for an external consumer: it demonstrates
the engine-hidden surface by handing ibda a live table and getting Arrow back, never
touching deephaven directly itself. It is intentionally inlined here (rather than imported
from a separate example module) so this test carries no first-party dependency outside
this package.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pyarrow as pa

import ibda


def _positions_snapshot_via_port(tables: Mapping[str, Any]) -> pa.Table:
    """Return the canonical `position` table as Arrow, via the port (consumer-side proof)."""
    port = ibda.connect(tables)
    return port.table("position").snapshot()


def _positions_table() -> Any:
    from deephaven import new_table
    from deephaven.column import double_col, long_col, string_col

    return new_table([
        string_col("Account", ["DU1", "DU1"]),
        long_col("ConId", [1, 2]),
        string_col("Sym", ["AAPL", "MSFT"]),
        double_col("Qty", [100.0, -50.0]),
        double_col("AvgCost", [150.0, 300.0]),
        double_col("MarketPrice", [155.0, 295.0]),
        double_col("MarketValue", [15500.0, -14750.0]),
        double_col("UnrealizedPnl", [500.0, 250.0]),
    ])


def test_port_path_matches_direct_path() -> None:
    from deephaven.pandas import to_pandas

    positions = _positions_table()
    tables = {"position": positions}

    via_port = _positions_snapshot_via_port(tables)              # consumer -> ibda -> Arrow
    direct = pa.Table.from_pandas(to_pandas(positions), preserve_index=False)  # current path

    assert via_port.num_rows == direct.num_rows == 2
    assert via_port.to_pylist() == direct.to_pylist()
