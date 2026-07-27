"""JVM regression test: apply_canonical_view scrubs IBKR's Double.MAX_VALUE
PnL sentinel to NULL, on both ``position_pnl`` and ``account_pnl``.

The live deephaven-view path (``apply_canonical_view``) previously performed no
sentinel handling on PnL columns, unlike the Arrow snapshot path
(``ibda.adapters.ibkr.pnl._f``, which already scrubs
``1.7976931348623157e+308`` to ``None``). A ``sum(RealizedPnl)`` over the live
book would silently add ~1.8e308 per not-yet-computed component. This test
proves the fix (``_PNL_SENTINEL_SCRUB_COLS`` in
``ibda.adapters.deephaven.views``) turns the sentinel into NULL in the
canonical output, while a normal finite value passes through unchanged.

Run with:
    uv run pytest ibda/tests_jvm/test_pnl_view_sentinel_scrub.py -q
"""
from __future__ import annotations

from typing import Any

import pytest

import ibda
from ibda.adapters.deephaven.views import apply_canonical_view
from ibda.adapters.ibkr.specs import CANONICAL_SPECS
from ibda.schema import ACCOUNT_PNL, POSITION_PNL

# IBKR's "not yet available" sentinel — Java's Double.MAX_VALUE.
_SENTINEL = 1.7976931348623157e308


# ---------------------------------------------------------------------------
# Raw table helpers
# ---------------------------------------------------------------------------


def _raw_accounts_pnl_single_sentinel_and_finite() -> Any:
    """accounts_pnl_single (position_pnl's raw source): one sentinel row, one finite row."""
    from deephaven import new_table
    from deephaven.column import double_col, string_col

    return new_table([
        string_col("Account", ["DU1", "DU1"]),
        string_col("ConId", ["12345", "67890"]),
        double_col("DailyPnL", [_SENTINEL, 10.0]),
        double_col("UnrealizedPnL", [_SENTINEL, 200.0]),
        double_col("RealizedPnL", [_SENTINEL, 5.0]),
        double_col("Value", [_SENTINEL, 15000.0]),
    ])


def _raw_accounts_positions_for_pnl_join() -> Any:
    """accounts_positions (position_pnl's join table) — one row per ConId."""
    from deephaven import new_table
    from deephaven.column import long_col, string_col

    return new_table([
        long_col("ContractId", [12345, 67890]),
        string_col("Account", ["DU1", "DU1"]),
        string_col("Symbol", ["AAPL", "MSFT"]),
    ])


def _raw_accounts_pnl_sentinel_and_finite() -> Any:
    """accounts_pnl (account_pnl's raw source): one sentinel row, one finite row."""
    from deephaven import new_table
    from deephaven.column import datetime_col, double_col, string_col
    from deephaven.time import to_j_instant

    t0 = to_j_instant("2026-06-10T15:00:00 UTC")
    t1 = to_j_instant("2026-06-10T15:01:00 UTC")
    return new_table([
        string_col("Account", ["DU1", "DU2"]),
        datetime_col("ReceiveTime", [t0, t1]),
        double_col("DailyPnl", [_SENTINEL, 20.0]),
        double_col("UnrealizedPnl", [_SENTINEL, 300.0]),
        double_col("RealizedPnl", [_SENTINEL, 7.5]),
    ])


# ---------------------------------------------------------------------------
# position_pnl
# ---------------------------------------------------------------------------


def test_position_pnl_view_scrubs_sentinel_to_null() -> None:
    spec = CANONICAL_SPECS["position_pnl"]
    viewed = apply_canonical_view(
        _raw_accounts_pnl_single_sentinel_and_finite(),
        spec,
        POSITION_PNL,
        join_raw=_raw_accounts_positions_for_pnl_join(),
    )
    port = ibda.connect({"position_pnl": viewed})
    arrow = port.table("position_pnl").snapshot()
    POSITION_PNL.validate(arrow)

    rows = {row["ConId"]: row for row in arrow.to_pylist()}

    sentinel_row = rows[12345]
    assert sentinel_row["DailyPnl"] is None
    assert sentinel_row["UnrealizedPnl"] is None
    assert sentinel_row["RealizedPnl"] is None
    assert sentinel_row["MarketValue"] is None


def test_position_pnl_view_finite_value_passes_through_unchanged() -> None:
    spec = CANONICAL_SPECS["position_pnl"]
    viewed = apply_canonical_view(
        _raw_accounts_pnl_single_sentinel_and_finite(),
        spec,
        POSITION_PNL,
        join_raw=_raw_accounts_positions_for_pnl_join(),
    )
    port = ibda.connect({"position_pnl": viewed})
    arrow = port.table("position_pnl").snapshot()

    rows = {row["ConId"]: row for row in arrow.to_pylist()}

    finite_row = rows[67890]
    assert finite_row["DailyPnl"] == pytest.approx(10.0)
    assert finite_row["UnrealizedPnl"] == pytest.approx(200.0)
    assert finite_row["RealizedPnl"] == pytest.approx(5.0)
    assert finite_row["MarketValue"] == pytest.approx(15000.0)


# ---------------------------------------------------------------------------
# account_pnl
# ---------------------------------------------------------------------------


def test_account_pnl_view_scrubs_sentinel_to_null() -> None:
    spec = CANONICAL_SPECS["account_pnl"]
    viewed = apply_canonical_view(_raw_accounts_pnl_sentinel_and_finite(), spec, ACCOUNT_PNL)
    port = ibda.connect({"account_pnl": viewed})
    arrow = port.table("account_pnl").snapshot()
    ACCOUNT_PNL.validate(arrow)

    rows = {row["Account"]: row for row in arrow.to_pylist()}

    sentinel_row = rows["DU1"]
    assert sentinel_row["DailyPnl"] is None
    assert sentinel_row["UnrealizedPnl"] is None
    assert sentinel_row["RealizedPnl"] is None


def test_account_pnl_view_finite_value_passes_through_unchanged() -> None:
    spec = CANONICAL_SPECS["account_pnl"]
    viewed = apply_canonical_view(_raw_accounts_pnl_sentinel_and_finite(), spec, ACCOUNT_PNL)
    port = ibda.connect({"account_pnl": viewed})
    arrow = port.table("account_pnl").snapshot()

    rows = {row["Account"]: row for row in arrow.to_pylist()}

    finite_row = rows["DU2"]
    assert finite_row["DailyPnl"] == pytest.approx(20.0)
    assert finite_row["UnrealizedPnl"] == pytest.approx(300.0)
    assert finite_row["RealizedPnl"] == pytest.approx(7.5)
