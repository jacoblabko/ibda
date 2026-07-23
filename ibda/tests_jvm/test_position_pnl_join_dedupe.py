"""JVM regression test: position_pnl's join must dedupe accounts_positions (a prior regression).

``position_pnl`` (``CANONICAL_SPECS["position_pnl"]``) joins ``accounts_pnl_single``
(left) to ``accounts_positions`` (right) on ``ConId=ContractId`` to resolve ``Sym``.
``accounts_positions`` is append-only and deephaven-ib issues TWO overlapping
``reqPositionsMulti`` subscriptions on connect (an "All" sweep plus a per-account
repeat from the ``managedAccounts`` callback) — see the "position" spec's
``dedupe_keys`` comment in ``ibda/adapters/ibkr/specs.py``. So the SAME raw table,
when joined here as the RIGHT side of ``position_pnl``'s join, can carry more than
one row for a given ``ContractId``. Deephaven's ``natural_join`` errors ("more than
one right row") whenever more than one right row matches a given left row.

This is the same shape of bug already fixed for the ``execution`` spec's commission
join, which added ``IbkrTableSpec.join_dedupe_raw_keys`` +
``apply_canonical_view``'s generic ``last_by`` pre-join reduction step. This test
proves ``position_pnl`` now uses that same mechanism (``join_dedupe_raw_keys=
("ContractId",)``) and — via the "would have caught it" test below — proves the
crash is real absent the fix.

Run with:
    uv run pytest ibda/tests_jvm/test_position_pnl_join_dedupe.py -q
"""
from __future__ import annotations

import dataclasses
from typing import Any

import pytest

import ibda
from ibda.adapters.deephaven.views import apply_canonical_view
from ibda.adapters.ibkr.specs import CANONICAL_SPECS
from ibda.schema import POSITION_PNL


# ---------------------------------------------------------------------------
# Raw table helpers
# ---------------------------------------------------------------------------


def _raw_accounts_pnl_single() -> Any:
    """accounts_pnl_single raw table (left side of the position_pnl join).

    ConId arrives as a String (parsed from the request Note) — see
    ``ibda.adapters.ibkr.pnl`` module docstring and the position_pnl spec's
    ``join_cast_left_col="ConId"``.
    """
    from deephaven import new_table
    from deephaven.column import double_col, string_col

    return new_table([
        string_col("Account", ["DU1"]),
        string_col("ConId", ["12345"]),
        double_col("DailyPnL", [10.0]),
        double_col("UnrealizedPnL", [200.0]),
        double_col("RealizedPnL", [5.0]),
        double_col("Value", [15000.0]),
    ])


def _raw_accounts_positions_duplicate_contract_id() -> Any:
    """accounts_positions with TWO rows for the SAME ContractId (12345).

    Simulates deephaven-ib's overlapping reqPositionsMulti sweep writing the
    same position twice at startup — the exact scenario that crashes
    natural_join absent the right-side dedupe. The two rows carry different
    Symbol values so the winner is unambiguous: last_by keeps the LAST row in
    table order ("AAPL-LATE"), not the first ("AAPL-EARLY").
    """
    from deephaven import new_table
    from deephaven.column import long_col, string_col

    return new_table([
        long_col("ContractId", [12345, 12345]),
        string_col("Account", ["DU1", "DU1"]),
        string_col("Symbol", ["AAPL-EARLY", "AAPL-LATE"]),
    ])


def _raw_accounts_positions_unique_contract_id() -> Any:
    """accounts_positions negative control: one row per ContractId (no duplicate)."""
    from deephaven import new_table
    from deephaven.column import long_col, string_col

    return new_table([
        long_col("ContractId", [12345]),
        string_col("Account", ["DU1"]),
        string_col("Symbol", ["AAPL"]),
    ])


# ---------------------------------------------------------------------------
# Pre-fix crash reproduction (proves the test would have caught the regression above)
# ---------------------------------------------------------------------------


def test_position_pnl_join_without_dedupe_crashes_on_duplicate_right_row() -> None:
    """Negative-proof: strip join_dedupe_raw_keys and confirm natural_join errors.

    This reconstructs the PRE-FIX spec (join_dedupe_raw_keys=None) via
    dataclasses.replace and feeds it the same duplicate-ContractId fixture used
    by the positive test below. It must raise — proving the duplicate-right-row
    fixture genuinely triggers deephaven's natural_join duplicate-right-key error
    (observed message: "Natural Join found duplicate right key for <key>"), and
    therefore that the fixed spec (with join_dedupe_raw_keys set) is the thing
    preventing the crash, not some incidental property of the fixture.
    """
    spec = CANONICAL_SPECS["position_pnl"]
    assert spec.join_dedupe_raw_keys == ("ContractId",)  # sanity: fix is in place
    pre_fix_spec = dataclasses.replace(spec, join_dedupe_raw_keys=None)

    with pytest.raises(Exception, match="(?i)duplicate right key"):
        apply_canonical_view(
            _raw_accounts_pnl_single(),
            pre_fix_spec,
            POSITION_PNL,
            join_raw=_raw_accounts_positions_duplicate_contract_id(),
        )


# ---------------------------------------------------------------------------
# Post-fix behavior
# ---------------------------------------------------------------------------


def test_position_pnl_spec_dedupes_accounts_positions_by_contract_id() -> None:
    """Fixed spec declares the right dedupe key for the right-side table."""
    spec = CANONICAL_SPECS["position_pnl"]
    assert spec.join_dedupe_raw_keys == ("ContractId",)


def test_position_pnl_view_builds_without_raising_on_duplicate_right_row() -> None:
    """The bug: BEFORE the fix, this fixture raised 'more than one right row'.

    AFTER the fix (join_dedupe_raw_keys=("ContractId",) on the spec),
    apply_canonical_view must build one row via last_by's dedupe of the RIGHT
    (accounts_positions) table before the join — no raise.
    """
    spec = CANONICAL_SPECS["position_pnl"]
    viewed = apply_canonical_view(
        _raw_accounts_pnl_single(),
        spec,
        POSITION_PNL,
        join_raw=_raw_accounts_positions_duplicate_contract_id(),
    )
    port = ibda.connect({"position_pnl": viewed})
    arrow = port.table("position_pnl").snapshot()
    POSITION_PNL.validate(arrow)
    assert arrow.num_rows == 1


def test_position_pnl_view_duplicate_right_row_resolves_to_last_by_winner() -> None:
    """The surviving joined Symbol must come from the LAST (winning) right row."""
    spec = CANONICAL_SPECS["position_pnl"]
    viewed = apply_canonical_view(
        _raw_accounts_pnl_single(),
        spec,
        POSITION_PNL,
        join_raw=_raw_accounts_positions_duplicate_contract_id(),
    )
    port = ibda.connect({"position_pnl": viewed})
    row = port.table("position_pnl").snapshot().to_pylist()[0]
    assert row["ConId"] == 12345
    assert row["Sym"] == "AAPL-LATE"
    assert row["Account"] == "DU1"
    assert row["DailyPnl"] == pytest.approx(10.0)
    assert row["UnrealizedPnl"] == pytest.approx(200.0)
    assert row["RealizedPnl"] == pytest.approx(5.0)
    assert row["MarketValue"] == pytest.approx(15000.0)


def test_position_pnl_view_unique_right_row_negative_control() -> None:
    """Negative control: a unique-ContractId fixture builds identically (Sym resolved)."""
    spec = CANONICAL_SPECS["position_pnl"]
    viewed = apply_canonical_view(
        _raw_accounts_pnl_single(),
        spec,
        POSITION_PNL,
        join_raw=_raw_accounts_positions_unique_contract_id(),
    )
    port = ibda.connect({"position_pnl": viewed})
    arrow = port.table("position_pnl").snapshot()
    POSITION_PNL.validate(arrow)
    row = port.table("position_pnl").snapshot().to_pylist()[0]
    assert arrow.num_rows == 1
    assert row["Sym"] == "AAPL"
