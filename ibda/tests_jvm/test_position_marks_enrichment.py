"""JVM contract test: enrich_position_with_marks joins live marks from accounts_portfolio.

NOTE: This test requires the Deephaven JVM (``deephaven`` package on the path).
Run with:
    uv run pytest ibda/tests_jvm/test_position_marks_enrichment.py -q

Correctness contract being tested:
- For a ConId that appears in both the position view and accounts_portfolio, the
  enriched view must carry non-null MarketPrice, MarketValue, and UnrealizedPnl.
- For a ConId present in positions but absent from accounts_portfolio, the enriched
  view must carry null marks (natural_join left semantics).
- An empty accounts_portfolio must never be passed to the function (the caller —
  ``_enrich_position_marks`` — guards this); the function itself is not tested with
  an empty portfolio here (that's the pure wiring test's concern).
- The column order in the returned table matches the POSITION schema order.
- The returned table schema-validates with ``POSITION.validate(arrow)``.

NOTE on reasoning: this join is LEFT (all position rows survive). Deephaven's
``natural_join`` has left-join semantics: rows in the left table with no match on
the right get null for right-side columns.  The marks view is right; the position
view is left.  This is the correct behaviour for IBKR data where some positions
may temporarily have no portfolio entry (e.g. just-opened positions).
"""
from __future__ import annotations

from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Raw table helpers
# ---------------------------------------------------------------------------


def _raw_position_view() -> Any:
    """Canonical position view as built by apply_canonical_view from accounts_positions.

    Mirrors the schema produced after Step 3 (view projection) + Step 4 (last_by
    dedupe) in apply_canonical_view, including null-filled mark columns.
    """
    from deephaven import new_table
    from deephaven.column import double_col, long_col, string_col

    return new_table([
        string_col("Account",      ["DU1", "DU1", "DU1"]),
        long_col("ConId",          [12345, 67890, 99999]),
        string_col("Sym",          ["AAPL", "MSFT", "ORPHAN"]),
        string_col("SecType",      ["STK", "STK", "STK"]),
        double_col("Qty",          [100.0, 50.0, 10.0]),
        double_col("AvgCost",      [150.0, 300.0, 50.0]),
        # Option-identity columns (null for these plain STK positions) — added
        # to POSITION so they precede the mark columns (which must stay last;
        # see enrich_position_with_marks's docstring).
        string_col("Right",        [None, None, None]),
        double_col("Strike",       [None, None, None]),
        string_col("LastTradeDateOrContractMonth", [None, None, None]),
        string_col("LocalSymbol",  ["AAPL", "MSFT", "ORPHAN"]),
        double_col("Multiplier",   [None, None, None]),
        string_col("Currency",     ["USD", "USD", "USD"]),
        string_col("Exchange",     ["SMART", "SMART", "SMART"]),
        # Null-filled marks (accounts_positions does not carry valuation).
        # After enrich_position_with_marks these are replaced for matching ConIds.
        double_col("MarketPrice",  [float("nan"), float("nan"), float("nan")]),
        double_col("MarketValue",  [float("nan"), float("nan"), float("nan")]),
        double_col("UnrealizedPnl", [float("nan"), float("nan"), float("nan")]),
    ]).update_view([
        # Convert nan to Deephaven NULL_DOUBLE so the join target is a proper null.
        "MarketPrice   = isNull(MarketPrice)   || Double.isNaN(MarketPrice)   ? NULL_DOUBLE : MarketPrice",
        "MarketValue   = isNull(MarketValue)   || Double.isNaN(MarketValue)   ? NULL_DOUBLE : MarketValue",
        "UnrealizedPnl = isNull(UnrealizedPnl) || Double.isNaN(UnrealizedPnl) ? NULL_DOUBLE : UnrealizedPnl",
    ])


def _raw_accounts_portfolio() -> Any:
    """accounts_portfolio raw table (the deephaven-ib source table).

    Contains marks for AAPL (ConId=12345) and MSFT (ConId=67890), but NOT for
    ORPHAN (ConId=99999) — so the join must leave ORPHAN's marks null.
    Two rows for AAPL to verify last_by dedup picks the later row.
    """
    from deephaven import new_table
    from deephaven.column import double_col, long_col, string_col

    return new_table([
        # ContractId (raw name before canonical rename)
        long_col("ContractId",   [12345,   12345,   67890]),
        # Extra columns that exist in accounts_portfolio (not all are needed for marks).
        string_col("Account",    ["DU1",   "DU1",   "DU1"]),
        string_col("Symbol",     ["AAPL",  "AAPL",  "MSFT"]),
        # Earlier AAPL row (should be superseded by last_by).
        double_col("MarketPrice",   [148.0,  152.0,  320.0]),
        double_col("MarketValue",   [14800.0, 15200.0, 16000.0]),
        # Raw name is UnrealizedPnL (capital L) — must be renamed to canonical UnrealizedPnl.
        double_col("UnrealizedPnL", [200.0, 400.0, 1000.0]),
    ])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_enrich_populates_marks_for_matching_conids() -> None:
    """ConIds that appear in accounts_portfolio must have non-null marks after enrich."""
    from deephaven.pandas import to_pandas

    from ibda.adapters.deephaven.views import enrich_position_with_marks

    position_view = _raw_position_view()
    portfolio_raw = _raw_accounts_portfolio()

    enriched = enrich_position_with_marks(position_view, portfolio_raw)
    df = to_pandas(enriched).sort_values("ConId").reset_index(drop=True)

    # AAPL (12345): must have marks from the LAST portfolio row (last_by dedup).
    aapl = df[df["ConId"] == 12345].iloc[0]
    assert aapl["MarketPrice"] == pytest.approx(152.0), (
        f"AAPL MarketPrice: expected 152.0 (last_by row), got {aapl['MarketPrice']}"
    )
    assert aapl["MarketValue"] == pytest.approx(15200.0), (
        f"AAPL MarketValue: expected 15200.0, got {aapl['MarketValue']}"
    )
    assert aapl["UnrealizedPnl"] == pytest.approx(400.0), (
        f"AAPL UnrealizedPnl (from UnrealizedPnL rename): expected 400.0, got {aapl['UnrealizedPnl']}"
    )

    # MSFT (67890): must have marks from portfolio.
    msft = df[df["ConId"] == 67890].iloc[0]
    assert msft["MarketPrice"] == pytest.approx(320.0)
    assert msft["MarketValue"] == pytest.approx(16000.0)
    assert msft["UnrealizedPnl"] == pytest.approx(1000.0)


def test_enrich_leaves_null_marks_for_non_matching_conid() -> None:
    """ORPHAN (ConId=99999) has no portfolio row — marks must remain null (left-join)."""
    import pandas as pd

    from deephaven.pandas import to_pandas

    from ibda.adapters.deephaven.views import enrich_position_with_marks

    enriched = enrich_position_with_marks(_raw_position_view(), _raw_accounts_portfolio())
    df = to_pandas(enriched)

    orphan = df[df["ConId"] == 99999].iloc[0]
    # Deephaven NULL_DOUBLE may materialise as NaN, None, or pd.NA after to_pandas;
    # pd.isna() accepts all three forms without false-negatives.
    assert pd.isna(orphan["MarketPrice"]), (
        f"ORPHAN MarketPrice should be null; got {orphan['MarketPrice']}"
    )
    assert pd.isna(orphan["MarketValue"]), (
        f"ORPHAN MarketValue should be null; got {orphan['MarketValue']}"
    )
    assert pd.isna(orphan["UnrealizedPnl"]), (
        f"ORPHAN UnrealizedPnl should be null; got {orphan['UnrealizedPnl']}"
    )


def test_enrich_preserves_all_position_rows() -> None:
    """All position rows must survive the left join — no rows dropped."""
    from deephaven.pandas import to_pandas

    from ibda.adapters.deephaven.views import enrich_position_with_marks

    enriched = enrich_position_with_marks(_raw_position_view(), _raw_accounts_portfolio())
    df = to_pandas(enriched)

    assert len(df) == 3, (
        f"Expected 3 position rows after enrichment (left-join semantics); got {len(df)}"
    )
    assert set(df["ConId"].tolist()) == {12345, 67890, 99999}


def test_enrich_schema_validates_against_position_schema() -> None:
    """The enriched table must pass POSITION.validate() after the join."""
    import ibda
    from ibda.adapters.deephaven.views import enrich_position_with_marks
    from ibda.schema import POSITION

    enriched = enrich_position_with_marks(_raw_position_view(), _raw_accounts_portfolio())
    # Route through the port for full schema validation.
    port = ibda.connect({"position": enriched})
    arrow = port.table("position").snapshot()
    POSITION.validate(arrow)
    assert arrow.num_rows == 3


def test_enrich_column_order_matches_position_schema() -> None:
    """Column order of the enriched table must match the POSITION schema definition."""
    from deephaven.pandas import to_pandas

    from ibda.adapters.deephaven.views import enrich_position_with_marks
    from ibda.schema import POSITION

    enriched = enrich_position_with_marks(_raw_position_view(), _raw_accounts_portfolio())
    df = to_pandas(enriched)

    expected_cols = list(POSITION.column_names)
    actual_cols = list(df.columns)
    assert actual_cols == expected_cols, (
        f"Column order mismatch.\nExpected: {expected_cols}\nActual:   {actual_cols}"
    )


def _two_account_position_view() -> Any:
    """The same contract held in two accounts, each with its own quantity."""
    from deephaven import new_table
    from deephaven.column import double_col, long_col, string_col

    return new_table([
        string_col("Account",      ["DU1", "DU2"]),
        long_col("ConId",          [12345, 12345]),
        string_col("Sym",          ["AAPL", "AAPL"]),
        string_col("SecType",      ["STK", "STK"]),
        double_col("Qty",          [100.0, 700.0]),
        double_col("AvgCost",      [150.0, 150.0]),
        string_col("Right",        [None, None]),
        double_col("Strike",       [None, None]),
        string_col("LastTradeDateOrContractMonth", [None, None]),
        string_col("LocalSymbol",  ["AAPL", "AAPL"]),
        double_col("Multiplier",   [None, None]),
        string_col("Currency",     ["USD", "USD"]),
        string_col("Exchange",     ["SMART", "SMART"]),
        double_col("MarketPrice",  [float("nan"), float("nan")]),
        double_col("MarketValue",  [float("nan"), float("nan")]),
        double_col("UnrealizedPnl", [float("nan"), float("nan")]),
    ]).update_view([
        "MarketPrice   = isNull(MarketPrice)   || Double.isNaN(MarketPrice)   ? NULL_DOUBLE : MarketPrice",
        "MarketValue   = isNull(MarketValue)   || Double.isNaN(MarketValue)   ? NULL_DOUBLE : MarketValue",
        "UnrealizedPnl = isNull(UnrealizedPnl) || Double.isNaN(UnrealizedPnl) ? NULL_DOUBLE : UnrealizedPnl",
    ])


def _two_account_portfolio() -> Any:
    """One mark row per account for the same contract, with different valuations."""
    from deephaven import new_table
    from deephaven.column import double_col, long_col, string_col

    return new_table([
        long_col("ContractId",      [12345,   12345],),
        string_col("Account",       ["DU1",   "DU2"]),
        string_col("Symbol",        ["AAPL",  "AAPL"]),
        double_col("MarketPrice",   [150.0,   150.0]),
        double_col("MarketValue",   [15000.0, 105000.0]),
        double_col("UnrealizedPnL", [0.0,     500.0]),
    ])


def test_a_contract_held_in_two_accounts_keeps_each_account_s_own_mark() -> None:
    """Account is part of the dedupe and join key, not just ContractId.

    `accounts_portfolio` emits one mark row per (account, contract). Keying the dedupe
    on the contract alone keeps only the last of them, and every other account's
    position then joins to a mark belonging to a DIFFERENT account — MarketValue
    attributed to the wrong book, silently and with no null to notice.

    Here DU1 holds 100 shares (MarketValue 15,000) and DU2 holds 700 (105,000). Under
    a contract-only key both rows take whichever mark survived, and the book is out by
    90,000 on one of them.
    """
    from ibda.adapters.deephaven.views import enrich_position_with_marks

    enriched = enrich_position_with_marks(
        _two_account_position_view(), _two_account_portfolio()
    )
    import deephaven.pandas as dhpd

    df = dhpd.to_pandas(enriched).set_index("Account")
    assert float(df.loc["DU1", "MarketValue"]) == 15000.0, (
        f"DU1 took another account's mark: {df.loc['DU1', 'MarketValue']}"
    )
    assert float(df.loc["DU2", "MarketValue"]) == 105000.0, (
        f"DU2 took another account's mark: {df.loc['DU2', 'MarketValue']}"
    )
    assert float(df.loc["DU2", "UnrealizedPnl"]) == 500.0
