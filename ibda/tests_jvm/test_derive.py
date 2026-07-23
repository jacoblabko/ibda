"""JVM contract test: DeephavenPort.derive — safe-DSL server-side column derivation.

Stands up a small in-process Deephaven engine, derives new columns via the
safe-DSL (`ibda.analytics.expr`), and asserts the values are correct end to
end (parse -> validate -> compile -> `Table.update([...])` -> snapshot).
Engine correctness is JVM-verified here, not mypy-verified (deephaven is
treated as untyped in mypy config).

Run with:
    uv run pytest ibda/tests_jvm/test_derive.py -v
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest


def _position_table() -> Any:
    from deephaven import new_table  # noqa: PLC0415 — JVM-gated
    from deephaven.column import double_col, long_col, string_col  # noqa: PLC0415

    return new_table([
        string_col("Account", ["DU1", "DU1"]),
        long_col("ConId", [1, 2]),
        string_col("Sym", ["AAPL", "MSFT"]),
        string_col("SecType", ["STK", "STK"]),
        double_col("Qty", [10.0, -5.0]),
        double_col("AvgCost", [150.0, 300.0]),
        string_col("Right", [None, None]),
        double_col("Strike", [None, None]),
        string_col("LastTradeDateOrContractMonth", [None, None]),
        string_col("LocalSymbol", ["AAPL", "MSFT"]),
        double_col("Multiplier", [None, None]),
        string_col("Currency", ["USD", "USD"]),
        string_col("Exchange", ["SMART", "SMART"]),
        double_col("MarketPrice", [155.0, 290.0]),
        double_col("MarketValue", [1550.0, -1450.0]),
        double_col("UnrealizedPnl", [50.0, -50.0]),
    ])


def _adapter() -> Any:
    from ibda.adapters.deephaven.adapter import DeephavenPort  # noqa: PLC0415

    return DeephavenPort({"position": _position_table()})


def test_derive_adds_a_computed_column_with_correct_values() -> None:
    port = _adapter()
    result = port.table("position")
    derived = result.derive(Doubled="Qty * 2")

    snapshot: pa.Table = derived.snapshot()
    assert "Doubled" in snapshot.schema.names
    assert snapshot.column("Doubled").to_pylist() == pytest.approx([20.0, -10.0])
    # Original columns are still present — derive only ADDS columns.
    assert set(result.schema.column_names) <= set(derived.schema.column_names)


def test_derive_schema_matches_actual_output_columns() -> None:
    port = _adapter()
    result = port.table("position")
    derived = result.derive(Doubled="Qty * 2")
    snapshot: pa.Table = derived.snapshot()

    actual_columns = tuple(snapshot.schema.names)
    assert derived.schema.column_names == actual_columns
    derived.schema.validate(snapshot)


def test_derive_supports_division_and_sqrt() -> None:
    port = _adapter()
    result = port.table("position")
    derived = result.derive(
        PerShareCost="MarketValue / Qty",
        Magnitude="sqrt(Qty * Qty)",
    )
    snapshot: pa.Table = derived.snapshot()

    assert snapshot.column("PerShareCost").to_pylist() == pytest.approx([155.0, 290.0])
    assert snapshot.column("Magnitude").to_pylist() == pytest.approx([10.0, 5.0])


def test_derive_multiple_columns_in_one_call() -> None:
    port = _adapter()
    result = port.table("position")
    derived = result.derive(A="Qty + 1", B="Qty - 1")
    snapshot: pa.Table = derived.snapshot()

    assert snapshot.column("A").to_pylist() == pytest.approx([11.0, -4.0])
    assert snapshot.column("B").to_pylist() == pytest.approx([9.0, -6.0])


def test_derive_rejects_malicious_expression_before_touching_the_engine() -> None:
    """A malicious expression raises ValueError from the safe-DSL gate — the
    engine's `.update()` is never reached (the whole point of the ADR)."""
    port = _adapter()
    result = port.table("position")

    with pytest.raises(ValueError, match="non-whitelisted function"):
        result.derive(Malicious="__import__('os').system('id')")


def test_derive_rejects_unknown_column() -> None:
    port = _adapter()
    result = port.table("position")

    with pytest.raises(ValueError, match="unknown or disallowed name"):
        result.derive(Bad="NotAColumn * 2")


def test_derive_type_invalid_expression_raises_valueerror_not_raw_engine_error() -> None:
    """Fix #3: `validate_expression` only checks grammar (AST allowlist) + that
    every Name is a known column — it does NOT check dtypes. `sqrt(Sym)` is
    grammar-valid (Sym is a known column, sqrt is whitelisted) but type-invalid
    (Sym is a String column; Math.sqrt takes a double) — it must be rejected as
    a ValueError once `.update()` compiles it, never as a raw `deephaven.DHError`.
    """
    from deephaven.dherror import DHError  # noqa: PLC0415 — JVM-gated

    port = _adapter()
    result = port.table("position")

    with pytest.raises(ValueError) as exc_info:
        result.derive(Bad="sqrt(Sym)")

    assert not isinstance(exc_info.value, DHError), (
        "a raw deephaven.DHError leaked through derive() instead of ValueError"
    )
    assert "sqrt(Sym)" in str(exc_info.value)
