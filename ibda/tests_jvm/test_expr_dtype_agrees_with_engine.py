"""JVM contract test: static dtype inference vs. the engine's own post-hoc recompute.

`ibda.analytics.expr`'s `validate_expression` now runs a static dtype-inference
pass over a `derive` expression BEFORE any engine call (see its module
docstring). `DeephavenPort.derive` / `register_derived` separately recompute
the derived column's dtype AFTER the engine has actually run the formula
(`_schema_from_dh_handle`, reading `meta_table`). These two are independent
code paths reaching a conclusion about the same thing — this suite is the
guard against them silently diverging.

For each representative expression: derive it, read the actual engine-derived
output dtype for the new column (via `Result.schema`, which
`DeephavenPort.derive` populates from `_schema_from_dh_handle`), and assert it
equals the dtype `validate_expression` inferred statically (`ValidatedExpr.dtype`)
for the identical expression against the identical column-dtype mapping.

Run with:
    uv run pytest ibda/tests_jvm/test_expr_dtype_agrees_with_engine.py -v
"""

from __future__ import annotations

from typing import Any

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
        double_col("MarketPrice", [155.0, 290.0]),
        double_col("MarketValue", [1550.0, -1450.0]),
        double_col("UnrealizedPnl", [50.0, -50.0]),
    ])


def _adapter() -> Any:
    from ibda.adapters.deephaven.adapter import DeephavenPort  # noqa: PLC0415

    return DeephavenPort({"position": _position_table()})


@pytest.mark.parametrize(
    ("expr_str", "out_col"),
    [
        ("Qty * 2", "Doubled"),  # float * int literal -> float
        ("ConId + 1", "NextConId"),  # int64 + int literal -> int64
        ("Qty + ConId", "Mixed"),  # float + int64 -> float (promotion)
        ("MarketValue / Qty", "PerShare"),  # float / float -> float
        ("sqrt(Qty * Qty)", "Magnitude"),  # sqrt(...) -> always float
        ("abs(ConId)", "AbsConId"),  # abs preserves int64
        ("abs(Qty)", "AbsQty"),  # abs preserves float
        ("max(Qty, AvgCost)", "MaxOf"),  # max(float, float) -> float
        ("Qty > 0", "IsLong"),  # comparison -> bool
        ("Qty > 0 and AvgCost > 0", "BothPositive"),  # bool op -> bool
    ],
)
def test_static_inference_agrees_with_engine_recompute(expr_str: str, out_col: str) -> None:
    from ibda.analytics.expr import validate_expression  # noqa: PLC0415
    from ibda.schema import DType  # noqa: PLC0415

    port = _adapter()
    result = port.table("position")

    column_dtypes: dict[str, DType] = {c.name: c.dtype for c in result.schema.columns}
    statically_inferred = validate_expression(expr_str, columns=column_dtypes).dtype
    assert statically_inferred is not None, (
        f"expected a definite static dtype for {expr_str!r}; got None "
        "(a conservative fallback triggered where this test expected full inference)"
    )

    derived = result.derive(**{out_col: expr_str})
    engine_col = next(c for c in derived.schema.columns if c.name == out_col)

    assert engine_col.dtype == statically_inferred, (
        f"static/engine dtype divergence for {expr_str!r}: "
        f"validate_expression inferred {statically_inferred}, but the engine's "
        f"post-hoc recompute (_schema_from_dh_handle) produced {engine_col.dtype}"
    )
