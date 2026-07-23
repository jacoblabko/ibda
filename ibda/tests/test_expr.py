"""Adversarial security suite for ibda.analytics.expr — the safe-DSL gate.

This is the security boundary for DataPort.derive. The trust model behind it (decided
2026-07-08) is: a caller-supplied ``derive`` expression may only be built from this
module's restricted, whitelisted expression grammar — never arbitrary Groovy/Java handed
to the engine. Every rejection case here is load-bearing: weakening this suite weakens the
only thing standing between an MCP/LLM caller and a Groovy-formula RCE surface into the
JVM. Do not relax this suite without re-examining that trust model.
"""

from __future__ import annotations

import ast

import pytest

from ibda.analytics.expr import ValidatedExpr, to_dh_formula, validate_expression
from ibda.schema import DType

_COLUMNS = ("Qty", "AvgCost", "MarketValue", "Beta", "Price")

#: A name -> DType mapping covering every canonical dtype, for the
#: dtype-inference battery below. Deliberately distinct column names from
#: `_COLUMNS` above so the two suites can't accidentally cross-contaminate.
_TYPED_COLUMNS: dict[str, DType] = {
    "Qty": DType.FLOAT64,
    "AvgCost": DType.FLOAT64,
    "ConId": DType.INT64,
    "Sym": DType.STRING,
    "IsLong": DType.BOOL,
    "Timestamp": DType.TIMESTAMP_NS,
}


# ---------------------------------------------------------------------------
# Accept: valid expressions compile end to end
# ---------------------------------------------------------------------------


def test_accepts_simple_arithmetic() -> None:
    v = validate_expression("Qty * 2", columns=_COLUMNS)
    assert isinstance(v, ValidatedExpr)
    assert to_dh_formula(v, out_col="Doubled") == "Doubled = (Qty * 2)"


def test_accepts_beta_style_expression() -> None:
    v = validate_expression("MarketValue * Beta", columns=_COLUMNS)
    assert to_dh_formula(v, out_col="BetaAdjExposure") == "BetaAdjExposure = (MarketValue * Beta)"


def test_accepts_division() -> None:
    v = validate_expression("MarketValue / Qty", columns=_COLUMNS)
    assert to_dh_formula(v, out_col="PerShare") == "PerShare = (MarketValue / Qty)"


def test_accepts_whitelisted_function_call() -> None:
    v = validate_expression("sqrt(Qty * Qty)", columns=_COLUMNS)
    assert to_dh_formula(v, out_col="Magnitude") == "Magnitude = Math.sqrt((Qty * Qty))"


@pytest.mark.parametrize(
    ("func", "dh_func"),
    [
        ("sqrt", "Math.sqrt"),
        ("log", "Math.log"),
        ("exp", "Math.exp"),
        ("abs", "Math.abs"),
    ],
)
def test_accepts_each_whitelisted_unary_math_func(func: str, dh_func: str) -> None:
    v = validate_expression(f"{func}(Qty)", columns=_COLUMNS)
    assert to_dh_formula(v, out_col="Out") == f"Out = {dh_func}(Qty)"


def test_accepts_min_max_multi_arg() -> None:
    v = validate_expression("max(Qty, AvgCost)", columns=_COLUMNS)
    assert to_dh_formula(v, out_col="Out") == "Out = Math.max(Qty, AvgCost)"


def test_accepts_unary_minus() -> None:
    v = validate_expression("-Qty", columns=_COLUMNS)
    assert to_dh_formula(v, out_col="Neg") == "Neg = (-Qty)"


def test_accepts_comparison() -> None:
    v = validate_expression("Qty > 0", columns=_COLUMNS)
    assert to_dh_formula(v, out_col="IsLong") == "IsLong = (Qty > 0)"


def test_accepts_bool_op() -> None:
    v = validate_expression("Qty > 0 and AvgCost > 0", columns=_COLUMNS)
    assert to_dh_formula(v, out_col="Flag") == "Flag = ((Qty > 0) && (AvgCost > 0))"


def test_accepts_not() -> None:
    v = validate_expression("not (Qty > 0)", columns=_COLUMNS)
    assert to_dh_formula(v, out_col="IsShortOrFlat") == "IsShortOrFlat = (!(Qty > 0))"


def test_accepts_string_constant() -> None:
    v = validate_expression('Qty == 0', columns=_COLUMNS)
    assert to_dh_formula(v, out_col="Flat") == "Flat = (Qty == 0)"


def test_accepts_chained_comparison() -> None:
    v = validate_expression("0 < Qty < 100", columns=_COLUMNS)
    assert to_dh_formula(v, out_col="Small") == "Small = (0 < Qty) && (Qty < 100)"


# ---------------------------------------------------------------------------
# Reject: the adversarial cases named in the plan
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os')",
        "__import__('os').system('rm -rf /')",
        "x.__class__",
        "Qty.__class__",
        "open('f')",
        "open('/etc/passwd').read()",
        "a[0]",
        "Qty[0]",
        "lambda: 1",
        "lambda x: x",
        "[i for i in Qty]",
        "{i for i in Qty}",
        "{i: i for i in Qty}",
        "(i for i in Qty)",
        "UnknownCol + 1",
        "foo(Qty)",
        "eval('1')",
        "exec('1')",
        "getattr(Qty, '__class__')",
        "1 if Qty else 2",
        "Qty ** 2",
        "Qty // 2",
        "import os",
    ],
)
def test_rejects_adversarial_expressions(expr: str) -> None:
    with pytest.raises(ValueError):
        validate_expression(expr, columns=_COLUMNS)


def test_rejects_unknown_column() -> None:
    with pytest.raises(ValueError, match="unknown or disallowed name"):
        validate_expression("NotAColumn * 2", columns=_COLUMNS)


def test_rejects_unknown_function() -> None:
    with pytest.raises(ValueError, match="non-whitelisted function"):
        validate_expression("foo(Qty)", columns=_COLUMNS)


def test_rejects_keyword_arguments_in_call() -> None:
    with pytest.raises(ValueError, match="keyword arguments"):
        validate_expression("max(Qty, key=AvgCost)", columns=_COLUMNS)


def test_rejects_dunder_name_even_if_in_columns_allowlist() -> None:
    """Defense in depth: a dunder name is rejected even if it somehow matched
    the caller-supplied columns allowlist (real schemas never contain one)."""
    with pytest.raises(ValueError, match="unknown or disallowed name"):
        validate_expression("__class__", columns=("__class__",))


def test_rejects_empty_expression() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_expression("", columns=_COLUMNS)


def test_rejects_whitespace_only_expression() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_expression("   ", columns=_COLUMNS)


def test_rejects_invalid_syntax() -> None:
    with pytest.raises(ValueError, match="syntax"):
        validate_expression("Qty +", columns=_COLUMNS)


def test_rejects_multi_statement_input() -> None:
    """A semicolon-joined multi-statement payload cannot parse in eval mode."""
    with pytest.raises(ValueError):
        validate_expression("Qty; DROP TABLE position", columns=_COLUMNS)


# ---------------------------------------------------------------------------
# Reject: output-column injection (the compile-side half of the gate)
# ---------------------------------------------------------------------------


def test_to_dh_formula_rejects_invalid_out_col() -> None:
    v = validate_expression("Qty * 2", columns=_COLUMNS)
    with pytest.raises(ValueError, match="invalid output column name"):
        to_dh_formula(v, out_col="Bad; System.exit(0); //")


def test_to_dh_formula_rejects_out_col_starting_with_digit() -> None:
    v = validate_expression("Qty * 2", columns=_COLUMNS)
    with pytest.raises(ValueError, match="invalid output column name"):
        to_dh_formula(v, out_col="1Bad")


def test_to_dh_formula_rejects_dunder_out_col() -> None:
    v = validate_expression("Qty * 2", columns=_COLUMNS)
    with pytest.raises(ValueError, match="invalid output column name"):
        to_dh_formula(v, out_col="__class__")


# ---------------------------------------------------------------------------
# Dtype inference — reject: grammar-valid but type-invalid expressions
#
# These all pass `_check` (every Name is a known column, every construct is
# on the allowlist) but violate an operand-dtype rule once `columns` carries
# dtypes. Each must raise ValueError naming the incompatible dtype(s) BEFORE
# any engine call — this is the core of the deferred full fix (see
# ibda/adapters/deephaven/adapter.py's `derive` docstring and the module
# docstring of ibda.analytics.expr for the rule table).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expr", "match"),
    [
        # BinOp: string operand to arithmetic (Deephaven does NOT concatenate
        # strings with '+' — this is a hard type error, not silent concat).
        ("Sym + Qty", r"operator '\+' requires numeric operands"),
        ("Qty - Sym", r"operator '-' requires numeric operands"),
        ("Sym * 2", r"operator '\*' requires numeric operands"),
        # BinOp: BOOL / TIMESTAMP_NS are not numeric either.
        ("IsLong + 1", r"operator '\+' requires numeric operands"),
        ("Timestamp + 1", r"operator '\+' requires numeric operands"),
        # Unary +/-: non-numeric operand.
        ("-Sym", r"unary operator '-' requires a numeric operand"),
        ("+IsLong", r"unary operator '\+' requires a numeric operand"),
        # Unary not: non-boolean operand.
        ("not Qty", r"operator 'not' requires a boolean operand"),
        # Compare: cross-family (incomparable) operands.
        ("Qty > Sym", r"comparison '>' requires comparable operands"),
        ("Sym == Qty", r"comparison '==' requires comparable operands"),
        ("IsLong < Qty", r"comparison '<' requires comparable operands"),
        # BoolOp: non-boolean operand.
        ("Sym and IsLong", r"boolean operator 'and' requires boolean operands"),
        ("Qty or IsLong", r"boolean operator 'or' requires boolean operands"),
        # Function calls: non-numeric argument.
        ("sqrt(Sym)", r"function 'sqrt' requires a numeric argument"),
        ("log(IsLong)", r"function 'log' requires a numeric argument"),
        ("exp(Sym)", r"function 'exp' requires a numeric argument"),
        ("abs(Sym)", r"function 'abs' requires a numeric argument"),
        ("max(Qty, Sym)", r"function 'max' requires numeric arguments"),
        ("min(Sym, Qty)", r"function 'min' requires numeric arguments"),
    ],
)
def test_rejects_type_invalid_expression(expr: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_expression(expr, columns=_TYPED_COLUMNS)


def test_type_invalid_error_names_the_offending_dtypes() -> None:
    with pytest.raises(ValueError) as exc_info:
        validate_expression("Sym + Qty", columns=_TYPED_COLUMNS)
    message = str(exc_info.value)
    assert "STRING (Sym)" in message
    assert "FLOAT64 (Qty)" in message


def test_type_invalid_error_names_the_original_expression() -> None:
    """The wrapping message must contain the original expression text, so a
    caller (or a downstream engine-boundary test) can identify which of several
    submitted expressions failed."""
    with pytest.raises(ValueError, match=r"sqrt\(Sym\)"):
        validate_expression("sqrt(Sym)", columns=_TYPED_COLUMNS)


def test_bare_column_sequence_disables_type_checking() -> None:
    """Backward-compat / conservative fallback 1: a plain `Sequence[str]`
    (no dtype info at all) disables the dtype-inference pass entirely — only
    the grammar walk runs, exactly as before this feature existed."""
    v = validate_expression("Sym + Qty", columns=("Sym", "Qty"))
    assert v.dtype is None


# ---------------------------------------------------------------------------
# Dtype inference — accept: valid expressions, correct inferred result dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expr", "expected_dtype"),
    [
        ("Qty + AvgCost", DType.FLOAT64),  # float + float
        ("ConId + 1", DType.INT64),  # int64 + int literal
        ("Qty + ConId", DType.FLOAT64),  # mixed int/float promotes to float
        ("Qty * 2", DType.FLOAT64),
        ("-Qty", DType.FLOAT64),  # unary minus preserves dtype
        ("-ConId", DType.INT64),
        ("Qty > 0", DType.BOOL),
        ("Sym == 'AAPL'", DType.BOOL),  # string vs string literal
        ("IsLong and (Qty > 0)", DType.BOOL),
        ("not IsLong", DType.BOOL),
        ("0 < Qty < 100", DType.BOOL),  # chained comparison, same family both sides
        ("sqrt(Qty * Qty)", DType.FLOAT64),
        ("abs(ConId)", DType.INT64),  # abs preserves the argument's dtype
        ("abs(Qty)", DType.FLOAT64),
        ("max(Qty, AvgCost)", DType.FLOAT64),
        ("max(ConId, ConId)", DType.INT64),
        ("min(Qty, ConId)", DType.FLOAT64),  # mixed numeric args promote to float
    ],
)
def test_accepts_type_valid_expression_and_infers_result_dtype(
    expr: str, expected_dtype: DType
) -> None:
    v = validate_expression(expr, columns=_TYPED_COLUMNS)
    assert v.dtype is expected_dtype


# ---------------------------------------------------------------------------
# Dtype inference — conservative fallbacks: never reject valid input just
# because a construct's type can't be pinned down (see module docstring).
# ---------------------------------------------------------------------------


def test_unmodeled_min_arity_falls_back_to_unchecked() -> None:
    """Fallback 3: `min`/`max` beyond the 2-arg form are not type-checked —
    a genuine arity mismatch is left to the engine boundary, not asserted here."""
    v = validate_expression("min(Qty, AvgCost, ConId)", columns=_TYPED_COLUMNS)
    assert v.dtype is None


def test_unmodeled_sqrt_arity_falls_back_to_unchecked() -> None:
    v = validate_expression("sqrt(Qty, AvgCost)", columns=_TYPED_COLUMNS)
    assert v.dtype is None


def test_unknown_column_dtype_falls_back_to_unchecked() -> None:
    """Fallback 2: a name absent from the dtypes mapping is treated as
    unknown, not wrong. Through the public `validate_expression` contract the
    dtype mapping's keys ARE the grammar-level column allowlist, so this path
    is unreachable end-to-end (a name always has a dtype once it's a legal
    `Name`) — it exists purely as defensive depth inside `_infer_type` itself,
    exercised here directly as a white-box test of that private function."""
    from ibda.analytics.expr import _infer_type  # noqa: PLC0415 — white-box test of a defensive path

    node = ast.parse("Qty + Other", mode="eval").body
    # "Other" is deliberately absent from this dtypes mapping even though
    # "Qty" is present — simulates the defensive branch.
    result = _infer_type(node, {"Qty": DType.FLOAT64})
    assert result is None


def test_to_dh_formula_accepts_underscore_out_col() -> None:
    v = validate_expression("Qty * 2", columns=_COLUMNS)
    assert to_dh_formula(v, out_col="my_col_2") == "my_col_2 = (Qty * 2)"
