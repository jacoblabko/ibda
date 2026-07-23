"""ibda.analytics.expr — the safe server-side derivation DSL (security core).

This module is THE security boundary for ``DataPort.derive``: a caller
(MCP/LLM/any external caller) supplies an expression string; it is parsed with
Python's ``ast``, walked against a strict allowlist, and only then re-emitted
as a Deephaven formula string. A raw/untrusted string NEVER reaches the engine
— only an expression that has passed :func:`validate_expression` may be
compiled by :func:`to_dh_formula`. This mirrors the project's existing trust
template one level up: ``ibda_mcp/query.py`` (``validate`` → ``to_predicate``)
never passes raw LLM text to the engine either.

Allowlist (everything else is rejected with ``ValueError``):

- ``Name`` — must be one of the caller-supplied ``columns`` (a base table's
  schema column names). Any dunder name (``__anything__``) is rejected
  outright, even if it happened to appear in ``columns``.
- ``Constant`` — ``int`` / ``float`` / ``str`` / ``bool`` literals only.
- ``BinOp`` — ``+ - * / %``.
- ``UnaryOp`` — ``+ - not``.
- ``Compare`` — ``== != > >= < <=`` (chained comparisons are supported and
  re-emitted as a conjunction).
- ``BoolOp`` — ``and`` / ``or``.
- ``Call`` — ONLY to a whitelisted function name: ``sqrt log exp abs min max``
  (no keyword arguments).

Rejected unconditionally: ``Attribute``, ``Subscript``, ``Lambda``, any
comprehension, any ``Call`` to a non-whitelisted name, any import (which
cannot even parse as a single expression), any name not in ``columns``.

Pure module: no engine (deephaven) import. Depends only on ``ibda.schema``
(also engine-neutral — no deephaven import) for the canonical ``DType``
vocabulary used by the dtype-inference pass below. Engine-neutral by design —
this is what lets ``ibda/adapters/deephaven/adapter.py`` be the only place a
Groovy formula string is ever handed to the engine.

Dtype inference (type-checking pass)
-------------------------------------
``validate_expression`` only ever checked *grammar* (the allowlist walk
above): a ``Name`` need only be a known column, with no regard to what that
column actually holds. That let grammar-valid but type-invalid expressions
through — e.g. ``Symbol + Quantity`` (string arithmetic) or ``sqrt(Symbol)``
(a string into ``Math.sqrt``) — which previously surfaced only as a raw
``deephaven.DHError`` once the engine tried to compile the re-emitted
formula (caught and relabelled as a ``ValueError`` at the adapter boundary —
that mitigation still stands as defense-in-depth, see
``ibda/adapters/deephaven/adapter.py``).

When ``validate_expression`` is called with a ``columns`` mapping (name ->
``DType``, not just a bare name sequence), it additionally walks the AST a
second time (``_infer_type``) assigning a dtype to every literal, column
reference, and sub-expression, and checking each allowed operator/function
against the operand-dtype rules below. A violation raises ``ValueError``
naming the offending sub-expression and its dtype — before any engine call.

Rules (result dtype given operand dtype(s)):

- ``+ - * / %`` (``BinOp``): both operands must be numeric (``INT64`` /
  ``FLOAT64``); Deephaven does **not** overload ``+`` as string
  concatenation, so a ``STRING`` operand is a hard error here, not silent
  concatenation. Result is ``FLOAT64`` if either operand is ``FLOAT64``,
  else ``INT64`` (Java/Deephaven-style numeric promotion).
- Unary ``+ -`` (``UnaryOp``): operand must be numeric; result dtype equals
  the operand's dtype. Unary ``not``: operand must be ``BOOL``; result is
  ``BOOL``.
- ``== != < <= > >=`` (``Compare``, chained or not): both sides of every
  comparison must be in the same *family* — numeric-vs-numeric,
  ``STRING``-vs-``STRING``, ``BOOL``-vs-``BOOL``, or
  ``TIMESTAMP_NS``-vs-``TIMESTAMP_NS``. Cross-family (e.g. ``Qty > Sym``) is
  rejected. Result is always ``BOOL``.
- ``and`` / ``or`` (``BoolOp``): every operand must be ``BOOL``. Result is
  ``BOOL``.
- ``sqrt`` / ``log`` / ``exp``: exactly one numeric argument; result is
  always ``FLOAT64`` (``Math.sqrt``/``log``/``exp`` return ``double``
  regardless of the argument's numeric type).
- ``abs``: exactly one numeric argument; result dtype equals the argument's
  dtype (``Math.abs`` is overloaded per numeric type).
- ``min`` / ``max``: exactly two numeric arguments; result is ``FLOAT64`` if
  either is ``FLOAT64``, else ``INT64`` (same promotion as ``BinOp``).

Conservative fallbacks — each returns/propagates ``None`` (unknown) rather
than guessing, so this pass can only ever REJECT something it is certain
about; it never invents a rejection from missing information:

1. **No dtype supplied at all** — a caller passing a bare ``Sequence[str]``
   (the original, name-only contract) disables type-checking entirely; only
   the grammar walk (``_check``) runs, exactly as before.
2. **A referenced column dtype is unknown** — not present in the ``columns``
   mapping (should not happen post-``_check``, but handled defensively).
3. **``sqrt``/``log``/``exp``/``abs`` called with an argument count other
   than one**, or **``min``/``max`` called with a count other than two** —
   arity is not modeled here (the grammar walk already permits any arg
   count); a genuine arity mismatch is still caught at the engine boundary.
4. **Any operand whose own dtype could not be determined** (transitively,
   via 1-3) disables checking for every operator built on top of it — a
   ``None`` never gets treated as "definitely wrong," only as "unknown."

A statically-inferred dtype, when available, is exposed on
``ValidatedExpr.dtype`` — see ``ibda/tests_jvm/test_expr_dtype_agrees_with_engine.py``
for the JVM check that this static inference agrees with the engine's own
post-hoc schema recompute (``ibda/adapters/deephaven/adapter.py``'s
``_schema_from_dh_handle``).
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ibda.schema import DType

# ---------------------------------------------------------------------------
# Allowlists
# ---------------------------------------------------------------------------

#: Function names a validated expression may call. Mapped to their Deephaven
#: (Java ``Math``) equivalent in :func:`to_dh_formula`.
ALLOWED_FUNCS: frozenset[str] = frozenset({"sqrt", "log", "exp", "abs", "min", "max"})

_FUNC_TO_DH: dict[str, str] = {
    "sqrt": "Math.sqrt",
    "log": "Math.log",
    "exp": "Math.exp",
    "abs": "Math.abs",
    "min": "Math.min",
    "max": "Math.max",
}

_BINOP_TO_DH: dict[type[ast.operator], str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.Mod: "%",
}

_UNARYOP_TO_DH: dict[type[ast.unaryop], str] = {
    ast.UAdd: "+",
    ast.USub: "-",
    ast.Not: "!",
}

#: Display symbol for a unary operator in a *source-level* (Python-syntax)
#: error message — as opposed to ``_UNARYOP_TO_DH``, which is the compiled
#: Deephaven/Groovy symbol (``!`` rather than ``not``).
_UNARYOP_DISPLAY: dict[type[ast.unaryop], str] = {
    ast.UAdd: "+",
    ast.USub: "-",
    ast.Not: "not",
}

_CMPOP_TO_DH: dict[type[ast.cmpop], str] = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
}

_BOOLOP_TO_DH: dict[type[ast.boolop], str] = {
    ast.And: "&&",
    ast.Or: "||",
}

#: Display word for a boolean operator in a source-level error message
#: (``and``/``or``, as opposed to the compiled ``&&``/``||`` above).
_BOOLOP_DISPLAY: dict[type[ast.boolop], str] = {
    ast.And: "and",
    ast.Or: "or",
}

#: Dtypes treated as numeric for arithmetic/function-argument rules.
_NUMERIC_DTYPES: frozenset[DType] = frozenset({DType.INT64, DType.FLOAT64})

#: A Deephaven/Java output-column identifier: letters/digits/underscore, not
#: starting with a digit. Also used to reject a dunder-prefixed output column.
_VALID_OUT_COL: re.Pattern[str] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# ---------------------------------------------------------------------------
# ValidatedExpr — the only thing to_dh_formula will accept
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidatedExpr:
    """An expression that has passed :func:`validate_expression`'s allowlist walk.

    Opaque to callers beyond ``source`` (the original text, for logging/display)
    and ``dtype`` (the statically-inferred result dtype, see module docstring
    "Dtype inference"). ``node`` is the parsed AST body, consumed only by
    :func:`to_dh_formula`. Constructing this directly (bypassing
    :func:`validate_expression`) is possible in Python but unsupported —
    always obtain a ``ValidatedExpr`` via :func:`validate_expression`.
    """

    source: str
    node: ast.expr = field(repr=False)
    #: The statically-inferred result dtype, or ``None`` if it could not be
    #: determined (no dtype-carrying ``columns`` mapping was supplied to
    #: :func:`validate_expression`, or inference conservatively fell back —
    #: see the module docstring's "Conservative fallbacks" list).
    dtype: DType | None = None


# ---------------------------------------------------------------------------
# Validation — the allowlist walk
# ---------------------------------------------------------------------------


def _check(node: ast.expr, columns: frozenset[str]) -> None:
    """Raise ``ValueError`` if *node* (or any descendant) is not on the allowlist."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or isinstance(node.value, (int, float, str)):
            return
        raise ValueError(f"unsupported literal type: {type(node.value).__name__!r}")

    if isinstance(node, ast.Name):
        if node.id.startswith("__") or node.id not in columns:
            raise ValueError(f"unknown or disallowed name: {node.id!r}")
        return

    if isinstance(node, ast.BinOp):
        if type(node.op) not in _BINOP_TO_DH:
            raise ValueError(f"disallowed binary operator: {type(node.op).__name__}")
        _check(node.left, columns)
        _check(node.right, columns)
        return

    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UNARYOP_TO_DH:
            raise ValueError(f"disallowed unary operator: {type(node.op).__name__}")
        _check(node.operand, columns)
        return

    if isinstance(node, ast.Compare):
        if any(type(op) not in _CMPOP_TO_DH for op in node.ops):
            raise ValueError("disallowed comparison operator")
        _check(node.left, columns)
        for comparator in node.comparators:
            _check(comparator, columns)
        return

    if isinstance(node, ast.BoolOp):
        if type(node.op) not in _BOOLOP_TO_DH:
            raise ValueError(f"disallowed boolean operator: {type(node.op).__name__}")
        for value in node.values:
            _check(value, columns)
        return

    if isinstance(node, ast.Call):
        if node.keywords:
            raise ValueError("keyword arguments are not allowed in function calls")
        if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCS:
            func_repr = node.func.id if isinstance(node.func, ast.Name) else type(node.func).__name__
            raise ValueError(f"call to non-whitelisted function: {func_repr!r}")
        for arg in node.args:
            _check(arg, columns)
        return

    # Attribute, Subscript, Lambda, ListComp/SetComp/DictComp/GeneratorExp,
    # IfExp, Starred, JoinedStr, NamedExpr, and anything else land here.
    raise ValueError(f"disallowed expression construct: {type(node).__name__}")


# ---------------------------------------------------------------------------
# Dtype inference — the type-checking pass (see module docstring)
# ---------------------------------------------------------------------------


def _promote_numeric(left: DType, right: DType) -> DType:
    """Java/Deephaven-style numeric promotion: FLOAT64 if either side is FLOAT64."""
    return DType.FLOAT64 if DType.FLOAT64 in (left, right) else DType.INT64


def _family(dtype: DType) -> str:
    """Group dtypes into comparability families: all numerics are mutually
    comparable; STRING/BOOL/TIMESTAMP_NS are each comparable only to themselves."""
    return "numeric" if dtype in _NUMERIC_DTYPES else dtype.value


def _describe(node: ast.expr, dtype: DType) -> str:
    """Render "DTYPE (source-snippet)" for an error message, e.g. "STRING (Sym)"."""
    return f"{dtype.value} ({ast.unparse(node)})"


def _infer_type(node: ast.expr, dtypes: Mapping[str, DType]) -> DType | None:
    """Infer the dtype *node* evaluates to, or ``None`` if it cannot be determined.

    Conservative by construction: returns ``None`` — rather than guessing —
    whenever a sub-expression's dtype is unknowable (see module docstring
    "Conservative fallbacks"). A ``None`` operand disables checking for
    whatever operator sits above it, so this function can only ever reject a
    construct it is certain violates a rule; it never rejects from missing
    information. Raises ``ValueError`` (mirroring :func:`_check`) the moment a
    rule is definitely violated. Only ever called on a tree that has already
    passed :func:`_check`, so every node type below is one `_check` allows.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return DType.BOOL
        if isinstance(node.value, int):
            return DType.INT64
        if isinstance(node.value, float):
            return DType.FLOAT64
        if isinstance(node.value, str):
            return DType.STRING
        return None  # unreachable post-_check; conservative regardless

    if isinstance(node, ast.Name):
        # Fallback 1/2: absent from `dtypes` (no dtype-carrying mapping was
        # supplied at all, or — defensively — this particular name isn't in
        # it) means "unknown," not "wrong."
        return dtypes.get(node.id)

    if isinstance(node, ast.BinOp):
        left = _infer_type(node.left, dtypes)
        right = _infer_type(node.right, dtypes)
        if left is None or right is None:
            return None  # fallback 4
        if left not in _NUMERIC_DTYPES or right not in _NUMERIC_DTYPES:
            op_sym = _BINOP_TO_DH[type(node.op)]
            raise ValueError(
                f"operator {op_sym!r} requires numeric operands, got "
                f"{_describe(node.left, left)} and {_describe(node.right, right)}"
            )
        return _promote_numeric(left, right)

    if isinstance(node, ast.UnaryOp):
        operand = _infer_type(node.operand, dtypes)
        if operand is None:
            return None  # fallback 4
        if isinstance(node.op, ast.Not):
            if operand is not DType.BOOL:
                raise ValueError(
                    "operator 'not' requires a boolean operand, got "
                    f"{_describe(node.operand, operand)}"
                )
            return DType.BOOL
        # UAdd / USub
        if operand not in _NUMERIC_DTYPES:
            op_sym = _UNARYOP_DISPLAY[type(node.op)]
            raise ValueError(
                f"unary operator {op_sym!r} requires a numeric operand, got "
                f"{_describe(node.operand, operand)}"
            )
        return operand

    if isinstance(node, ast.Compare):
        left_node: ast.expr = node.left
        left = _infer_type(left_node, dtypes)
        for op, right_node in zip(node.ops, node.comparators):
            right = _infer_type(right_node, dtypes)
            if left is not None and right is not None and _family(left) != _family(right):
                op_sym = _CMPOP_TO_DH[type(op)]
                raise ValueError(
                    f"comparison {op_sym!r} requires comparable operands, got "
                    f"{_describe(left_node, left)} and {_describe(right_node, right)}"
                )
            left_node, left = right_node, right
        return DType.BOOL

    if isinstance(node, ast.BoolOp):
        op_word = _BOOLOP_DISPLAY[type(node.op)]
        for value in node.values:
            value_type = _infer_type(value, dtypes)
            if value_type is not None and value_type is not DType.BOOL:
                raise ValueError(
                    f"boolean operator {op_word!r} requires boolean operands, got "
                    f"{_describe(value, value_type)}"
                )
        return DType.BOOL

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("call target must be a simple name")  # unreachable post-_check
        func_name = node.func.id
        arg_types: list[DType | None] = [_infer_type(a, dtypes) for a in node.args]

        if func_name in ("sqrt", "log", "exp"):
            # Fallback 3: arity isn't modeled — only the 1-arg form is checked.
            if len(arg_types) != 1 or arg_types[0] is None:
                return None
            arg_type = arg_types[0]
            if arg_type not in _NUMERIC_DTYPES:
                raise ValueError(
                    f"function {func_name!r} requires a numeric argument, got "
                    f"{_describe(node.args[0], arg_type)}"
                )
            return DType.FLOAT64  # Math.sqrt/log/exp always return double

        if func_name == "abs":
            if len(arg_types) != 1 or arg_types[0] is None:
                return None
            arg_type = arg_types[0]
            if arg_type not in _NUMERIC_DTYPES:
                raise ValueError(
                    f"function 'abs' requires a numeric argument, got "
                    f"{_describe(node.args[0], arg_type)}"
                )
            return arg_type  # Math.abs is overloaded per numeric type

        if func_name in ("min", "max"):
            # Fallback 3: only the 2-arg overload is modeled (Java's Math.min/
            # max have no 3+-arg overload either, but a genuine arity mismatch
            # is left to the engine boundary rather than asserted here).
            if len(arg_types) != 2 or arg_types[0] is None or arg_types[1] is None:
                return None
            a0, a1 = arg_types[0], arg_types[1]
            if a0 not in _NUMERIC_DTYPES or a1 not in _NUMERIC_DTYPES:
                raise ValueError(
                    f"function {func_name!r} requires numeric arguments, got "
                    f"{_describe(node.args[0], a0)} and {_describe(node.args[1], a1)}"
                )
            return _promote_numeric(a0, a1)

        return None  # unreachable: every ALLOWED_FUNCS member is handled above

    return None  # any other node type: unreachable post-_check; conservative regardless


def validate_expression(
    expr: str, *, columns: Sequence[str] | Mapping[str, DType]
) -> ValidatedExpr:
    """Parse *expr* and validate it against the safe-DSL allowlist.

    Args:
        expr: The candidate expression (e.g. ``"sqrt(Qty * Qty) / AvgCost"``).
        columns: The base table's column names — the only ``Name`` references
            *expr* may use. Pass a bare ``Sequence[str]`` for grammar-only
            validation (the original contract), or a ``Mapping[str, DType]``
            (name -> canonical dtype, e.g. from ``Schema.columns``) to ALSO
            run the dtype-inference pass described in the module docstring —
            rejecting grammar-valid but type-invalid expressions (e.g.
            ``"Symbol + Quantity"``, ``"sqrt(Symbol)"``) before any engine
            call, rather than only at the engine boundary.

    Returns:
        A :class:`ValidatedExpr` wrapping the parsed AST for
        :func:`to_dh_formula`, and (when *columns* carried dtypes) the
        statically-inferred result dtype on ``.dtype``. Never hand a raw
        string to the engine — always go through this function first.

    Raises:
        ValueError: On invalid syntax, any construct/name/function not on
            the allowlist (see module docstring), or — when *columns*
            supplies dtypes — a dtype-inference rule violation. This is the
            sole gate: nothing in *expr* reaches the compute engine until it
            has passed here.
    """
    if not expr or not expr.strip():
        raise ValueError("expression must not be empty")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid expression syntax: {exc}") from exc

    body = tree.body
    column_dtypes: Mapping[str, DType]
    if isinstance(columns, Mapping):
        column_names = frozenset(columns.keys())
        column_dtypes = columns
    else:
        column_names = frozenset(columns)
        column_dtypes = {}
    _check(body, column_names)
    try:
        result_dtype = _infer_type(body, column_dtypes)
    except ValueError as exc:
        raise ValueError(f"type error in expression {expr!r}: {exc}") from exc
    return ValidatedExpr(source=expr, node=body, dtype=result_dtype)


# ---------------------------------------------------------------------------
# Compilation — re-emit a validated AST as a Deephaven formula
# ---------------------------------------------------------------------------


def _emit(node: ast.expr) -> str:
    """Recursively re-emit *node* (already validated) as Deephaven syntax.

    Defensive: mirrors the allowlist in :func:`_check` and raises ``ValueError``
    on anything unexpected, even though only a validated tree should ever
    reach this function.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return "true" if node.value else "false"
        if isinstance(node.value, str):
            return json.dumps(node.value)
        return repr(node.value)

    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.BinOp):
        op = _BINOP_TO_DH[type(node.op)]
        return f"({_emit(node.left)} {op} {_emit(node.right)})"

    if isinstance(node, ast.UnaryOp):
        op = _UNARYOP_TO_DH[type(node.op)]
        return f"({op}{_emit(node.operand)})"

    if isinstance(node, ast.Compare):
        clauses: list[str] = []
        left = node.left
        for cmp_op, comparator in zip(node.ops, node.comparators):
            clauses.append(f"({_emit(left)} {_CMPOP_TO_DH[type(cmp_op)]} {_emit(comparator)})")
            left = comparator
        return clauses[0] if len(clauses) == 1 else " && ".join(clauses)

    if isinstance(node, ast.BoolOp):
        op = _BOOLOP_TO_DH[type(node.op)]
        return "(" + f" {op} ".join(_emit(v) for v in node.values) + ")"

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("call target must be a simple name")  # unreachable post-validation
        dh_func = _FUNC_TO_DH[node.func.id]
        args = ", ".join(_emit(a) for a in node.args)
        return f"{dh_func}({args})"

    raise ValueError(f"cannot compile node type {type(node).__name__} to a formula")  # unreachable post-validation


def to_dh_formula(validated: ValidatedExpr, *, out_col: str) -> str:
    """Re-emit *validated* as a Deephaven ``"OutCol = <expr>"`` formula string.

    Args:
        validated: A :class:`ValidatedExpr` produced by
            :func:`validate_expression`.
        out_col: The new column name. Validated against a strict
            Java-identifier pattern here (not by :func:`validate_expression`,
            which only checks the expression body) — this closes the
            injection vector on the *output* side: a caller cannot smuggle
            engine syntax through the output-column name.

    Returns:
        A formula string suitable for ``Table.update([...])``.

    Raises:
        ValueError: If *out_col* is not a valid identifier (or is
            dunder-prefixed).
    """
    if not _VALID_OUT_COL.match(out_col) or out_col.startswith("__"):
        raise ValueError(f"invalid output column name: {out_col!r}")
    return f"{out_col} = {_emit(validated.node)}"
