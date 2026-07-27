"""ibda.adapters.deephaven.adapter — the Deephaven implementation of DataPort.

THE ONLY MODULE IN ibda/ ALLOWED TO IMPORT deephaven. Consumers never see a
deephaven Table; every terminal op returns Apache Arrow. deephaven is untyped —
this module is verified by the JVM contract tests, not by mypy.

Type-mismatch notes (verified empirically, deephaven-server 41.6):
- deephaven tables round-trip via pandas and produce pa.large_string() for
  string columns, not pa.string(). _normalize_arrow() casts these to string.
- update.added() in table_listener returns dict[str, np.ndarray]; numpy Unicode
  arrays ('< U…') also produce large_string on pa.Table.from_pandas. Same cast
  is applied in _subscribe's inner callback.
- Timestamps from update.added() numpy arrays arrive as datetime64[ns] (no tz);
  _normalize_arrow() casts these to timestamp[ns, tz=UTC] to match the schema.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import pyarrow as pa

from ibda.analytics.expr import to_dh_formula, validate_expression
from ibda.errors import UnknownTable
from ibda.port import DataPort
from ibda.result import DeltaCallback, Result, Stream
from ibda.schema import ALL as ALL_SCHEMAS
from ibda.schema import DType, Schema
from ibda.schema._core import Column


#: Aggregation vocabulary accepted by ``group_by``'s ``aggs`` mapping. Kept in
#: sync with the ``how`` values documented on ``DataPort.group_by`` (ibda/port.py).
_SUPPORTED_AGGS: frozenset[str] = frozenset({"sum", "mean", "last", "count"})


def _normalize_arrow(tbl: pa.Table, schema: Schema | None = None) -> pa.Table:
    """Cast deephaven-specific Arrow types to canonical schema types.

    Deephaven string columns round-trip through pandas as pa.large_string();
    the canonical schemas declare pa.string(). Timestamp columns from
    update.added() numpy arrays arrive without timezone info.

    When *schema* is provided, float64 columns declared INT64 in the schema are
    cast back to nullable int64.  Current Deephaven (server 41.6+) returns
    ``pd.Int64Dtype()`` for long columns with nulls, so ``to_pandas``→
    ``from_pandas`` already yields Arrow int64 with a null bitmap — this cast
    guard is NOT triggered on the current version.  It is kept as
    forward-compatibility protection in case an older Deephaven version is used
    or the behaviour regresses.  See ``ibda/tests_jvm/
    test_nullable_int64_snapshot.py`` for the JVM characterisation (step 1–3)
    and the guard's unit-test coverage (steps 4–5b).
    """
    new_arrays: list[pa.Array] = []
    new_fields: list[pa.Field] = []
    for i in range(tbl.num_columns):
        field = tbl.schema.field(i)
        col = tbl.column(i)
        if field.type == pa.large_string():
            col = col.cast(pa.string())
            field = field.with_type(pa.string())
        elif field.type == pa.timestamp("ns"):
            col = col.cast(pa.timestamp("ns", tz="UTC"))
            field = field.with_type(pa.timestamp("ns", tz="UTC"))
        elif schema is not None and pa.types.is_floating(field.type):
            # If the canonical schema declares this column INT64 but deephaven
            # returned float64 (NULL_LONG round-trip), cast back to int64.
            schema_col: Column | None = next(
                (c for c in schema.columns if c.name == field.name), None
            )
            if schema_col is not None and schema_col.dtype == DType.INT64:
                try:
                    col = col.cast(pa.int64())
                    field = field.with_type(pa.int64())
                except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                    # Non-integer values in an INT64 column — leave as float64
                    # so schema.validate() surfaces the issue to the caller.
                    pass
        new_arrays.append(col)
        new_fields.append(field)
    return pa.table(
        {f.name: arr for f, arr in zip(new_fields, new_arrays)},
        schema=pa.schema(new_fields),
    )


def _schema_from_dh_handle(handle: Any) -> Schema:
    """Derive a canonical Schema from a Deephaven table's meta_table.

    Uses ``meta_table`` (cheap — no data scan) to read column names and Java
    DataType strings, then maps them to canonical DType values.  Called after
    ``group_by`` and ``as_of_join``, which change the column set so the input
    schema is no longer accurate.

    The returned Schema has ``name="<derived>"``.  Unmapped DataType strings
    fall back to STRING (conservative — schema.validate() will catch
    genuine mismatches).
    """
    from deephaven.pandas import to_pandas  # noqa: PLC0415 — engine-confined

    meta_df = to_pandas(handle.meta_table)

    # Map Deephaven DataType string substrings → canonical DType.
    # Ordered from most-specific to least-specific to avoid false matches.
    _DH_TYPE_MAP: list[tuple[str, DType]] = [
        ("Instant", DType.TIMESTAMP_NS),
        ("double", DType.FLOAT64),
        ("float", DType.FLOAT64),
        ("long", DType.INT64),
        ("int", DType.INT64),
        ("Boolean", DType.BOOL),
        ("boolean", DType.BOOL),
        ("String", DType.STRING),
    ]

    columns: list[Column] = []
    for _, row in meta_df.iterrows():
        col_name = str(row["Name"])
        dt_str = str(row["DataType"])
        dtype = next(
            (v for k, v in _DH_TYPE_MAP if k in dt_str),
            DType.STRING,
        )
        columns.append(Column(name=col_name, dtype=dtype, nullable=True))

    return Schema(name="<derived>", columns=tuple(columns))


class DeephavenPort(DataPort):
    """Adapts canonical DataPort operations onto live deephaven Tables."""

    def __init__(self, tables: Mapping[str, Any]) -> None:
        # name -> live deephaven Table. In SP2 these come from the IBKR adapter.
        self._tables: dict[str, Any] = dict(tables)
        # Registered derived tables (register_derived): name -> its computed
        # Schema (not in ALL_SCHEMAS — resolved here instead). The first
        # supported post-connect mutation of self._tables (see ADR).
        self._derived_schemas: dict[str, Schema] = {}
        # Trust-origin tag per derived name; v1 is DSL-only so every entry is
        # "untrusted" today, but the tag lets a future trusted tier be added
        # without weakening this default (see ADR "Trust tiers").
        self._derived_origin: dict[str, str] = {}

    def _wrap(self, handle: Any, schema: Schema) -> Result:
        return Result(handle=handle, schema=schema, port=self)

    def table(self, name: str) -> Result:
        if name in self._derived_schemas:
            return self._wrap(self._tables[name], self._derived_schemas[name])
        if name not in self._tables:
            raise UnknownTable(
                f"no canonical table {name!r} (have: {sorted(self._tables)})"
            )
        if name not in ALL_SCHEMAS:
            raise UnknownTable(f"{name!r} is not a known canonical schema")
        return self._wrap(self._tables[name], ALL_SCHEMAS[name])

    def derived_origin(self, name: str) -> str | None:
        """Return the trust-origin tag for a registered derived table, or None."""
        return self._derived_origin.get(name)

    def derived_schema(self, name: str) -> Schema | None:
        """Schema of a `register_derived` table, or None. Pure lookup; no engine touch."""
        return self._derived_schemas.get(name)

    def live_table(self, name: str) -> Any:
        """IN-ENGINE ESCAPE HATCH. Return the live deephaven Table for *name*.

        Deliberately declared ONLY on ``DeephavenPort`` — NOT on the
        ``DataPort`` ABC, so it is invisible to any consumer holding a
        ``DataPort``-typed reference (including external, non-Deephaven
        adapters). Valid only inside a Deephaven worker (e.g. a
        ``deephaven.ui``-based UI or other in-engine analytics) that needs
        live table algebra ``DataPort``'s validated op set cannot express.
        External clients must use ``table(name).snapshot()`` /
        ``.subscribe()`` instead — engine-hiding is preserved for them
        because this method simply does not exist on any type they can hold.

        Reuses ``table()``'s name resolution, so both canonical and
        registered-derived (custom-lambda) names resolve identically, and
        ``UnknownTable`` still guards a typo'd name.

        This escape hatch was a deliberate, considered design decision (dated 2026-07-08),
        not an oversight: it deliberately reintroduces engine visibility for in-process
        callers, in exchange for table algebra the validated ``DataPort`` op set cannot
        express, while every out-of-process consumer keeps the engine fully hidden.

        Parameters
        ----------
        name:
            A canonical table name (e.g. ``"position"``) or a name
            previously passed to ``register_derived``.

        Returns
        -------
        The live deephaven ``Table`` object (untyped; ``Any`` — deephaven is
        fully untyped in this project). Raises ``UnknownTable`` if *name* is
        neither a canonical nor a registered-derived table.
        """
        return self.table(name).handle

    def filter(self, result: Result, predicate: str) -> Result:
        # `filter`'s predicate is handed to `.where()` as-is — it never routes
        # through `ibda.analytics.expr` at all (unlike `derive`, below), so
        # neither the grammar walk nor the dtype-inference pass ever sees it.
        # A grammar-valid-looking-but-type-invalid predicate (e.g.
        # `Symbol > Qty`) therefore still reaches the engine and raises a raw
        # `deephaven.DHError`, breaking the "engine is an implementation
        # detail" contract — reraise as a port-level ValueError instead.
        # MITIGATION (defense-in-depth, unchanged by the `derive`-side dtype
        # inference added below): catches the engine's own rejection and
        # relabels it. Routing `filter` through the safe-DSL validator is a
        # separate, larger change (predicates aren't the DSL's `derive`
        # column-expression grammar) — out of scope here.
        from deephaven.dherror import DHError  # noqa: PLC0415 — deephaven import, lazy

        try:
            handle = result.handle.where(predicate)  # type: ignore[attr-defined]
        except DHError as exc:
            raise ValueError(f"invalid filter predicate {predicate!r}: {exc}") from exc
        return self._wrap(handle, result.schema)

    def time_window(
        self, result: Result, *, column: str, start: str, end: str
    ) -> Result:
        expr = (
            f"{column} >= parseInstant(`{start}`) "
            f"&& {column} <= parseInstant(`{end}`)"
        )
        # Same mitigation as `filter`: this hand-built `parseInstant(...)`
        # expression never routes through `ibda.analytics.expr` either (it
        # isn't expressible in the DSL grammar — `parseInstant` isn't in
        # ALLOWED_FUNCS and the backtick literal isn't valid Python syntax),
        # so `column` may name a non-timestamp column, which is
        # grammar-valid-looking but type-invalid once compared against
        # `parseInstant(...)` — surface a ValueError, not a raw engine error.
        from deephaven.dherror import DHError  # noqa: PLC0415 — deephaven import, lazy

        try:
            handle = result.handle.where(expr)  # type: ignore[attr-defined]
        except DHError as exc:
            raise ValueError(
                f"invalid time_window on column {column!r}: {exc}"
            ) from exc
        return self._wrap(handle, result.schema)

    def group_by(
        self, result: Result, *, by: Sequence[str], aggs: dict[str, str]
    ) -> Result:
        # Validate every requested `how` BEFORE importing deephaven / touching the
        # engine, so an unsupported value raises a port-level ValueError (never a
        # bare KeyError leaking the internal `_list_fn` dispatch table) and is
        # cheaply testable without a JVM. Keep _SUPPORTED_AGGS in sync with the
        # vocabulary advertised by DataPort.group_by's docstring (ibda/port.py).
        unsupported = sorted({how for how in aggs.values() if how not in _SUPPORTED_AGGS})
        if unsupported:
            raise ValueError(
                f"unsupported aggregation {unsupported[0]!r}; "
                f"supported: {sorted(_SUPPORTED_AGGS)}"
            )

        from deephaven import agg
        from deephaven.dherror import DHError  # noqa: PLC0415 — deephaven import, lazy

        # agg.sum_ / agg.avg / agg.last accept a list of column names.
        # agg.count_ is special: it takes a SINGLE STRING (the output column name),
        # not a list. Mixing the two calling conventions requires an explicit branch.
        _list_fn: dict[str, Any] = {
            "sum": agg.sum_,
            "mean": agg.avg,
            "last": agg.last,
        }
        agg_list: list[Any] = []
        for col, how in aggs.items():
            if how == "count":
                agg_list.append(agg.count_(col))  # plain string, not a list
            else:
                agg_list.append(_list_fn[how]([col]))
        # Same mitigation as `filter`/`time_window`/`derive` above: the
        # `unsupported`-function check and (upstream) query.validate() cover the
        # common cases, but a key that names a column of the wrong dtype for its
        # aggregation (e.g. sum on a STRING column) still reaches the engine and
        # raises a raw `deephaven.DHError` — reraise as a port-level ValueError
        # so it surfaces as the documented `invalid_query` at the MCP boundary
        # instead of an uncaught traceback (group_by was the one op missing this).
        try:
            handle = result.handle.agg_by(agg_list, by=list(by))  # type: ignore[attr-defined]
        except DHError as exc:
            raise ValueError(f"invalid group_by aggs {dict(aggs)!r}: {exc}") from exc
        # group_by changes the column set — derive a Schema from the actual output
        # so result.schema.column_names is accurate and snapshot() validates correctly.
        derived = _schema_from_dh_handle(handle)
        return self._wrap(handle, derived)

    def as_of_join(
        self,
        left: Result,
        right: Result,
        *,
        on: Sequence[str],
        joins: Sequence[str],
    ) -> Result:
        handle = left.handle.aj(  # type: ignore[attr-defined]
            right.handle, on=list(on), joins=list(joins)
        )
        # as_of_join adds columns from the right table — derive a Schema from
        # the actual output so result.schema reflects the joined column set.
        derived = _schema_from_dh_handle(handle)
        return self._wrap(handle, derived)

    def sort(self, result: Result, *, keys: Sequence[tuple[str, bool]]) -> Result:
        """Sort *result* by *keys* (column, descending) pairs.

        Delegates to ``deephaven.Table.sort(order_by, order)`` which accepts
        per-key ``SortDirection`` values — this is the only API that supports
        mixed ascending/descending multi-key sorts.  Ascending uses
        ``SortDirection.ASCENDING``; descending uses ``SortDirection.DESCENDING``.

        Note: this layer is JVM-verified, not mypy-verified (deephaven is
        treated as untyped); correctness is confirmed by the contract tests.
        """
        from deephaven import SortDirection  # noqa: PLC0415 — deephaven import, lazy

        handle = result.handle.sort(  # type: ignore[attr-defined]
            order_by=[c for c, _ in keys],
            order=[
                SortDirection.DESCENDING if desc else SortDirection.ASCENDING
                for _, desc in keys
            ],
        )
        return self._wrap(handle, result.schema)

    def derive(self, result: Result, *, columns: Mapping[str, str]) -> Result:
        """Add engine-computed columns via `Table.update([...])`.

        THE ONLY PLACE a Groovy formula string is handed to the engine for a
        caller-supplied expression. Each expression is validated against
        `result.schema` by `ibda.analytics.expr` BEFORE `to_dh_formula`
        re-emits it and BEFORE `update()` is called — an invalid/malicious
        expression raises `ValueError` here and never reaches the engine.

        `validate_expression` is called with a full name -> `DType` mapping
        (not just column names), so it runs BOTH the grammar walk (AST
        allowlist) AND the dtype-inference pass (see `ibda.analytics.expr`
        module docstring): a grammar-valid but type-invalid expression (e.g.
        `Symbol + Quantity`, string arithmetic; `sqrt(Symbol)`, a string into
        `Math.sqrt`) is now rejected here, statically, before any engine call
        — not merely reclassified after the engine rejects it.

        The dtype-inference pass is deliberately conservative (falls back to
        "unknown" rather than guessing — see its module docstring), so a
        construct it cannot reason about still reaches `update()` unchecked.
        DEFENSE-IN-DEPTH MITIGATION (kept, not removed by the above): catch
        `DHError` from `update()` and re-raise as a port-level `ValueError`
        naming the offending expression, so the "engine is an implementation
        detail" contract holds even for whatever the static pass conservatively
        let through.
        """
        from deephaven.dherror import DHError  # noqa: PLC0415 — deephaven import, lazy

        column_dtypes: dict[str, DType] = {c.name: c.dtype for c in result.schema.columns}
        formulas: list[str] = []
        for out_col, expr in columns.items():
            validated = validate_expression(expr, columns=column_dtypes)
            formulas.append(to_dh_formula(validated, out_col=out_col))
        try:
            handle = result.handle.update(formulas)  # type: ignore[attr-defined]
        except DHError as exc:
            raise ValueError(
                f"invalid derived-column expression(s) {dict(columns)!r}: {exc}"
            ) from exc
        # update() adds columns — derive a Schema from the actual output so
        # result.schema reflects the new column set (same pattern as
        # group_by/as_of_join above).
        derived = _schema_from_dh_handle(handle)
        return self._wrap(handle, derived)

    def register_derived(
        self, name: str, *, base: Result, columns: Mapping[str, str]
    ) -> Result:
        if name in ALL_SCHEMAS:
            raise ValueError(
                f"cannot register derived table {name!r}: collides with a canonical table name"
            )
        if name in self._tables:
            raise ValueError(f"derived table {name!r} is already registered")
        derived = self.derive(base, columns=columns)
        # Store the handle directly: table(name) re-wraps the SAME handle on
        # every call, so every caller resolving "name" shares one live
        # engine-side computation — never recomputed per lookup.
        self._tables[name] = derived.handle
        self._derived_schemas[name] = derived.schema
        self._derived_origin[name] = "untrusted"
        return self._wrap(derived.handle, derived.schema)

    def _snapshot(self, result: Result) -> pa.Table:
        from deephaven.pandas import to_pandas

        df: pd.DataFrame = to_pandas(result.handle)
        raw = pa.Table.from_pandas(df, preserve_index=False)
        return _normalize_arrow(raw, result.schema)

    def _subscribe(self, result: Result, callback: DeltaCallback) -> Stream:
        from deephaven.table_listener import listen

        def _on_update(update: Any, is_replay: bool) -> None:
            # update.added() returns dict[str, np.ndarray] (one array per column)
            added: dict[str, Any] = update.added()
            if added:
                df = pd.DataFrame(added)
                raw = pa.Table.from_pandas(df, preserve_index=False)
                callback(_normalize_arrow(raw, result.schema))

        handle = listen(result.handle, _on_update)
        return Stream(cancel=handle.stop)
