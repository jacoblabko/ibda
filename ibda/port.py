"""ibda.port — the engine-neutral access port.

DataPort is the capability contract the domain depends on: the small, stable
set of table operations (table / filter / time_window / group_by / as_of_join)
plus snapshot/subscribe. The compute engine is an implementation detail behind
this ABC (see ibda.adapters.deephaven). No engine import here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

import pyarrow as pa

from ibda.result import DeltaCallback, Result, Stream
from ibda.schema import Schema


class DataPort(ABC):
    """Abstract access port. One adapter implements it per compute engine."""

    @abstractmethod
    def table(self, name: str) -> Result:
        """Return the named canonical table (e.g. 'position', 'execution') as a Result."""

    @abstractmethod
    def filter(self, result: Result, predicate: str) -> Result:
        """Filter rows by an engine-neutral boolean predicate string (e.g. "Qty > 0")."""

    @abstractmethod
    def time_window(self, result: Result, *, column: str, start: str, end: str) -> Result:
        """Restrict to rows whose `column` timestamp falls in [start, end] (ISO-8601 UTC)."""

    @abstractmethod
    def group_by(
        self, result: Result, *, by: Sequence[str], aggs: dict[str, str]
    ) -> Result:
        """Group by `by` columns, applying `aggs` (column -> 'sum'|'mean'|'last'|'count')."""

    @abstractmethod
    def as_of_join(
        self,
        left: Result,
        right: Result,
        *,
        on: Sequence[str],
        joins: Sequence[str],
    ) -> Result:
        """As-of join `right` onto `left` on `on` keys (last `on` is the temporal key),
        bringing `joins` columns from `right`. This is the headline cross-source op."""

    @abstractmethod
    def sort(self, result: Result, *, keys: Sequence[tuple[str, bool]]) -> Result:
        """Sort by `keys` (column, descending) pairs; later keys break ties of earlier keys."""

    @abstractmethod
    def derive(self, result: Result, *, columns: Mapping[str, str]) -> Result:
        """Add engine-computed columns to `result`.

        `columns` maps new output column name -> a safe, validated DSL
        expression string (see `ibda.analytics.expr`) over `result.schema`'s
        columns — NEVER a raw engine formula. Returns a new live `Result`;
        `result` itself is unchanged. This is the ONLY op that adds an
        arbitrary caller-defined column (as opposed to `group_by`'s fixed
        aggregation functions or `as_of_join`'s borrowed columns).
        """

    @abstractmethod
    def register_derived(
        self, name: str, *, base: Result, columns: Mapping[str, str]
    ) -> Result:
        """Register a derived table (base + `derive(base, columns=columns)`) under `name`.

        Once registered, `table(name)` resolves it alongside canonical tables,
        and every caller resolving it by name shares the SAME underlying live
        computation (one engine-side `derive`, not one per subscriber).

        Raises:
            ValueError: If `name` collides with a canonical table name or an
                already-registered name.
        """

    def derived_schema(self, name: str) -> Schema | None:
        """Return the schema of a table registered via `register_derived`, else None.

        A pure, non-raising registry lookup — it touches no engine and runs nothing
        caller-supplied, so it is safe to call ahead of the query-validation gate to
        resolve a derived table's name and columns. Canonical tables are NOT reported
        here (they live in `ibda.schema.ALL`); this covers only names created this
        session via `register_derived`. The base ABC has no registry, so it returns
        None; an adapter that supports `register_derived` overrides this.
        """
        return None

    # --- terminal operations, called via Result ------------------------------

    @abstractmethod
    def _snapshot(self, result: Result) -> pa.Table:
        """Materialize `result` as an Arrow table. Invoked by Result.snapshot()."""

    @abstractmethod
    def _subscribe(self, result: Result, callback: DeltaCallback) -> Stream:
        """Subscribe to `result`. Invoked by Result.subscribe()."""
