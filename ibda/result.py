"""ibda.result — Result/Stream handles returned to consumers.

A Result is the single return type of every query: a consumer either takes a
point-in-time Arrow snapshot or subscribes to a changefeed. The underlying
engine handle is opaque (`object`); consumers never touch it. No engine import.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import pyarrow as pa

from ibda.schema import Schema

# A delta callback receives an Arrow table of changed (added) rows.
DeltaCallback = Callable[[pa.Table], None]


class _PortLike(Protocol):
    """The slice of DataPort that Result and watch() need — keeps them engine-agnostic.

    ``filter``/``derive`` are included here (not just on DataPort) so that
    ``ibda.watch``/``Result.derive`` can call ``result.port.filter(...)`` /
    ``result.port.derive(...)`` without importing the concrete DataPort ABC,
    preserving the one-way boundary. Any fake port used in tests must implement
    all five methods.
    """

    def filter(self, result: Result, predicate: str) -> Result: ...
    def _snapshot(self, result: Result) -> pa.Table: ...
    def _subscribe(self, result: Result, callback: DeltaCallback) -> Stream: ...
    def group_by(self, result: Result, *, by: Sequence[str], aggs: dict[str, str]) -> Result: ...
    def derive(self, result: Result, *, columns: Mapping[str, str]) -> Result: ...


@dataclass
class Stream:
    """A live subscription handle. Call cancel() to stop receiving deltas."""

    cancel: Callable[[], None]


@dataclass
class Result:
    """The handle returned by every query. Engine-neutral."""

    handle: object          # opaque engine table handle — never inspected by consumers
    schema: Schema
    port: _PortLike

    def snapshot(self) -> pa.Table:
        """Return a point-in-time Apache Arrow table conforming to `schema`."""
        return self.port._snapshot(self)

    def subscribe(self, callback: DeltaCallback) -> Stream:
        """Register `callback`, invoked with an Arrow table of new rows on each update."""
        return self.port._subscribe(self, callback)

    def derive(self, **columns: str) -> Result:
        """Add computed columns via a safe, validated DSL expression per column.

        Sugar over ``port.derive(self, columns=columns)``. Each keyword is the
        new output column name; each value is a DSL expression string over this
        result's existing columns (see ``ibda.analytics.expr`` for the
        allowlist) — e.g. ``result.derive(Doubled="Qty * 2")``. Returns a new
        `Result`; never mutates `self`.
        """
        return self.port.derive(self, columns=columns)
