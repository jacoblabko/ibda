from __future__ import annotations

from collections.abc import Mapping, Sequence

import pyarrow as pa

from ibda.result import Result, Stream
from ibda.schema import QUOTE
from ibda.watch import watch


class _FakePort:
    """Records filter/subscribe calls; lets the test drive deltas manually."""

    def __init__(self) -> None:
        self.filtered_with: list[str] = []
        self._subscriber: object = None  # the callback subscribe() registered
        self.cancelled = False

    def filter(self, result: Result, predicate: str) -> Result:
        self.filtered_with.append(predicate)
        return Result(handle=("filtered", predicate), schema=result.schema, port=self)

    def _snapshot(self, result: Result) -> pa.Table:
        return pa.table({c.name: [] for c in QUOTE.columns}, schema=QUOTE.to_arrow_schema())

    def _subscribe(self, result: Result, callback: object) -> Stream:
        self._subscriber = callback
        return Stream(cancel=self._cancel)

    def _cancel(self) -> None:
        self.cancelled = True

    def group_by(self, result: Result, *, by: Sequence[str], aggs: dict[str, str]) -> Result:
        return Result(handle=("group_by", by), schema=result.schema, port=self)

    def derive(self, result: Result, *, columns: Mapping[str, str]) -> Result:
        return Result(handle=("derive", tuple(columns.items())), schema=result.schema, port=self)

    # test helper: simulate the engine delivering a delta
    def deliver(self, tbl: pa.Table) -> None:
        assert callable(self._subscriber)
        self._subscriber(tbl)


def _result(port: _FakePort) -> Result:
    return Result(handle=object(), schema=QUOTE, port=port)


def _delta() -> pa.Table:
    return pa.table({c.name: [None] for c in QUOTE.columns}, schema=QUOTE.to_arrow_schema())


def test_watch_without_predicate_subscribes_directly() -> None:
    port = _FakePort()
    seen: list[pa.Table] = []
    watch(_result(port), seen.append)
    assert port.filtered_with == []          # no predicate -> no filter
    port.deliver(_delta())
    assert len(seen) == 1


def test_watch_with_predicate_filters_first() -> None:
    port = _FakePort()
    watch(_result(port), lambda _t: None, predicate="Last > 150")
    assert port.filtered_with == ["Last > 150"]


def test_watch_once_cancels_after_first_delivery() -> None:
    port = _FakePort()
    seen: list[pa.Table] = []
    watch(_result(port), seen.append, once=True)
    port.deliver(_delta())
    assert len(seen) == 1
    assert port.cancelled is True            # auto-cancelled after first hit


def test_watch_returns_cancellable_stream() -> None:
    port = _FakePort()
    stream = watch(_result(port), lambda _t: None)
    stream.cancel()
    assert port.cancelled is True
