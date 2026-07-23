from __future__ import annotations

from collections.abc import Mapping, Sequence

import pyarrow as pa

from ibda.result import DeltaCallback, Result, Stream
from ibda.schema import QUOTE


class _FakePort:
    def __init__(self) -> None:
        self.snapshot_calls = 0
        self.derived_with: dict[str, str] | None = None

    def filter(self, result: Result, predicate: str) -> Result:
        return Result(handle=("filtered", predicate), schema=result.schema, port=self)

    def _snapshot(self, result: Result) -> pa.Table:
        self.snapshot_calls += 1
        return pa.table({c.name: [] for c in QUOTE.columns}, schema=QUOTE.to_arrow_schema())

    def _subscribe(self, result: Result, callback: DeltaCallback) -> Stream:
        return Stream(cancel=lambda: None)

    def group_by(self, result: Result, *, by: Sequence[str], aggs: dict[str, str]) -> Result:
        return Result(handle=("group_by", by), schema=result.schema, port=self)

    def derive(self, result: Result, *, columns: Mapping[str, str]) -> Result:
        self.derived_with = dict(columns)
        return Result(handle=("derive", tuple(columns.items())), schema=result.schema, port=self)


def test_result_snapshot_delegates_to_port() -> None:
    port = _FakePort()
    r = Result(handle=object(), schema=QUOTE, port=port)
    tbl = r.snapshot()
    assert port.snapshot_calls == 1
    assert tbl.schema == QUOTE.to_arrow_schema()


def test_result_derive_sugar_delegates_to_port_derive() -> None:
    port = _FakePort()
    r = Result(handle=object(), schema=QUOTE, port=port)
    derived = r.derive(Doubled="Bid * 2")
    assert port.derived_with == {"Doubled": "Bid * 2"}
    assert derived.schema == QUOTE
    assert derived.port is port


def test_stream_cancel_runs_callback() -> None:
    flag = {"cancelled": False}
    s = Stream(cancel=lambda: flag.__setitem__("cancelled", True))
    s.cancel()
    assert flag["cancelled"] is True
