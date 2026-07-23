"""Pure tests for the intraday-bars surface — no JVM, no TWS.

Covers ``IbkrSupervisor.from_session`` plus
``request_intraday_bars`` / ``read_intraday_bars`` in
``ibda.adapters.ibkr.marketdata``.

A fake session records ``request_bars_historical`` calls and hands back
``[_FakeRequest(request_id=N)]`` with an incrementing N; ``get_registered_contract``
returns a fake registered contract with ``is_multi()``/contract-id.  Canned
``bars_historical`` rows tagged with ``RequestId`` are fed through a real
``IbkrSupervisor`` via an injected snapshot helper.  ``request_intraday_bars``
imports ``deephaven_ib`` (BarSize/Duration/BarDataType) which needs a JVM, so
the tests fake that module.
"""
from __future__ import annotations

import datetime as dt
import re
import sys
import types
from typing import Any

import pytest

from ibda.adapters.ibkr.marketdata import (
    read_intraday_bars,
    request_intraday_bars,
)
from ibda.adapters.ibkr.supervisor import IbkrSupervisor
from ibda.schema import BAR


def _fake_deephaven_ib(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a fake `deephaven_ib` so the lazy vendor import works without a JVM."""
    fake = types.ModuleType("deephaven_ib")
    fake.BarSize = types.SimpleNamespace(MIN_1="MIN_1")  # type: ignore[attr-defined]
    fake.BarDataType = types.SimpleNamespace(TRADES="TRADES")  # type: ignore[attr-defined]
    fake.Duration = types.SimpleNamespace(  # type: ignore[attr-defined]
        seconds=lambda n: f"Duration.seconds({n})",
    )
    monkeypatch.setitem(sys.modules, "deephaven_ib", fake)


class _FakeRequest:
    def __init__(self, request_id: int) -> None:
        self.request_id = request_id


class _FakeRegisteredContract:
    def __init__(self, conid: int, *, multi: bool = False) -> None:
        self._conid = conid
        self._multi = multi

    def is_multi(self) -> bool:
        return self._multi


class _FakeSession:
    """Records request_bars_historical calls; hands back incrementing request ids."""

    def __init__(self, *, multi_symbols: frozenset[str] = frozenset()) -> None:
        self.multi_symbols = multi_symbols
        self.registered: list[str] = []
        self.bars_calls: list[dict[str, Any]] = []
        self._next_req_id = 0
        self._next_conid = 1000

    def get_registered_contract(self, contract: Any) -> _FakeRegisteredContract:
        sym = str(contract.symbol)
        self.registered.append(sym)
        self._next_conid += 1
        return _FakeRegisteredContract(
            self._next_conid, multi=sym in self.multi_symbols
        )

    def request_bars_historical(
        self,
        rc: Any,
        *,
        duration: Any,
        bar_size: Any,
        bar_type: Any,
        end: Any,
        keep_up_to_date: bool,
    ) -> list[_FakeRequest]:
        self._next_req_id += 1
        self.bars_calls.append({
            "rc": rc, "duration": duration, "bar_size": bar_size,
            "bar_type": bar_type, "end": end, "keep_up_to_date": keep_up_to_date,
        })
        return [_FakeRequest(self._next_req_id)]

    def is_connected(self) -> bool:
        return True


def _to_int_local(value: Any) -> int | None:
    """Coerce *value* to int; return None on failure."""
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _apply_request_id_predicate(
    rows: list[dict[str, Any]], predicate: str
) -> list[dict[str, Any]]:
    """Simulate engine-side RequestId filtering from a predicate string.

    Extracts all ``RequestId == N`` clauses from *predicate* (handles both
    single-id ``"RequestId == 1"`` and OR-chained ``"(RequestId == 1) ||
    (RequestId == 2)"`` forms) and returns only matching rows.
    """
    ids: set[int] = {int(m) for m in re.findall(r"RequestId\s*==\s*(\d+)", predicate)}
    if not ids:
        return rows
    return [
        r for r in rows
        if (rid := _to_int_local(r.get("RequestId"))) is not None and rid in ids
    ]


def _supervisor_with_bars(
    rows: list[dict[str, Any]],
    *,
    where_calls: list[str] | None = None,
) -> IbkrSupervisor:
    """A supervisor whose snapshot_raw_rows_where('bars_historical', pred) filters *rows*.

    Parameters
    ----------
    rows:
        Canned ``bars_historical`` rows returned from the fake engine.
    where_calls:
        Optional list; each predicate string passed to ``snapshot_raw_rows_where``
        is appended here so tests can assert the WHERE path was taken.
    """

    def snapshot_rows(raw_table: Any) -> list[dict[str, Any]]:
        # Kept for the full-snapshot path used by other supervisor methods.
        return rows

    def snapshot_rows_where(raw_table: Any, predicate: str) -> list[dict[str, Any]]:
        if where_calls is not None:
            where_calls.append(predicate)
        return _apply_request_id_predicate(rows, predicate)

    sup = IbkrSupervisor(
        snapshot_rows_fn=snapshot_rows,
        snapshot_rows_where_fn=snapshot_rows_where,
    )
    sup._session = types.SimpleNamespace(tables={"bars_historical": object()})
    return sup


_TS = dt.datetime(2026, 6, 15, 14, 30, tzinfo=dt.timezone.utc)


def _bar(req_id: int, *, close: float, ts: dt.datetime = _TS) -> dict[str, Any]:
    return {
        "RequestId": req_id, "ContractId": 1, "Timestamp": ts,
        "Open": close, "High": close, "Low": close, "Close": close, "Volume": 100.0,
    }


# -- from_session ------------------------------------------------------------


def test_from_session_wraps_without_connecting() -> None:
    sentinel = object()
    sup = IbkrSupervisor.from_session(sentinel)
    assert sup._session is sentinel
    assert sup._account == ""
    assert sup._last_error == ""


def test_from_session_passes_snapshot_fn() -> None:
    canned: list[dict[str, Any]] = [{"RequestId": 7}]

    def snapshot_rows(raw_table: Any) -> list[dict[str, Any]]:
        return canned

    session = types.SimpleNamespace(tables={"bars_historical": object()})
    sup = IbkrSupervisor.from_session(session, snapshot_rows_fn=snapshot_rows)
    assert sup.snapshot_raw_rows("bars_historical") == canned


# -- request_intraday_bars ---------------------------------------------------


def test_request_intraday_bars_issues_one_per_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_deephaven_ib(monkeypatch)
    session = _FakeSession()
    sup = IbkrSupervisor.from_session(session)

    out = request_intraday_bars(
        sup, ["aapl", "msft"], duration_secs=1800, bar_size="1 min"
    )

    assert out == {"AAPL": 1, "MSFT": 2}
    assert len(session.bars_calls) == 2
    call = session.bars_calls[0]
    assert call["duration"] == "Duration.seconds(1800)"
    assert call["bar_size"] == "MIN_1"
    assert call["bar_type"] == "TRADES"
    assert call["end"] is None
    assert call["keep_up_to_date"] is False


def test_request_intraday_bars_skips_multi_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_deephaven_ib(monkeypatch)
    session = _FakeSession(multi_symbols=frozenset({"BRK"}))
    sup = IbkrSupervisor.from_session(session)

    out = request_intraday_bars(sup, ["AAPL", "BRK", "MSFT"], duration_secs=900)

    assert "BRK" not in out
    assert set(out) == {"AAPL", "MSFT"}
    # The multi-contract symbol must NOT have issued a bars request.
    assert len(session.bars_calls) == 2


def test_request_intraday_bars_reuses_reg_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_deephaven_ib(monkeypatch)
    session = _FakeSession()
    sup = IbkrSupervisor.from_session(session)
    cache: dict[str, Any] = {}

    request_intraday_bars(sup, ["AAPL"], duration_secs=900, reg_cache=cache)
    request_intraday_bars(sup, ["AAPL"], duration_secs=900, reg_cache=cache)

    # Contract registered exactly once across two cycles; both cycles issued.
    assert session.registered == ["AAPL"]
    assert "AAPL" in cache
    assert len(session.bars_calls) == 2


def test_request_intraday_bars_unsupported_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_deephaven_ib(monkeypatch)
    sup = IbkrSupervisor.from_session(_FakeSession())
    with pytest.raises(ValueError, match="unsupported bar_size"):
        request_intraday_bars(sup, ["AAPL"], duration_secs=900, bar_size="5 min")


# -- read_intraday_bars ------------------------------------------------------


def test_read_intraday_bars_isolates_by_request_id() -> None:
    rows = [
        _bar(1, close=150.0),
        _bar(1, close=151.0),
        _bar(2, close=300.0),
    ]
    sup = _supervisor_with_bars(rows)

    out = read_intraday_bars(sup, {"AAPL": 1, "MSFT": 2})

    assert set(out) == {"AAPL", "MSFT"}
    aapl, msft = out["AAPL"], out["MSFT"]
    assert aapl.schema.equals(BAR.to_arrow_schema())
    assert msft.schema.equals(BAR.to_arrow_schema())
    assert aapl.num_rows == 2
    assert msft.num_rows == 1
    assert aapl.column("Close").to_pylist() == [150.0, 151.0]
    assert msft.column("Close").to_pylist() == [300.0]
    assert aapl.column("Sym").to_pylist() == ["AAPL", "AAPL"]


def test_read_intraday_bars_empty_when_no_match() -> None:
    sup = _supervisor_with_bars([_bar(1, close=150.0)])
    out = read_intraday_bars(sup, {"AAPL": 1, "ZZZZ": 99})
    assert out["ZZZZ"].schema.equals(BAR.to_arrow_schema())
    assert out["ZZZZ"].num_rows == 0
    assert out["AAPL"].num_rows == 1


def test_read_intraday_bars_handles_missing_request_id() -> None:
    # A row with no RequestId must not crash and must not match any symbol.
    rows = [{"ContractId": 1, "Timestamp": _TS,
             "Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1.0}]
    sup = _supervisor_with_bars(rows)
    out = read_intraday_bars(sup, {"AAPL": 1})
    assert out["AAPL"].num_rows == 0


def test_read_intraday_bars_uses_engine_where_filter() -> None:
    """read_intraday_bars must call snapshot_raw_rows_where, not snapshot_raw_rows.

    Proves the engine-side filter path is taken: snapshot_rows_where must be
    called exactly once with a predicate that references the target RequestId.
    """
    rows = [_bar(1, close=150.0), _bar(1, close=151.0), _bar(2, close=300.0)]
    where_calls: list[str] = []
    sup = _supervisor_with_bars(rows, where_calls=where_calls)

    out = read_intraday_bars(sup, {"AAPL": 1})

    assert len(where_calls) == 1, "snapshot_raw_rows_where must be called exactly once"
    assert "RequestId" in where_calls[0], "predicate must filter by RequestId"
    assert "1" in where_calls[0], "predicate must reference request_id=1"
    # Result must still be correct (engine filter + Python split = same semantics).
    assert out["AAPL"].num_rows == 2
    assert out["AAPL"].column("Close").to_pylist() == [150.0, 151.0]


def test_read_intraday_bars_multi_id_single_snapshot() -> None:
    """Multiple symbols must produce one combined WHERE, not N per-symbol snapshots.

    The predicate must cover all tracked RequestIds so one engine round-trip
    serves all symbols — verifies the O(1) snapshot-count property.
    """
    rows = [_bar(1, close=150.0), _bar(2, close=300.0), _bar(3, close=500.0)]
    where_calls: list[str] = []
    sup = _supervisor_with_bars(rows, where_calls=where_calls)

    out = read_intraday_bars(sup, {"AAPL": 1, "MSFT": 2})

    assert len(where_calls) == 1, "one combined snapshot, not one per symbol"
    pred = where_calls[0]
    # Both RequestIds must appear in the combined predicate.
    assert "1" in pred and "2" in pred, (
        f"predicate must reference both RequestIds; got {pred!r}"
    )
    # RequestId=3 (not tracked) must not appear.
    assert out["AAPL"].num_rows == 1
    assert out["MSFT"].num_rows == 1


def test_read_intraday_bars_empty_request_ids_skips_snapshot() -> None:
    """Empty request_ids must return an empty dict without issuing any snapshot."""
    where_calls: list[str] = []
    sup = _supervisor_with_bars([], where_calls=where_calls)

    out = read_intraday_bars(sup, {})

    assert out == {}
    assert len(where_calls) == 0, "no snapshot should be issued for empty request_ids"
