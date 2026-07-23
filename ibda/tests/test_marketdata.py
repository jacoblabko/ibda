"""Pure tests for ibda.adapters.ibkr.marketdata (snapshot_quote / snapshot_trades) — no JVM, no TWS.

A fake session stubs get_registered_contract / request_market_data /
request_tick_data_historical / the raw client, and an injected snapshot_rows
feeds canned ticks_price/ticks_size/ticks_trade rows through a real
IbkrSupervisor (so snapshot_raw_rows works end-to-end). snapshot_trades imports
deephaven_ib (for TickDataType) which needs a JVM, so the tests fake that module.
"""
from __future__ import annotations

import re
import sys
import types
from typing import Any

import pytest

from ibda.adapters.ibkr.marketdata import snapshot_quote, snapshot_trades
from ibda.adapters.ibkr.supervisor import IbkrSupervisor


def _to_int_local(value: Any) -> int | None:
    """Local helper: coerce to int for test predicate evaluation."""
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _apply_int_eq_predicate(
    rows: list[dict[str, Any]], predicate: str
) -> list[dict[str, Any]]:
    """Evaluate a simple ``Column == int_value`` filter for test stubs."""
    m = re.match(r"^(\w+)\s*==\s*(-?\d+)$", predicate.strip())
    if not m:
        return rows
    col, val_str = m.group(1), m.group(2)
    val = int(val_str)
    return [r for r in rows if _to_int_local(r.get(col)) == val]


class _FakeDuration:
    """Minimal stand-in for ``dhib.Duration``.  Stores the raw IB duration string."""

    def __init__(self, value: str) -> None:
        self.value = value


class _FakeBarSize:
    """Minimal stand-in for ``dhib.BarSize``.

    Constructed by value (like the real Enum reverse-lookup).  Raises ``ValueError``
    for unrecognised strings — mirroring ``dhib.BarSize("unknown")``.
    """

    _KNOWN: frozenset[str] = frozenset(
        ["1 day", "1 hour", "30 mins", "15 mins", "5 mins", "1 min",
         "30 secs", "5 secs", "1 secs"]
    )

    def __init__(self, value: str) -> None:
        if value not in self._KNOWN:
            raise ValueError(f"'{value}' is not a valid BarSize")
        self.value = value


def _fake_deephaven_ib(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a fake ``deephaven_ib`` so lazy imports work without a JVM."""
    fake = types.ModuleType("deephaven_ib")
    fake.TickDataType = types.SimpleNamespace(LAST="Last")  # type: ignore[attr-defined]
    fake.BarDataType = types.SimpleNamespace(TRADES="TRADES")  # type: ignore[attr-defined]
    fake.MarketDataType = types.SimpleNamespace(FROZEN="FROZEN")  # type: ignore[attr-defined]
    fake.Duration = _FakeDuration  # type: ignore[attr-defined]
    fake.BarSize = _FakeBarSize  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "deephaven_ib", fake)


class _FakeClient:
    def __init__(self) -> None:
        self.md_type: int | None = None
        self.raw_mkt_data_calls: list[dict[str, Any]] = []
        self.raw_hist_tick_calls: list[dict[str, Any]] = []
        self.raw_hist_data_calls: list[dict[str, Any]] = []

    def reqMarketDataType(self, market_data_type: int) -> None:
        self.md_type = market_data_type

    def reqMktData(
        self,
        req_id: int,
        contract: Any,
        generic_ticks: str,
        snapshot: bool,
        regulatory: bool,
        mkt_data_options: Any,
    ) -> None:
        self.raw_mkt_data_calls.append(
            {"req_id": req_id, "conId": contract.conId, "snapshot": snapshot}
        )

    def reqHistoricalTicks(
        self,
        req_id: int,
        contract: Any,
        start_date_time: str,
        end_date_time: str,
        number_of_ticks: int,
        what_to_show: str,
        use_rth: int,
        ignore_size: bool,
        misc_options: Any,
    ) -> None:
        self.raw_hist_tick_calls.append(
            {"req_id": req_id, "conId": contract.conId, "n": number_of_ticks}
        )

    def reqHistoricalData(
        self,
        req_id: int,
        contract: Any,
        end_date_time: str,
        duration_str: str,
        bar_size_setting: str,
        what_to_show: str,
        use_rth: int,
        format_date: int,
        keep_up_to_date: bool,
        chart_options: Any,
    ) -> None:
        self.raw_hist_data_calls.append(
            {
                "req_id": req_id,
                "conId": contract.conId,
                "duration": duration_str,
                "bar_size": bar_size_setting,
            }
        )


class _StubContract:
    def __init__(self, conid: int) -> None:
        self.conId = conid


class _StubContractDetails:
    def __init__(self, conid: int) -> None:
        self.contract = _StubContract(conid)


class _StubRegisteredContract:
    def __init__(self, conid: int) -> None:
        self.contract_details = [_StubContractDetails(conid)]


class _FakeSession:
    def __init__(self, tables: dict[str, object], *, conid: int, resolve_error: str | None) -> None:
        self.tables = tables
        self._client = _FakeClient()
        self._conid = conid
        self._resolve_error = resolve_error
        self.md_calls: list[bool] = []
        self.hist_calls: list[int] = []
        self.registered: list[Any] = []
        self.bars_calls: list[Any] = []

    def get_registered_contract(self, contract: Any) -> _StubRegisteredContract:
        self.registered.append(contract)
        if self._resolve_error is not None:
            raise Exception(self._resolve_error)
        return _StubRegisteredContract(self._conid)

    def request_bars_realtime(
        self, rc: Any, *, bar_type: Any, bar_size: int, market_data_type: Any
    ) -> None:
        self.bars_calls.append((rc, bar_size))

    def request_market_data(self, contract: Any, snapshot: bool = False) -> list[Any]:
        self.md_calls.append(snapshot)
        return []

    def request_tick_data_historical(
        self, contract: Any, tick_type: Any, number_of_ticks: int = 0
    ) -> list[Any]:
        self.hist_calls.append(number_of_ticks)
        return []

    def is_connected(self) -> bool:
        return True


def _supervisor(
    price_rows: list[dict[str, Any]],
    size_rows: list[dict[str, Any]] | None = None,
    *,
    conid: int = 12345,
    resolve_error: str | None = None,
    trade_rows: list[dict[str, Any]] | None = None,
) -> IbkrSupervisor:
    fake_tables: dict[str, object] = {
        "ticks_price": object(), "ticks_size": object(), "ticks_trade": object(),
        "bars_historical": object(),
    }
    session = _FakeSession(fake_tables, conid=conid, resolve_error=resolve_error)
    data: dict[str, list[dict[str, Any]]] = {
        "ticks_price": price_rows,
        "ticks_size": size_rows or [],
        "ticks_trade": trade_rows or [],
        "bars_historical": [],
    }

    def snapshot_rows(raw_table: Any) -> list[dict[str, Any]]:
        for name, obj in session.tables.items():
            if obj is raw_table:
                return data.get(name, [])
        return []

    def snapshot_rows_where(raw_table: Any, predicate: str) -> list[dict[str, Any]]:
        # Evaluate "Column == int_value" predicates in Python for test isolation.
        return _apply_int_eq_predicate(snapshot_rows(raw_table), predicate)

    sup = IbkrSupervisor(
        snapshot_rows_fn=snapshot_rows,
        snapshot_rows_where_fn=snapshot_rows_where,
    )
    sup._session = session
    return sup


def _price(tick_type: str, price: float, conid: int = 12345) -> dict[str, Any]:
    return {"ContractId": conid, "TickType": tick_type, "Price": price,
            "ReceiveTime": "2026-06-12T14:30:00Z"}


def _size(tick_type: str, size: float, conid: int = 12345) -> dict[str, Any]:
    return {"ContractId": conid, "TickType": tick_type, "Size": size}


def test_snapshot_quote_happy_path() -> None:
    sup = _supervisor(
        [_price("LAST", 150.0), _price("BID", 149.9), _price("ASK", 150.1)],
        [_size("BID_SIZE", 80.0), _size("ASK_SIZE", 120.0), _size("LAST_SIZE", 5.0)],
    )
    out = snapshot_quote(sup, "aapl", timeout_s=1.0, poll_interval_s=0.01)
    assert out["symbol"] == "AAPL"
    assert out["conId"] == 12345
    assert out["last"] == 150.0
    assert out["bid"] == 149.9
    assert out["ask"] == 150.1
    assert out["bid_size"] == 80.0
    assert out["ask_size"] == 120.0
    assert "error" not in out


def test_snapshot_quote_accepts_delayed_tick_types() -> None:
    sup = _supervisor(
        [_price("DELAYED_LAST", 99.0), _price("DELAYED_BID", 98.5), _price("DELAYED_ASK", 99.5)],
    )
    out = snapshot_quote(sup, "MSFT", timeout_s=1.0, poll_interval_s=0.01)
    assert out["last"] == 99.0
    assert out["bid"] == 98.5
    assert out["ask"] == 99.5


def test_snapshot_quote_timeout_when_no_ticks() -> None:
    sup = _supervisor([])
    out = snapshot_quote(sup, "AAPL", timeout_s=0.05, poll_interval_s=0.01)
    assert out["error"] == "quote_timeout"
    assert out["conId"] == 12345


def test_snapshot_quote_symbol_not_found() -> None:
    sup = _supervisor([], resolve_error="No security definition has been found")
    out = snapshot_quote(sup, "ZZZZ", timeout_s=0.05, poll_interval_s=0.01)
    assert out["error"] == "symbol_not_found"
    assert "ZZZZ" in out["symbol"]


def test_snapshot_quote_empty_symbol() -> None:
    sup = _supervisor([])
    out = snapshot_quote(sup, "   ", timeout_s=0.05, poll_interval_s=0.01)
    assert out["error"] == "symbol_not_found"


def test_snapshot_quote_filters_by_conid() -> None:
    # A tick for a different contract must be ignored.
    sup = _supervisor(
        [_price("LAST", 1.0, conid=99999), _price("LAST", 150.0, conid=12345)],
    )
    out = snapshot_quote(sup, "AAPL", timeout_s=1.0, poll_interval_s=0.01)
    assert out["last"] == 150.0


def test_snapshot_quote_sets_delayed_frozen_and_requests_snapshot() -> None:
    sup = _supervisor([_price("LAST", 150.0)])
    snapshot_quote(sup, "AAPL", timeout_s=1.0, poll_interval_s=0.01)
    session: Any = sup._session
    assert session._client.md_type == 4  # DELAYED_FROZEN
    assert session.md_calls == [True]  # snapshot=True


# -- snapshot_trades ---------------------------------------------------------


def _trade(price: float, size: float, conid: int = 12345) -> dict[str, Any]:
    return {"ContractId": conid, "Price": price, "Size": size,
            "Timestamp": "2026-06-12T15:30:00Z"}


def test_snapshot_trades_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_deephaven_ib(monkeypatch)
    sup = _supervisor([], trade_rows=[_trade(291.5, 100.0), _trade(291.6, 50.0)])
    out = snapshot_trades(sup, "aapl", count=10, timeout_s=1.0, poll_interval_s=0.01)
    assert out["symbol"] == "AAPL"
    assert out["conId"] == 12345
    assert out["count"] == 2
    assert out["trades"][0]["price"] == 291.5
    assert out["trades"][1]["size"] == 50.0
    assert "error" not in out


def test_snapshot_trades_respects_count(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_deephaven_ib(monkeypatch)
    sup = _supervisor([], trade_rows=[_trade(1.0, 1.0), _trade(2.0, 1.0), _trade(3.0, 1.0)])
    out = snapshot_trades(sup, "AAPL", count=2, timeout_s=1.0, poll_interval_s=0.01)
    assert out["count"] == 2
    assert out["trades"][-1]["price"] == 3.0  # most-recent kept


def test_snapshot_trades_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_deephaven_ib(monkeypatch)
    sup = _supervisor([], trade_rows=[])
    out = snapshot_trades(sup, "AAPL", timeout_s=0.05, poll_interval_s=0.01)
    assert out["error"] == "trade_timeout"


def test_snapshot_trades_symbol_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_deephaven_ib(monkeypatch)
    sup = _supervisor([], trade_rows=[], resolve_error="No security definition")
    out = snapshot_trades(sup, "ZZZZ", timeout_s=0.05, poll_interval_s=0.01)
    assert out["error"] == "symbol_not_found"


def test_snapshot_trades_filters_by_conid(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_deephaven_ib(monkeypatch)
    sup = _supervisor(
        [], trade_rows=[_trade(9.9, 1.0, conid=999), _trade(291.5, 100.0, conid=12345)]
    )
    out = snapshot_trades(sup, "AAPL", timeout_s=1.0, poll_interval_s=0.01)
    assert out["count"] == 1
    assert out["trades"][0]["price"] == 291.5


def test_subscribe_bars_builds_contract_with_conid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: the bars path must pass a Contract (conId set), not a contract_id kwarg.

    The old code called get_registered_contract(contract_id=con_id); that raised
    (unexpected kwarg) and was swallowed by the best-effort except, so bars never
    subscribed. This asserts a real Contract reaches the session and bars are requested.
    """
    _fake_deephaven_ib(monkeypatch)
    from ibda.adapters.ibkr.marketdata import subscribe_bars
    from ibda.adapters.ibkr.pacing import PacingGovernor

    sup = _supervisor([])
    session: Any = sup._session
    subscribe_bars(sup, [12345], PacingGovernor())

    assert session.registered, "get_registered_contract was not called with a Contract"
    assert session.registered[0].conId == 12345
    assert session.bars_calls, "request_bars_realtime was not reached"


# -- historical_bars ---------------------------------------------------------


def test_historical_bars_builds_canonical_table(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_deephaven_ib(monkeypatch)
    from ibda.adapters.ibkr import marketdata
    from ibda.schema import BAR

    class _FakeSession:
        def get_registered_contract(self, c: Any) -> object:
            return object()

        def request_bars_historical(self, *a: Any, **k: Any) -> None:
            return None

    class _FakeSup:
        _session = _FakeSession()

        def snapshot_raw_rows(self, name: str) -> list[dict[str, Any]]:
            assert name == "bars_historical"
            import datetime as dt

            base = dt.datetime(2026, 1, 2, tzinfo=dt.timezone.utc)
            return [{"ContractId": 1, "Timestamp": base, "Open": 1.0, "High": 1.0,
                     "Low": 1.0, "Close": 1.0, "Volume": 10.0}]

        def snapshot_raw_rows_where(self, name: str, predicate: str) -> list[dict[str, Any]]:
            # Test stub: delegate to snapshot_raw_rows (engine filter is verified by JVM tests).
            return self.snapshot_raw_rows(name)

    monkeypatch.setattr(marketdata, "_conid_of", lambda rc: 1)
    monkeypatch.setattr(marketdata, "_set_delayed_frozen", lambda sup: None)
    table = marketdata.historical_bars(_FakeSup(), "QQQ", timeout_s=0.1)  # type: ignore[arg-type]
    assert table.schema.equals(BAR.to_arrow_schema())
    assert table.num_rows == 1
    assert table.column("Sym").to_pylist() == ["QQQ"]


def test_historical_bars_empty_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_deephaven_ib(monkeypatch)
    from ibda.adapters.ibkr import marketdata
    from ibda.schema import BAR

    class _FakeSession:
        def get_registered_contract(self, c: Any) -> object:
            return object()

        def request_bars_historical(self, *a: Any, **k: Any) -> None:
            return None

    class _FakeSup:
        _session = _FakeSession()

        def snapshot_raw_rows(self, name: str) -> list[dict[str, Any]]:
            return []

        def snapshot_raw_rows_where(self, name: str, predicate: str) -> list[dict[str, Any]]:
            return []

    monkeypatch.setattr(marketdata, "_conid_of", lambda rc: 1)
    monkeypatch.setattr(marketdata, "_set_delayed_frozen", lambda sup: None)
    table = marketdata.historical_bars(_FakeSup(), "QQQ", timeout_s=0.1)  # type: ignore[arg-type]
    assert table.schema.equals(BAR.to_arrow_schema())
    assert table.num_rows == 0


def test_historical_bars_passes_objects_not_strings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: duration and bar_size must arrive as .value objects, never raw str."""
    _fake_deephaven_ib(monkeypatch)
    from ibda.adapters.ibkr import marketdata

    captured: dict[str, Any] = {}

    class _FakeSession:
        def get_registered_contract(self, c: Any) -> object:
            return object()

        def request_bars_historical(self, rc: Any, **k: Any) -> None:
            captured.update(k)

    class _FakeSup:
        _session = _FakeSession()

        def snapshot_raw_rows(self, name: str) -> list[dict[str, Any]]:
            return []

    monkeypatch.setattr(marketdata, "_conid_of", lambda rc: None)
    monkeypatch.setattr(marketdata, "_set_delayed_frozen", lambda sup: None)
    marketdata.historical_bars(
        _FakeSup(), "SPY",  # type: ignore[arg-type]
        duration="1 Y", bar_size="1 day", timeout_s=0.05,
    )

    assert "duration" in captured, "duration kwarg not passed to request_bars_historical"
    assert "bar_size" in captured, "bar_size kwarg not passed to request_bars_historical"
    # Must NOT be plain strings — must be wrapper objects exposing .value
    assert not isinstance(captured["duration"], str), "duration was passed as raw str"
    assert not isinstance(captured["bar_size"], str), "bar_size was passed as raw str"
    assert captured["duration"].value == "1 Y"
    assert captured["bar_size"].value == "1 day"


def test_historical_bars_keep_up_to_date_defaults_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default (no keep_up_to_date kwarg) must forward True to request_bars_historical --
    preserving this function's long-standing implicit-True behavior for callers
    that rely on a ticking historical-bars feed (e.g. a live VaR calculation)."""
    _fake_deephaven_ib(monkeypatch)
    from ibda.adapters.ibkr import marketdata

    captured: dict[str, Any] = {}

    class _FakeSession:
        def get_registered_contract(self, c: Any) -> object:
            return object()

        def request_bars_historical(self, rc: Any, **k: Any) -> None:
            captured.update(k)

    class _FakeSup:
        _session = _FakeSession()

        def snapshot_raw_rows(self, name: str) -> list[dict[str, Any]]:
            return []

    monkeypatch.setattr(marketdata, "_conid_of", lambda rc: None)
    monkeypatch.setattr(marketdata, "_set_delayed_frozen", lambda sup: None)
    marketdata.historical_bars(_FakeSup(), "SPY", timeout_s=0.05)  # type: ignore[arg-type]

    assert captured["keep_up_to_date"] is True


def test_historical_bars_keep_up_to_date_false_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    """keep_up_to_date=False must be forwarded verbatim to request_bars_historical."""
    _fake_deephaven_ib(monkeypatch)
    from ibda.adapters.ibkr import marketdata

    captured: dict[str, Any] = {}

    class _FakeSession:
        def get_registered_contract(self, c: Any) -> object:
            return object()

        def request_bars_historical(self, rc: Any, **k: Any) -> None:
            captured.update(k)

    class _FakeSup:
        _session = _FakeSession()

        def snapshot_raw_rows(self, name: str) -> list[dict[str, Any]]:
            return []

    monkeypatch.setattr(marketdata, "_conid_of", lambda rc: None)
    monkeypatch.setattr(marketdata, "_set_delayed_frozen", lambda sup: None)
    marketdata.historical_bars(
        _FakeSup(), "SPY", timeout_s=0.05, keep_up_to_date=False,  # type: ignore[arg-type]
    )

    assert captured["keep_up_to_date"] is False


def test_historical_bars_bad_bar_size_returns_empty_gracefully(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unsupported bar_size string must return an empty bar table and log a warning."""
    import logging
    _fake_deephaven_ib(monkeypatch)
    from ibda.adapters.ibkr import marketdata
    from ibda.schema import BAR

    class _FakeSession:
        def get_registered_contract(self, c: Any) -> object:
            return object()

        def request_bars_historical(self, *a: Any, **k: Any) -> None:
            return None  # should never be reached

    class _FakeSup:
        _session = _FakeSession()

        def snapshot_raw_rows(self, name: str) -> list[dict[str, Any]]:
            return []

    monkeypatch.setattr(marketdata, "_conid_of", lambda rc: None)
    monkeypatch.setattr(marketdata, "_set_delayed_frozen", lambda sup: None)

    with caplog.at_level(logging.WARNING, logger="ibda.adapters.ibkr.marketdata"):
        table = marketdata.historical_bars(
            _FakeSup(), "AAPL",  # type: ignore[arg-type]
            bar_size="3 fortnights", timeout_s=0.05,
        )

    assert table.schema.equals(BAR.to_arrow_schema())
    assert table.num_rows == 0
    assert any("request_bars_historical failed" in r.message for r in caplog.records), (
        "Expected a warning log for bad bar_size; got: " + str([r.message for r in caplog.records])
    )


# ---------------------------------------------------------------------------
# conid_hint fast path (Fix 3)
# ---------------------------------------------------------------------------


def test_snapshot_quote_conid_hint_skips_get_registered_contract() -> None:
    """When conid_hint is given, get_registered_contract must NOT be called."""
    sup = _supervisor(
        [_price("LAST", 150.0), _price("BID", 149.9), _price("ASK", 150.1)],
        conid=12345,
    )
    out = snapshot_quote(sup, "AAPL", conid_hint=12345, timeout_s=1.0, poll_interval_s=0.01)

    session: Any = sup._session
    # get_registered_contract must have been skipped entirely
    assert session.registered == [], (
        f"get_registered_contract was called {len(session.registered)} time(s) "
        "even though conid_hint was supplied — fast path not engaged"
    )
    # The raw client's reqMktData must have been called with snapshot=True
    assert session._client.raw_mkt_data_calls, "reqMktData was not called on the raw client"
    assert session._client.raw_mkt_data_calls[0]["snapshot"] is True
    assert session._client.raw_mkt_data_calls[0]["conId"] == 12345
    # Quote should still work end-to-end
    assert out.get("error") is None
    assert out["symbol"] == "AAPL"
    assert out["conId"] == 12345
    assert out["last"] == 150.0


def test_snapshot_quote_no_hint_uses_get_registered_contract() -> None:
    """Without conid_hint, the original get_registered_contract path is used."""
    sup = _supervisor([_price("LAST", 150.0)], conid=12345)
    snapshot_quote(sup, "AAPL", timeout_s=1.0, poll_interval_s=0.01)
    session: Any = sup._session
    # Slow path: get_registered_contract must have been called
    assert session.registered, "get_registered_contract was not called on the slow path"
    # request_market_data (not raw reqMktData) must have been used
    assert session.md_calls == [True]
    assert session._client.raw_mkt_data_calls == []


def test_snapshot_trades_conid_hint_skips_get_registered_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When conid_hint is given to snapshot_trades, get_registered_contract must NOT be called."""
    _fake_deephaven_ib(monkeypatch)
    sup = _supervisor([], trade_rows=[_trade(291.5, 100.0)], conid=12345)
    out = snapshot_trades(
        sup, "AAPL", conid_hint=12345, timeout_s=1.0, poll_interval_s=0.01
    )

    session: Any = sup._session
    assert session.registered == [], (
        f"get_registered_contract was called {len(session.registered)} time(s) "
        "even though conid_hint was supplied"
    )
    assert session._client.raw_hist_tick_calls, "reqHistoricalTicks was not called"
    assert session._client.raw_hist_tick_calls[0]["conId"] == 12345
    assert out.get("error") is None
    assert out["symbol"] == "AAPL"
    assert out["count"] == 1


def test_snapshot_trades_no_hint_uses_get_registered_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without conid_hint, snapshot_trades uses get_registered_contract as before."""
    _fake_deephaven_ib(monkeypatch)
    sup = _supervisor([], trade_rows=[_trade(291.5, 100.0)], conid=12345)
    snapshot_trades(sup, "AAPL", timeout_s=1.0, poll_interval_s=0.01)
    session: Any = sup._session
    assert session.registered, "get_registered_contract was not called on the slow path"
    assert session.hist_calls == [30]  # default count=30
    assert session._client.raw_hist_tick_calls == []


# ---------------------------------------------------------------------------
# historical_bars always uses request_bars_historical (table-populating path)
# ---------------------------------------------------------------------------


def test_historical_bars_always_uses_get_registered_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """historical_bars must always call get_registered_contract and request_bars_historical.

    The conid_hint fast-path (reqHistoricalData raw) was reverted because it did not
    populate the bars_historical table that the poller reads.  The only correct path
    is session.request_bars_historical which goes through deephaven-ib's table writer.
    """
    _fake_deephaven_ib(monkeypatch)
    from ibda.adapters.ibkr import marketdata

    class _FakeSession2:
        def __init__(self) -> None:
            self.registered: list[Any] = []
            self._client = _FakeClient()

        def get_registered_contract(self, c: Any) -> Any:
            self.registered.append(c)
            return object()

        def request_bars_historical(self, *a: Any, **k: Any) -> None:
            pass

    class _FakeSup2:
        _session = _FakeSession2()

        def snapshot_raw_rows(self, name: str) -> list[dict[str, Any]]:
            return []

    monkeypatch.setattr(marketdata, "_conid_of", lambda rc: None)
    monkeypatch.setattr(marketdata, "_set_delayed_frozen", lambda sup: None)
    marketdata.historical_bars(
        _FakeSup2(),  # type: ignore[arg-type]
        "AAPL",
        timeout_s=0.05,
    )
    session: Any = _FakeSup2._session
    assert session.registered, "get_registered_contract was not called"
    assert session._client.raw_hist_data_calls == [], (
        "reqHistoricalData (raw) must never be called — only request_bars_historical populates the table"
    )


# ---------------------------------------------------------------------------
# historical_bars must scope its poll to its OWN RequestId, not ContractId
# alone (ContractId-only unions bars from any other duration/bar_size request
# already outstanding on the same contract).
# ---------------------------------------------------------------------------


def test_historical_bars_scopes_by_request_id_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulates a bars_historical table carrying rows from an EARLIER daily-bar
    request (RequestId=10) and THIS call's 5-min request (RequestId=20), both on
    the same ContractId — asserts only RequestId=20's rows come back, and that the
    poll predicate is scoped by RequestId (not ContractId alone)."""
    import datetime as dt

    _fake_deephaven_ib(monkeypatch)
    from ibda.adapters.ibkr import marketdata
    from ibda.schema import BAR

    all_rows = [
        {"RequestId": 10, "ContractId": 1, "Timestamp": dt.datetime(2026, 7, 8, tzinfo=dt.timezone.utc),
         "Open": 100.0, "High": 100.0, "Low": 100.0, "Close": 100.0, "Volume": 10.0},
        {"RequestId": 10, "ContractId": 1, "Timestamp": dt.datetime(2026, 7, 9, tzinfo=dt.timezone.utc),
         "Open": 101.0, "High": 101.0, "Low": 101.0, "Close": 101.0, "Volume": 10.0},
        {"RequestId": 20, "ContractId": 1, "Timestamp": dt.datetime(2026, 7, 9, 13, 30, tzinfo=dt.timezone.utc),
         "Open": 102.0, "High": 102.0, "Low": 102.0, "Close": 102.0, "Volume": 5.0},
        {"RequestId": 20, "ContractId": 1, "Timestamp": dt.datetime(2026, 7, 9, 13, 35, tzinfo=dt.timezone.utc),
         "Open": 102.5, "High": 102.5, "Low": 102.5, "Close": 102.5, "Volume": 5.0},
    ]

    class _FakeSession:
        def get_registered_contract(self, c: Any) -> object:
            return object()

        def request_bars_historical(self, rc: Any, **kwargs: Any) -> list[Any]:
            return [types.SimpleNamespace(request_id=20)]

    class _FakeSup:
        _session = _FakeSession()

        def snapshot_raw_rows(self, name: str) -> list[dict[str, Any]]:
            return all_rows

        def snapshot_raw_rows_where(self, name: str, predicate: str) -> list[dict[str, Any]]:
            assert predicate.strip().startswith("RequestId == 20"), (
                f"expected the poll scoped to RequestId == 20 (this call's own "
                f"request), not ContractId alone; got predicate={predicate!r}"
            )
            return [r for r in all_rows if r["RequestId"] == 20]

    monkeypatch.setattr(marketdata, "_conid_of", lambda rc: 1)
    monkeypatch.setattr(marketdata, "_set_delayed_frozen", lambda sup: None)

    table = marketdata.historical_bars(
        _FakeSup(),  # type: ignore[arg-type]
        "AAPL", duration="1 D", bar_size="5 mins", timeout_s=0.5, poll_interval_s=0.01,
    )

    assert table.schema.equals(BAR.to_arrow_schema())
    assert table.num_rows == 2, (
        f"expected 2 rows (RequestId=20 only); got {table.num_rows} — "
        "ContractId-only filtering would union in the 2 daily-bar rows (RequestId=10)"
    )
    assert table.column("Close").to_pylist() == [102.0, 102.5]


def test_historical_bars_falls_back_to_contract_id_when_request_bars_historical_returns_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If request_bars_historical fails / returns no Request objects, historical_bars
    falls back to the pre-fix ContractId-only filter (never an unscoped full scan)
    when the conId is known."""
    _fake_deephaven_ib(monkeypatch)
    from ibda.adapters.ibkr import marketdata

    predicates: list[str] = []

    class _FakeSession:
        def get_registered_contract(self, c: Any) -> object:
            return object()

        def request_bars_historical(self, rc: Any, **kwargs: Any) -> list[Any]:
            return []

    class _FakeSup:
        _session = _FakeSession()

        def snapshot_raw_rows_where(self, name: str, predicate: str) -> list[dict[str, Any]]:
            predicates.append(predicate)
            return []

    monkeypatch.setattr(marketdata, "_conid_of", lambda rc: 7)
    monkeypatch.setattr(marketdata, "_set_delayed_frozen", lambda sup: None)

    marketdata.historical_bars(
        _FakeSup(), "AAPL", timeout_s=0.05, poll_interval_s=0.01,  # type: ignore[arg-type]
    )

    assert predicates, "expected at least one snapshot_raw_rows_where call"
    assert predicates[0] == "ContractId == 7"


# ---------------------------------------------------------------------------
# Live-ticking intraday bars — request_intraday_bars_ticking
# ---------------------------------------------------------------------------


def test_request_intraday_bars_ticking_issues_keep_up_to_date_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Must request 5-min TRADES bars with keep_up_to_date=True and return the
    issued RequestId (the scoping key the caller filters bars_historical on)."""
    _fake_deephaven_ib(monkeypatch)
    from ibda.adapters.ibkr import marketdata

    captured: dict[str, Any] = {}

    class _FakeSession:
        def get_registered_contract(self, c: Any) -> object:
            return object()

        def request_bars_historical(self, rc: Any, **kwargs: Any) -> list[Any]:
            captured.update(kwargs)
            return [types.SimpleNamespace(request_id=42)]

    class _FakeSup:
        _session = _FakeSession()

    monkeypatch.setattr(marketdata, "_set_delayed_frozen", lambda sup: None)
    request_id = marketdata.request_intraday_bars_ticking(_FakeSup(), "aapl")  # type: ignore[arg-type]

    assert request_id == 42
    assert captured["keep_up_to_date"] is True
    assert captured["duration"].value == "1 D"
    assert captured["bar_size"].value == "5 mins"


def test_request_intraday_bars_ticking_returns_none_on_resolution_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_deephaven_ib(monkeypatch)
    from ibda.adapters.ibkr import marketdata

    class _FakeSession:
        def get_registered_contract(self, c: Any) -> object:
            raise RuntimeError("No security definition has been found")

    class _FakeSup:
        _session = _FakeSession()

    monkeypatch.setattr(marketdata, "_set_delayed_frozen", lambda sup: None)
    assert marketdata.request_intraday_bars_ticking(_FakeSup(), "ZZZZ") is None  # type: ignore[arg-type]


def test_request_intraday_bars_ticking_returns_none_when_no_requests_issued(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_deephaven_ib(monkeypatch)
    from ibda.adapters.ibkr import marketdata

    class _FakeSession:
        def get_registered_contract(self, c: Any) -> object:
            return object()

        def request_bars_historical(self, rc: Any, **kwargs: Any) -> list[Any]:
            return []

    class _FakeSup:
        _session = _FakeSession()

    monkeypatch.setattr(marketdata, "_set_delayed_frozen", lambda sup: None)
    assert marketdata.request_intraday_bars_ticking(_FakeSup(), "AAPL") is None  # type: ignore[arg-type]
