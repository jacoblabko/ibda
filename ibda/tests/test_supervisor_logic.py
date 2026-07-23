"""Pure tests for IbkrSupervisor — uses a fake session stub, no JVM.

The fake session simulates the deephaven-ib IbSessionTws interface:
  - .tables: dict mapping table names to small fake objects
  - .is_connected(): bool

snapshot_rows is injected as a pure function returning pre-built row lists,
so no engine (deephaven) import is needed here.

connect() is NOT tested here (it calls deephaven_ib.IbSessionTws + time.sleep).
The connect path is verified by the controller's live TWS script.
"""
from __future__ import annotations

from typing import Any

import pytest

from ibda.adapters.ibkr.supervisor import IbkrSupervisor


# ---------------------------------------------------------------------------
# Fake session + snapshot helper
# ---------------------------------------------------------------------------

class _FakeSession:
    """Minimal fake mimicking the IbSessionTws interface used by the supervisor."""

    def __init__(self, connected: bool = True, tables: dict[str, Any] | None = None) -> None:
        self._connected = connected
        self.tables: dict[str, Any] = tables or {}
        self.disconnect_calls: int = 0

    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self._connected = False


def _make_supervisor(
    tables: dict[str, list[dict[str, Any]]],
    connected: bool = True,
    account: str = "",
) -> IbkrSupervisor:
    """Build a supervisor with a fake session and injected snapshot helper."""

    fake_tables: dict[str, object] = {name: object() for name in tables}
    session = _FakeSession(connected=connected, tables=fake_tables)

    def snapshot_rows(raw_table: Any) -> list[dict[str, Any]]:
        # Look up the fake table object in the session tables to get the name.
        for k, v in session.tables.items():
            if v is raw_table:
                return tables.get(k, [])
        return []

    sup = IbkrSupervisor(snapshot_rows_fn=snapshot_rows)
    sup._session = session  # inject session directly (bypasses connect())
    sup._account = account
    return sup


# ---------------------------------------------------------------------------
# health()
# ---------------------------------------------------------------------------

def test_health_when_disconnected() -> None:
    sup = IbkrSupervisor()
    h = sup.health()
    assert h["connected"] is False
    assert h["position_count"] == 0


def test_health_when_connected_no_positions() -> None:
    sup = _make_supervisor({"accounts_positions": []}, connected=True)
    h = sup.health()
    assert h["connected"] is True
    assert h["position_count"] == 0


def test_health_reflects_position_count() -> None:
    rows = [{"Account": "DU1", "Symbol": f"SYM{i}"} for i in range(42)]
    sup = _make_supervisor({"accounts_positions": rows}, connected=True, account="DU1")
    h = sup.health()
    assert h["connected"] is True
    assert h["position_count"] == 42
    assert h["account"] == "DU1"
    assert h["last_error"] == ""


def test_health_when_session_is_none() -> None:
    sup = IbkrSupervisor()
    sup._session = None
    h = sup.health()
    assert h["connected"] is False


# ---------------------------------------------------------------------------
# is_connected()
# ---------------------------------------------------------------------------

def test_is_connected_false_when_no_session() -> None:
    sup = IbkrSupervisor()
    assert sup.is_connected() is False


def test_is_connected_true_when_session_connected() -> None:
    sup = _make_supervisor({}, connected=True)
    assert sup.is_connected() is True


def test_is_connected_false_when_session_not_connected() -> None:
    sup = _make_supervisor({}, connected=False)
    assert sup.is_connected() is False


# ---------------------------------------------------------------------------
# discover_account()
# ---------------------------------------------------------------------------

def test_discover_account_returns_first_account() -> None:
    rows = [{"Account": "DU9999"}, {"Account": "DU0001"}]
    sup = _make_supervisor({"accounts_managed": rows})
    result = sup.discover_account()
    assert result == "DU9999"
    assert sup._account == "DU9999"


def test_discover_account_uses_fallback_on_empty_table() -> None:
    sup = _make_supervisor({"accounts_managed": []})
    result = sup.discover_account(fallback="FALLBACK")
    assert result == "FALLBACK"
    assert sup._account == "FALLBACK"


def test_discover_account_returns_empty_string_when_no_fallback() -> None:
    sup = _make_supervisor({"accounts_managed": []})
    result = sup.discover_account()
    assert result == ""


def test_discover_account_surfaces_empty_reason_in_last_error() -> None:
    """A failed discovery records why via last_error, even when a fallback is used."""
    sup = _make_supervisor({"accounts_managed": []})
    sup.discover_account(fallback="FB")
    assert sup._last_error == "accounts_managed is empty"


def test_discover_account_handles_missing_table() -> None:
    # accounts_managed table not in fake tables → snapshot returns []
    sup = _make_supervisor({})
    result = sup.discover_account(fallback="FB")
    assert result == "FB"


# ---------------------------------------------------------------------------
# canonical_account_snapshot() — the account PIVOT
# ---------------------------------------------------------------------------

def _account_kv_rows(account: str = "DU1") -> list[dict[str, Any]]:
    """Build fake accounts_overview key-value rows using real IBKR key names."""
    return [
        {"Account": account, "Key": "NetLiquidation",    "DoubleValue": 1_234_567.89, "Currency": "USD"},
        {"Account": account, "Key": "BuyingPower",       "DoubleValue": 500_000.00,   "Currency": "USD"},
        # IBKR emits "MaintMarginReq", NOT "MaintMargin" — mapped to canonical "MaintMargin"
        {"Account": account, "Key": "MaintMarginReq",    "DoubleValue": 100_000.00,   "Currency": "USD"},
        {"Account": account, "Key": "GrossPositionValue","DoubleValue": 2_000_000.00, "Currency": "USD"},
        # Extra keys that should be ignored
        {"Account": account, "Key": "UnrealizedPnL",     "DoubleValue": 50_000.00,    "Currency": "USD"},
    ]


def test_account_snapshot_produces_one_row() -> None:
    sup = _make_supervisor({"accounts_overview": _account_kv_rows()}, account="DU1")
    rows = sup.canonical_account_snapshot()
    assert len(rows) == 1


def test_account_snapshot_pivot_values() -> None:
    sup = _make_supervisor({"accounts_overview": _account_kv_rows()}, account="DU1")
    row = sup.canonical_account_snapshot()[0]
    assert row["Account"] == "DU1"
    assert row["NetLiquidation"] == pytest.approx(1_234_567.89)
    assert row["BuyingPower"]    == pytest.approx(500_000.00)
    assert row["MaintMargin"]    == pytest.approx(100_000.00)
    assert row["GrossPositionValue"] == pytest.approx(2_000_000.00)
    assert row["Currency"] == "USD"


def test_account_snapshot_timestamp_is_set() -> None:
    sup = _make_supervisor({"accounts_overview": _account_kv_rows()}, account="DU1")
    row = sup.canonical_account_snapshot()[0]
    from datetime import datetime

    ts = row["Timestamp"]
    assert isinstance(ts, datetime)
    assert ts.tzinfo is not None


def test_account_snapshot_filters_by_account() -> None:
    """Rows for a different account must be excluded."""
    rows = _account_kv_rows("DU1") + [
        {"Account": "DU999", "Key": "NetLiquidation", "DoubleValue": 999.0, "Currency": "USD"},
    ]
    sup = _make_supervisor({"accounts_overview": rows}, account="DU1")
    result = sup.canonical_account_snapshot()
    assert len(result) == 1
    assert result[0]["NetLiquidation"] == pytest.approx(1_234_567.89)


def test_account_snapshot_empty_when_no_metric_rows() -> None:
    """If no metric keys exist for the account, return an empty list."""
    rows = [{"Account": "DU1", "Key": "UnrealizedPnL", "DoubleValue": 1.0, "Currency": "USD"}]
    sup = _make_supervisor({"accounts_overview": rows}, account="DU1")
    result = sup.canonical_account_snapshot()
    assert result == []


def test_account_snapshot_falls_back_to_value_string_column() -> None:
    """If DoubleValue is absent, parse the string Value column."""
    rows = [
        {"Account": "DU1", "Key": "NetLiquidation",    "Value": "1234.56", "Currency": "USD"},
        {"Account": "DU1", "Key": "BuyingPower",       "Value": "500.00",  "Currency": "USD"},
        # IBKR emits MaintMarginReq; canonical output key is MaintMargin
        {"Account": "DU1", "Key": "MaintMarginReq",    "Value": "100.00",  "Currency": "USD"},
        {"Account": "DU1", "Key": "GrossPositionValue","Value": "2000.00", "Currency": "USD"},
    ]
    sup = _make_supervisor({"accounts_overview": rows}, account="DU1")
    result = sup.canonical_account_snapshot()
    assert len(result) == 1
    assert result[0]["NetLiquidation"] == pytest.approx(1234.56)


def test_account_snapshot_empty_on_read_error() -> None:
    """snapshot_rows raising must produce an empty list, not an exception."""

    def bad_snapshot(raw_table: Any) -> list[dict[str, Any]]:
        raise RuntimeError("simulated read failure")

    sup = IbkrSupervisor(snapshot_rows_fn=bad_snapshot)
    session = _FakeSession(connected=True, tables={"accounts_overview": object()})
    sup._session = session
    sup._account = "DU1"
    result = sup.canonical_account_snapshot()
    assert result == []


def test_account_snapshot_prefers_usd_over_other_currencies() -> None:
    """With mixed-currency rows where EUR arrives first, USD must win (FIX 5)."""
    rows = [
        # EUR rows arrive FIRST
        {"Account": "DU1", "Key": "NetLiquidation",     "DoubleValue": 1_100_000.0, "Currency": "EUR"},
        {"Account": "DU1", "Key": "BuyingPower",        "DoubleValue": 450_000.0,   "Currency": "EUR"},
        {"Account": "DU1", "Key": "MaintMarginReq",     "DoubleValue": 90_000.0,    "Currency": "EUR"},
        {"Account": "DU1", "Key": "GrossPositionValue", "DoubleValue": 1_800_000.0, "Currency": "EUR"},
        # USD rows arrive AFTER
        {"Account": "DU1", "Key": "NetLiquidation",     "DoubleValue": 1_234_567.89, "Currency": "USD"},
        {"Account": "DU1", "Key": "BuyingPower",        "DoubleValue": 500_000.00,   "Currency": "USD"},
        {"Account": "DU1", "Key": "MaintMarginReq",     "DoubleValue": 100_000.00,   "Currency": "USD"},
        {"Account": "DU1", "Key": "GrossPositionValue", "DoubleValue": 2_000_000.00, "Currency": "USD"},
    ]
    sup = _make_supervisor({"accounts_overview": rows}, account="DU1")
    result = sup.canonical_account_snapshot()
    assert len(result) == 1
    assert result[0]["Currency"] == "USD"
    assert result[0]["NetLiquidation"] == pytest.approx(1_234_567.89)


def test_account_snapshot_falls_back_to_first_currency_when_no_usd() -> None:
    """When no USD rows exist, first currency seen is used (FIX 5)."""
    rows = [
        {"Account": "DU1", "Key": "NetLiquidation",     "DoubleValue": 1_100_000.0, "Currency": "EUR"},
        {"Account": "DU1", "Key": "BuyingPower",        "DoubleValue": 450_000.0,   "Currency": "EUR"},
        {"Account": "DU1", "Key": "MaintMarginReq",     "DoubleValue": 90_000.0,    "Currency": "EUR"},
        {"Account": "DU1", "Key": "GrossPositionValue", "DoubleValue": 1_800_000.0, "Currency": "EUR"},
    ]
    sup = _make_supervisor({"accounts_overview": rows}, account="DU1")
    result = sup.canonical_account_snapshot()
    assert len(result) == 1
    assert result[0]["Currency"] == "EUR"
    assert result[0]["NetLiquidation"] == pytest.approx(1_100_000.0)


def test_account_snapshot_read_error_sets_last_error(caplog: Any) -> None:
    """A read error in canonical_account_snapshot logs at WARNING and sets _last_error (FIX 6)."""
    import logging

    def bad_snapshot(raw_table: Any) -> list[dict[str, Any]]:
        raise RuntimeError("simulated accounts_overview failure")

    sup = IbkrSupervisor(snapshot_rows_fn=bad_snapshot)
    session = _FakeSession(connected=True, tables={"accounts_overview": object()})
    sup._session = session

    with caplog.at_level(logging.WARNING):
        result = sup.canonical_account_snapshot()

    assert result == []
    assert "simulated accounts_overview failure" in sup._last_error
    assert any("accounts_overview" in r.message for r in caplog.records)


def test_wait_for_positions_snapshot_error_continues_loop() -> None:
    """A transient snapshot error in wait_for_positions_settled keeps the loop going (FIX 6)."""
    call_count = 0

    def flaky_snapshot(raw_table: Any) -> list[dict[str, Any]]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("transient error")
        # After the first failure, return stable data so the loop ends.
        return [{"Account": "DU1"}]

    fake_tables: dict[str, object] = {"accounts_positions": object()}
    session = _FakeSession(connected=True, tables=fake_tables)

    sup = IbkrSupervisor(snapshot_rows_fn=flaky_snapshot)
    sup._session = session

    # stable_secs=1 so we exit after 1 consistent read; max_secs=5
    n = sup.wait_for_positions_settled(stable_secs=1, max_secs=5)
    assert n == 1, "should settle to the one stable row after the first error"
    assert call_count >= 2, "must have retried after the transient error"


def test_wait_for_positions_settled_uses_portfolio_fallback(monkeypatch: Any) -> None:
    """When accounts_positions is empty, accounts_portfolio.size is used as the count.

    This tests the 'in' + [] interface fix: the portfolio table is looked up via
    membership test + subscript, not .get().  The _FakeSession.tables dict supports
    both, so this test is style-agnostic; the key assertion is that n=42 is returned.
    """
    import types

    monkeypatch.setattr(
        "ibda.adapters.ibkr.supervisor.time.sleep", lambda _: None
    )

    # accounts_portfolio has a .size attribute mimicking a deephaven Table.
    fake_portfolio = types.SimpleNamespace(size=42)
    fake_positions: object = object()

    # snapshot_rows for accounts_positions always returns [] (empty → triggers fallback)
    def snapshot_rows(raw_table: Any) -> list[dict[str, Any]]:
        if raw_table is fake_positions:
            return []
        return []  # shouldn't be reached for portfolio (accessed via .size, not snapshot_rows)

    session = _FakeSession(
        connected=True,
        tables={"accounts_positions": fake_positions, "accounts_portfolio": fake_portfolio},
    )
    sup = IbkrSupervisor(snapshot_rows_fn=snapshot_rows)
    sup._session = session

    n = sup.wait_for_positions_settled(stable_secs=1, max_secs=10)
    assert n == 42, "portfolio fallback must report accounts_portfolio.size"


def test_wait_for_positions_settled_no_portfolio_table(monkeypatch: Any) -> None:
    """When accounts_portfolio is absent the 'in' check guards the [] access.

    If the 'in' test were omitted and [] used directly, a KeyError would be raised.
    This test confirms the guard works: n stays at 0 (not an exception).
    """
    monkeypatch.setattr(
        "ibda.adapters.ibkr.supervisor.time.sleep", lambda _: None
    )

    fake_positions: object = object()

    def snapshot_rows(raw_table: Any) -> list[dict[str, Any]]:
        return []  # always empty

    # accounts_portfolio deliberately absent from tables
    session = _FakeSession(connected=True, tables={"accounts_positions": fake_positions})
    sup = IbkrSupervisor(snapshot_rows_fn=snapshot_rows)
    sup._session = session

    # stable_secs=2 means we need 2 consecutive stable reads of 0; then max_secs elapses
    n = sup.wait_for_positions_settled(stable_secs=2, max_secs=3)
    assert n == 0, "when portfolio is absent, count stays 0 — no KeyError"


def test_account_snapshot_maintmarginreq_maps_to_maintmargin() -> None:
    """Regression: IBKR emits 'MaintMarginReq'; canonical output key must be 'MaintMargin'.

    A row with the old/wrong key 'MaintMargin' must NOT populate the output,
    while 'MaintMarginReq' MUST populate it under the canonical name 'MaintMargin'.
    """
    # Wrong key (old code assumed this) — must NOT populate MaintMargin output
    rows_wrong_key = [
        {"Account": "DU1", "Key": "MaintMargin",    "DoubleValue": 99999.0, "Currency": "USD"},
        {"Account": "DU1", "Key": "NetLiquidation", "DoubleValue": 1.0,     "Currency": "USD"},
    ]
    sup_wrong = _make_supervisor({"accounts_overview": rows_wrong_key}, account="DU1")
    result_wrong = sup_wrong.canonical_account_snapshot()
    # NetLiquidation still populates (so result is non-empty), but MaintMargin is None
    assert len(result_wrong) == 1
    assert result_wrong[0]["MaintMargin"] is None

    # Correct key — MUST populate MaintMargin under the canonical name
    rows_correct_key = [
        {"Account": "DU1", "Key": "MaintMarginReq", "DoubleValue": 12345.0, "Currency": "USD"},
        {"Account": "DU1", "Key": "NetLiquidation", "DoubleValue": 1.0,     "Currency": "USD"},
    ]
    sup_correct = _make_supervisor({"accounts_overview": rows_correct_key}, account="DU1")
    result_correct = sup_correct.canonical_account_snapshot()
    assert len(result_correct) == 1
    assert result_correct[0]["MaintMargin"] == pytest.approx(12345.0)


# ---------------------------------------------------------------------------
# disconnect() — audit fix E
# ---------------------------------------------------------------------------


def test_disconnect_with_no_session_does_not_raise() -> None:
    """disconnect() is a no-op and does not raise when no session was ever set."""
    sup = IbkrSupervisor()
    assert sup._session is None
    sup.disconnect()  # must not raise


def test_disconnect_calls_session_disconnect_once() -> None:
    """disconnect() delegates to the underlying session's disconnect() exactly once."""
    session = _FakeSession(connected=True)
    sup = IbkrSupervisor()
    sup._session = session

    sup.disconnect()

    assert session.disconnect_calls == 1


def test_disconnect_clears_session_reference() -> None:
    """disconnect() sets _session to None so is_connected() returns False immediately."""
    session = _FakeSession(connected=True)
    sup = IbkrSupervisor()
    sup._session = session
    assert sup.is_connected() is True

    sup.disconnect()

    assert sup._session is None
    assert sup.is_connected() is False


def test_disconnect_is_idempotent_called_twice() -> None:
    """Calling disconnect() twice must not raise.

    First call delegates to the session and clears _session; second call finds
    _session is None and returns immediately — neither call raises.
    """
    session = _FakeSession(connected=True)
    sup = IbkrSupervisor()
    sup._session = session

    sup.disconnect()  # first call — delegates to session
    sup.disconnect()  # second call — _session is already None; must not raise

    assert session.disconnect_calls == 1  # session.disconnect() called only once


def test_disconnect_swallows_session_exception() -> None:
    """If the underlying session.disconnect() raises, disconnect() must not propagate it.

    The guard protects callers that use disconnect() as best-effort teardown
    (e.g. MCP server reconnect) from being broken by a stale/erroring session.
    """

    class _ErrorSession(_FakeSession):
        def disconnect(self) -> None:
            super().disconnect()
            raise RuntimeError("simulated disconnect error")

    session = _ErrorSession(connected=True)
    sup = IbkrSupervisor()
    sup._session = session

    sup.disconnect()  # must not raise despite the session raising

    assert sup._session is None  # session reference still cleared in finally
    assert session.disconnect_calls == 1
