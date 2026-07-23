"""Pure tests for ibda.adapters.ibkr.pnl — no JVM, no TWS.

Tests the row-to-Arrow projection logic for both account_pnl and position_pnl,
including:
- Happy-path column mapping
- ConId String→int cast for position_pnl
- Sym resolution via conid_to_sym
- Double.MAX_VALUE sentinel handling (_f helper)
- None values for missing/bad ConIds
- Empty input → zero-row canonical table with correct schema
- Schema conformance via ACCOUNT_PNL.validate / POSITION_PNL.validate

The subscription entry point (request_single_pnl_for_conids) is not tested
here because it requires a live TWS session; that path is covered by the
live integration test.
"""
from __future__ import annotations

from typing import Any

import pytest

from ibda.adapters.ibkr.pnl import (
    _account_pnl_arrow_from_rows,
    _f,
    _position_pnl_arrow_from_rows,
)
from ibda.schema import ACCOUNT_PNL, POSITION_PNL


# ---------------------------------------------------------------------------
# _f sentinel helper
# ---------------------------------------------------------------------------


def test_f_none_returns_none() -> None:
    assert _f(None) is None


def test_f_float_passthrough() -> None:
    assert _f(3.14) == pytest.approx(3.14)


def test_f_int_cast() -> None:
    assert _f(100) == pytest.approx(100.0)


def test_f_string_int() -> None:
    assert _f("42") == pytest.approx(42.0)


def test_f_double_max_sentinel_returns_none() -> None:
    # IBKR uses Double.MAX_VALUE (~1.7976931e+308) as "not available" sentinel.
    sentinel = 1.7976931348623157e+308
    assert _f(sentinel) is None


def test_f_large_but_not_sentinel() -> None:
    # A very large but non-sentinel float should pass through.
    large = 1.0e+300
    assert _f(large) == pytest.approx(large)


def test_f_bad_string_returns_none() -> None:
    assert _f("not-a-number") is None


# ---------------------------------------------------------------------------
# account_pnl row mapping
# ---------------------------------------------------------------------------

_FAKE_TS: Any = None  # ReceiveTime is opaque in pure tests (no JVM instant)


def _account_pnl_row(
    account: str = "DUQ123",
    daily: float | None = 123.45,
    unrealized: float | None = 200.0,
    realized: float | None = 0.0,
) -> dict[str, Any]:
    return {
        "RequestId": 1,
        "ReceiveTime": _FAKE_TS,
        "Account": account,
        "ModelCode": "",
        "DailyPnl": daily,
        "UnrealizedPnl": unrealized,
        "RealizedPnl": realized,
    }


def test_account_pnl_happy_path_schema_conforms() -> None:
    rows = [_account_pnl_row()]
    tbl = _account_pnl_arrow_from_rows(rows)
    assert tbl.schema.equals(ACCOUNT_PNL.to_arrow_schema())
    ACCOUNT_PNL.validate(tbl)
    assert tbl.num_rows == 1


def test_account_pnl_happy_path_values() -> None:
    rows = [_account_pnl_row(account="DUQ999", daily=500.0, unrealized=1000.0, realized=50.0)]
    tbl = _account_pnl_arrow_from_rows(rows)
    row = tbl.to_pylist()[0]
    assert row["Account"] == "DUQ999"
    assert row["DailyPnl"] == pytest.approx(500.0)
    assert row["UnrealizedPnl"] == pytest.approx(1000.0)
    assert row["RealizedPnl"] == pytest.approx(50.0)


def test_account_pnl_multiple_accounts() -> None:
    rows = [
        _account_pnl_row(account="A1", daily=100.0),
        _account_pnl_row(account="A2", daily=-50.0),
    ]
    tbl = _account_pnl_arrow_from_rows(rows)
    assert tbl.num_rows == 2
    accounts = tbl.column("Account").to_pylist()
    assert "A1" in accounts
    assert "A2" in accounts


def test_account_pnl_empty_input() -> None:
    tbl = _account_pnl_arrow_from_rows([])
    assert tbl.schema.equals(ACCOUNT_PNL.to_arrow_schema())
    assert tbl.num_rows == 0


def test_account_pnl_none_pnl_values() -> None:
    rows = [_account_pnl_row(daily=None, unrealized=None, realized=None)]
    tbl = _account_pnl_arrow_from_rows(rows)
    row = tbl.to_pylist()[0]
    assert row["DailyPnl"] is None
    assert row["UnrealizedPnl"] is None
    assert row["RealizedPnl"] is None


def test_account_pnl_sentinel_becomes_none() -> None:
    rows = [_account_pnl_row(daily=1.7976931348623157e+308)]
    tbl = _account_pnl_arrow_from_rows(rows)
    row = tbl.to_pylist()[0]
    assert row["DailyPnl"] is None


# ---------------------------------------------------------------------------
# position_pnl row mapping
# ---------------------------------------------------------------------------


def _position_pnl_row(
    account: str = "DUQ123",
    conid_str: str = "265598",  # AAPL conId (typical)
    daily: float | None = 75.0,
    unrealized: float | None = 80.0,
    realized: float | None = 0.0,
    value: float | None = 15000.0,
) -> dict[str, Any]:
    return {
        "RequestId": 2,
        "ReceiveTime": _FAKE_TS,
        "Account": account,
        "ModelCode": "",
        "ConId": conid_str,        # arrives as String from deephaven-ib
        "Position": 100.0,
        "DailyPnL": daily,        # capital L in accounts_pnl_single
        "UnrealizedPnL": unrealized,
        "RealizedPnL": realized,
        "Value": value,
    }


_CONID_TO_SYM: dict[int, str] = {265598: "AAPL", 272093: "MSFT"}


def test_position_pnl_happy_path_schema_conforms() -> None:
    rows = [_position_pnl_row()]
    tbl = _position_pnl_arrow_from_rows(rows, _CONID_TO_SYM)
    assert tbl.schema.equals(POSITION_PNL.to_arrow_schema())
    POSITION_PNL.validate(tbl)
    assert tbl.num_rows == 1


def test_position_pnl_conid_cast_from_string() -> None:
    """ConId arrives as String; adapter must cast to int64."""
    rows = [_position_pnl_row(conid_str="265598")]
    tbl = _position_pnl_arrow_from_rows(rows, _CONID_TO_SYM)
    row = tbl.to_pylist()[0]
    assert row["ConId"] == 265598
    assert isinstance(row["ConId"], int)


def test_position_pnl_sym_resolved_from_conid() -> None:
    rows = [_position_pnl_row(conid_str="265598")]
    tbl = _position_pnl_arrow_from_rows(rows, _CONID_TO_SYM)
    assert tbl.to_pylist()[0]["Sym"] == "AAPL"


def test_position_pnl_sym_none_for_unknown_conid() -> None:
    rows = [_position_pnl_row(conid_str="999999")]  # not in conid_to_sym
    tbl = _position_pnl_arrow_from_rows(rows, _CONID_TO_SYM)
    assert tbl.to_pylist()[0]["Sym"] is None


def test_position_pnl_happy_path_values() -> None:
    rows = [_position_pnl_row(daily=75.0, unrealized=80.0, realized=5.0, value=15000.0)]
    tbl = _position_pnl_arrow_from_rows(rows, _CONID_TO_SYM)
    row = tbl.to_pylist()[0]
    assert row["DailyPnl"] == pytest.approx(75.0)
    assert row["UnrealizedPnl"] == pytest.approx(80.0)
    assert row["RealizedPnl"] == pytest.approx(5.0)
    assert row["MarketValue"] == pytest.approx(15000.0)


def test_position_pnl_multiple_positions() -> None:
    rows = [
        _position_pnl_row(conid_str="265598", daily=75.0),
        _position_pnl_row(conid_str="272093", daily=-20.0),
    ]
    tbl = _position_pnl_arrow_from_rows(rows, _CONID_TO_SYM)
    assert tbl.num_rows == 2
    syms = tbl.column("Sym").to_pylist()
    assert "AAPL" in syms
    assert "MSFT" in syms


def test_position_pnl_empty_input() -> None:
    tbl = _position_pnl_arrow_from_rows([], _CONID_TO_SYM)
    assert tbl.schema.equals(POSITION_PNL.to_arrow_schema())
    assert tbl.num_rows == 0


def test_position_pnl_empty_conid_to_sym() -> None:
    """With no sym map, all Sym values are None."""
    rows = [_position_pnl_row(conid_str="265598")]
    tbl = _position_pnl_arrow_from_rows(rows, {})
    assert tbl.to_pylist()[0]["Sym"] is None


def test_position_pnl_none_numeric_values() -> None:
    rows = [_position_pnl_row(daily=None, unrealized=None, realized=None, value=None)]
    tbl = _position_pnl_arrow_from_rows(rows, _CONID_TO_SYM)
    row = tbl.to_pylist()[0]
    assert row["DailyPnl"] is None
    assert row["MarketValue"] is None


def test_position_pnl_sentinel_becomes_none() -> None:
    rows = [_position_pnl_row(daily=1.7976931348623157e+308)]
    tbl = _position_pnl_arrow_from_rows(rows, _CONID_TO_SYM)
    assert tbl.to_pylist()[0]["DailyPnl"] is None


def test_position_pnl_bad_conid_string_skips_row_with_none_conid() -> None:
    """A non-integer ConId string should produce ConId=None (parse fails gracefully)."""
    rows = [_position_pnl_row(conid_str="not-an-int")]
    tbl = _position_pnl_arrow_from_rows(rows, _CONID_TO_SYM)
    assert tbl.num_rows == 1
    row = tbl.to_pylist()[0]
    assert row["ConId"] is None
    assert row["Sym"] is None


# ---------------------------------------------------------------------------
# snapshot_account_pnl / snapshot_position_pnl via supervisor injection
# ---------------------------------------------------------------------------


def _make_supervisor(raw_rows: dict[str, list[dict[str, Any]]]) -> Any:
    """Fake supervisor returning canned raw rows from snapshot_raw_rows."""
    import types

    sup = types.SimpleNamespace(
        snapshot_raw_rows=lambda name: raw_rows.get(name, []),
    )
    return sup


def test_snapshot_account_pnl_via_supervisor() -> None:
    from ibda.adapters.ibkr.pnl import snapshot_account_pnl
    from ibda.adapters.ibkr.supervisor import IbkrSupervisor

    rows = [_account_pnl_row(account="DUQ", daily=200.0)]

    def fake_snapshot_rows(_view: Any) -> list[dict[str, Any]]:
        return rows

    sup = IbkrSupervisor(snapshot_rows_fn=fake_snapshot_rows)
    import types
    sup._session = types.SimpleNamespace(tables={"accounts_pnl": object()})

    tbl = snapshot_account_pnl(sup)
    assert tbl.num_rows == 1
    assert tbl.to_pylist()[0]["Account"] == "DUQ"
    assert tbl.to_pylist()[0]["DailyPnl"] == pytest.approx(200.0)


def test_snapshot_position_pnl_via_supervisor() -> None:
    from ibda.adapters.ibkr.pnl import snapshot_position_pnl
    from ibda.adapters.ibkr.supervisor import IbkrSupervisor

    rows = [_position_pnl_row(conid_str="265598", daily=75.0)]

    def fake_snapshot_rows(_view: Any) -> list[dict[str, Any]]:
        return rows

    sup = IbkrSupervisor(snapshot_rows_fn=fake_snapshot_rows)
    import types
    sup._session = types.SimpleNamespace(tables={"accounts_pnl_single": object()})

    tbl = snapshot_position_pnl(sup, _CONID_TO_SYM)
    assert tbl.num_rows == 1
    row = tbl.to_pylist()[0]
    assert row["ConId"] == 265598
    assert row["Sym"] == "AAPL"
    assert row["DailyPnl"] == pytest.approx(75.0)
