"""Position view falls back to accounts_portfolio when accounts_positions is empty
(read-only TWS login withholds reqPositions; updatePortfolio is served)."""

from __future__ import annotations

from typing import Any

from ibda.adapters.ibkr.live import _resolve_position_spec
from ibda.adapters.ibkr.specs import CANONICAL_SPECS, POSITION_PORTFOLIO_SPEC


class _FakeTable:
    def __init__(self, size: int) -> None:
        self.size = size


class _FakeSupervisor:
    def __init__(self, tables: dict[str, Any]) -> None:
        self._tables = tables

    def raw_table(self, name: str) -> Any:
        if name not in self._tables:
            raise KeyError(name)
        return self._tables[name]


def test_uses_default_spec_when_positions_present() -> None:
    sup = _FakeSupervisor({"accounts_positions": _FakeTable(2140), "accounts_portfolio": _FakeTable(2103)})
    spec = _resolve_position_spec(sup, CANONICAL_SPECS["position"])
    assert spec.raw_table == "accounts_positions"


def test_falls_back_to_portfolio_when_positions_empty() -> None:
    sup = _FakeSupervisor({"accounts_positions": _FakeTable(0), "accounts_portfolio": _FakeTable(2103)})
    spec = _resolve_position_spec(sup, CANONICAL_SPECS["position"])
    assert spec.raw_table == "accounts_portfolio"
    assert spec is POSITION_PORTFOLIO_SPEC


def test_keeps_default_when_both_empty() -> None:
    sup = _FakeSupervisor({"accounts_positions": _FakeTable(0), "accounts_portfolio": _FakeTable(0)})
    spec = _resolve_position_spec(sup, CANONICAL_SPECS["position"])
    assert spec.raw_table == "accounts_positions"


def test_keeps_default_when_portfolio_table_absent() -> None:
    sup = _FakeSupervisor({"accounts_positions": _FakeTable(0)})
    spec = _resolve_position_spec(sup, CANONICAL_SPECS["position"])
    assert spec.raw_table == "accounts_positions"


def test_portfolio_spec_maps_rich_columns() -> None:
    r = POSITION_PORTFOLIO_SPEC.renames
    assert r["Position"] == "Qty"
    assert r["ContractId"] == "ConId"
    assert r["Symbol"] == "Sym"
    assert r["MarketValue"] == "MarketValue"
    assert r["UnrealizedPnL"] == "UnrealizedPnl"
    assert POSITION_PORTFOLIO_SPEC.schema_name == "position"
    assert POSITION_PORTFOLIO_SPEC.dedupe_keys == ["Account", "ConId"]
