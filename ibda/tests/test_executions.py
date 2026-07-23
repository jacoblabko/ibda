"""Pure tests for ibda.adapters.ibkr.executions — no JVM, no TWS.

Covers:
- OrderId is present in the EXECUTION schema and its Arrow schema (INT64,
  nullable, positioned right after ConId).
- The ``execution`` spec renames include ``OrderId: OrderId`` (and keep the
  existing renames).
- ``snapshot_executions`` / ``_executions_arrow_from_rows``: the pure
  Arrow-assembly path is unit-tested with canned canonical rows fed through a
  real ``IbkrSupervisor`` via an injected snapshot helper.

JVM-only note: the canonical-view construction (``apply_canonical_view`` over
the live ``orders_exec_details`` table) needs the engine/JVM, so it is covered
by the paper flip-compare + tests_jvm, not here.
"""
from __future__ import annotations

import types
from typing import Any

import pyarrow as pa

from ibda.adapters.ibkr import executions as executions_mod
from ibda.adapters.ibkr.executions import (
    _executions_arrow_from_rows,
    snapshot_executions,
)
from ibda.adapters.ibkr.specs import CANONICAL_SPECS
from ibda.adapters.ibkr.supervisor import IbkrSupervisor
from ibda.schema import EXECUTION


# ---------------------------------------------------------------------------
# 1a — OrderId on the canonical EXECUTION schema
# ---------------------------------------------------------------------------


def test_execution_schema_has_orderid_after_conid() -> None:
    names = EXECUTION.column_names
    assert "OrderId" in names
    assert names.index("OrderId") == names.index("ConId") + 1


def test_execution_orderid_is_int64_nullable() -> None:
    col = next(c for c in EXECUTION.columns if c.name == "OrderId")
    assert col.dtype.name == "INT64"
    assert col.nullable is True


def test_execution_arrow_schema_has_orderid_nullable() -> None:
    arrow = EXECUTION.to_arrow_schema()
    field = arrow.field("OrderId")
    assert field.type == pa.int64()
    assert field.nullable is True


# ---------------------------------------------------------------------------
# 1b — execution spec rename
# ---------------------------------------------------------------------------


def test_execution_spec_renames_orderid() -> None:
    spec = CANONICAL_SPECS["execution"]
    assert spec.renames.get("OrderId") == "OrderId"
    # Existing renames still present.
    assert spec.renames.get("Symbol") == "Sym"
    assert spec.renames.get("Shares") == "Qty"
    assert spec.renames.get("ExecId") == "ExecId"
    assert spec.renames.get("ContractId") == "ConId"


# ---------------------------------------------------------------------------
# 1c — pure Arrow-assembly
# ---------------------------------------------------------------------------


def _canonical_row(
    *,
    exec_id: str,
    sym: str,
    side: str,
    qty: float,
    price: float,
    order_id: int | None,
) -> dict[str, Any]:
    """A canonical EXECUTION snapshot row (as the live view would produce)."""
    return {
        "ExecId": exec_id,
        "Timestamp": None,
        "Account": "DU1",
        "ConId": 12345,
        "OrderId": order_id,
        "Sym": sym,
        "SecType": "STK",
        "Side": side,
        "Qty": qty,
        "Price": price,
        "Multiplier": 1.0,
        "Venue": "NASDAQ",
        "Liquidity": None,
        "Commission": None,
        "RealizedPnl": None,
    }


def test_executions_arrow_conforms_to_schema() -> None:
    rows = [_canonical_row(exec_id="e1", sym="AAPL", side="BUY", qty=100.0, price=155.0, order_id=987)]
    tbl = _executions_arrow_from_rows(rows)
    assert tbl.schema.equals(EXECUTION.to_arrow_schema())
    EXECUTION.validate(tbl)
    assert tbl.num_rows == 1
    row = tbl.to_pylist()[0]
    assert row["ExecId"] == "e1"
    assert row["Sym"] == "AAPL"
    assert row["Qty"] == 100.0
    assert row["OrderId"] == 987
    assert row["Commission"] is None


def test_executions_arrow_empty() -> None:
    tbl = _executions_arrow_from_rows([])
    assert tbl.schema.equals(EXECUTION.to_arrow_schema())
    assert tbl.num_rows == 0


def test_executions_arrow_orderid_nullable_value() -> None:
    rows = [_canonical_row(exec_id="e2", sym="ZZZ", side="SELL", qty=1.0, price=1.0, order_id=None)]
    tbl = _executions_arrow_from_rows(rows)
    assert tbl.to_pylist()[0]["OrderId"] is None


# ---------------------------------------------------------------------------
# snapshot_executions end-to-end via an injected snapshot helper (no JVM)
# ---------------------------------------------------------------------------


def _supervisor_with_executions(rows: list[dict[str, Any]]) -> IbkrSupervisor:
    def snapshot_rows(_view: Any) -> list[dict[str, Any]]:
        return rows

    sup = IbkrSupervisor(snapshot_rows_fn=snapshot_rows)
    sup._session = types.SimpleNamespace(tables={"orders_exec_details": object()})
    return sup


def test_snapshot_executions_builds_arrow(monkeypatch: Any) -> None:
    rows = [
        _canonical_row(exec_id="e1", sym="AAPL", side="BUY", qty=100.0, price=155.0, order_id=987),
        _canonical_row(exec_id="e2", sym="MSFT", side="SELL", qty=50.0, price=300.0, order_id=988),
    ]
    sup = _supervisor_with_executions(rows)
    monkeypatch.setattr(
        executions_mod, "_canonical_execution_view", lambda _sup: object()
    )

    tbl = snapshot_executions(sup)
    assert tbl.schema.equals(EXECUTION.to_arrow_schema())
    assert tbl.num_rows == 2
    syms = [r["Sym"] for r in tbl.to_pylist()]
    assert syms == ["AAPL", "MSFT"]


# ---------------------------------------------------------------------------
# Feature 3a — OrderRef on canonical EXECUTION for strategy attribution
# ---------------------------------------------------------------------------


def test_execution_schema_includes_order_ref() -> None:
    names = [c.name for c in EXECUTION.columns]
    assert "OrderRef" in names


def test_execution_orderref_is_string_nullable() -> None:
    col = next(c for c in EXECUTION.columns if c.name == "OrderRef")
    assert col.dtype.name == "STRING"
    assert col.nullable is True


def test_execution_spec_renames_order_ref() -> None:
    spec = CANONICAL_SPECS["execution"]
    assert spec.renames.get("OrderRef") == "OrderRef"
