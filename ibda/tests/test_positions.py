"""Pure tests for ibda.adapters.ibkr.positions — no JVM, no TWS.

Covers:
- SecType is present in the POSITION schema and its Arrow schema (nullable,
  positioned right after Sym).
- The ``position`` spec renames include ``SecType: SecType``.
- ``snapshot_positions`` / ``_positions_arrow_from_rows``: the pure
  Arrow-assembly + account-filter path is unit-tested with canned canonical
  rows fed through a real ``IbkrSupervisor`` via an injected snapshot helper.
- (Finding misc-e) ``_VIEW_CACHE`` is a WeakKeyDictionary: entries are
  auto-evicted when the session is garbage-collected, preventing stale-view
  serve and unbounded growth under session churn.

JVM-only note: the canonical-view construction (``apply_canonical_view`` over
the live ``accounts_positions`` table) needs the engine/JVM, so it is covered
by the paper flip-compare, not here.  We exercise ``snapshot_positions``
end-to-end by injecting a fake snapshot helper (returning canned canonical rows)
and a fake session, so the cached-view seam is hit without an engine — the
helper's view object is opaque to the injected snapshot_rows.
"""
from __future__ import annotations

import gc
import types
import weakref
from typing import Any

from ibda.adapters.ibkr import positions as positions_mod
from ibda.adapters.ibkr.positions import (
    _positions_arrow_from_rows,
    snapshot_positions,
)
from ibda.adapters.ibkr.specs import CANONICAL_SPECS
from ibda.adapters.ibkr.supervisor import IbkrSupervisor
from ibda.schema import POSITION


# ---------------------------------------------------------------------------
# 1a — SecType on the canonical POSITION schema
# ---------------------------------------------------------------------------


def test_position_schema_has_sectype_after_sym() -> None:
    names = POSITION.column_names
    assert "SecType" in names
    assert names.index("SecType") == names.index("Sym") + 1


def test_position_sectype_is_string_nullable() -> None:
    col = next(c for c in POSITION.columns if c.name == "SecType")
    assert col.dtype.name == "STRING"
    assert col.nullable is True


def test_position_arrow_schema_has_sectype_nullable() -> None:
    arrow = POSITION.to_arrow_schema()
    field = arrow.field("SecType")
    assert field.type == __import__("pyarrow").string()
    assert field.nullable is True


# ---------------------------------------------------------------------------
# 1b — position spec rename
# ---------------------------------------------------------------------------


def test_position_spec_renames_sectype() -> None:
    spec = CANONICAL_SPECS["position"]
    assert spec.renames.get("SecType") == "SecType"
    # The existing core renames are still present.
    assert spec.renames.get("Symbol") == "Sym"
    assert spec.renames.get("ContractId") == "ConId"
    assert spec.renames.get("Position") == "Qty"
    assert spec.renames.get("AvgCost") == "AvgCost"


# ---------------------------------------------------------------------------
# 1c — pure Arrow-assembly + account filter
# ---------------------------------------------------------------------------


def _canonical_row(
    *,
    account: str,
    sym: str,
    sectype: str | None,
    qty: float,
    avg_cost: float,
) -> dict[str, Any]:
    """A canonical POSITION snapshot row (as the live view would produce)."""
    return {
        "Account": account,
        "ConId": 1000 + len(sym),
        "Sym": sym,
        "SecType": sectype,
        "Qty": qty,
        "AvgCost": avg_cost,
        "MarketPrice": None,
        "MarketValue": None,
        "UnrealizedPnl": None,
    }


def test_positions_arrow_conforms_to_schema() -> None:
    rows = [_canonical_row(account="DU1", sym="AAPL", sectype="STK", qty=100.0, avg_cost=150.0)]
    tbl = _positions_arrow_from_rows(rows)
    assert tbl.schema.equals(POSITION.to_arrow_schema())
    POSITION.validate(tbl)
    assert tbl.num_rows == 1
    row = tbl.to_pylist()[0]
    assert row["Sym"] == "AAPL"
    assert row["SecType"] == "STK"
    assert row["Qty"] == 100.0
    assert row["MarketPrice"] is None


def test_positions_arrow_empty() -> None:
    tbl = _positions_arrow_from_rows([])
    assert tbl.schema.equals(POSITION.to_arrow_schema())
    assert tbl.num_rows == 0


def test_positions_arrow_account_filter() -> None:
    rows = [
        _canonical_row(account="DU1", sym="AAPL", sectype="STK", qty=100.0, avg_cost=150.0),
        _canonical_row(account="DU2", sym="MSFT", sectype="STK", qty=50.0, avg_cost=300.0),
    ]
    tbl = _positions_arrow_from_rows(rows, account="DU1")
    assert tbl.num_rows == 1
    assert tbl.to_pylist()[0]["Sym"] == "AAPL"


def test_positions_arrow_sectype_nullable_value() -> None:
    rows = [_canonical_row(account="DU1", sym="ZZZ", sectype=None, qty=1.0, avg_cost=1.0)]
    tbl = _positions_arrow_from_rows(rows)
    assert tbl.to_pylist()[0]["SecType"] is None


# ---------------------------------------------------------------------------
# snapshot_positions end-to-end via an injected snapshot helper (no JVM)
# ---------------------------------------------------------------------------


def _supervisor_with_positions(rows: list[dict[str, Any]]) -> IbkrSupervisor:
    """A real IbkrSupervisor whose injected snapshot helper returns canned rows.

    The canonical-view object is opaque to the snapshot helper, so this hits the
    snapshot_positions seam without building a real engine view.
    """

    def snapshot_rows(_view: Any) -> list[dict[str, Any]]:
        return rows

    sup = IbkrSupervisor(snapshot_rows_fn=snapshot_rows)
    sup._session = types.SimpleNamespace(tables={"accounts_positions": object()})
    return sup


def test_snapshot_positions_builds_arrow(monkeypatch: Any) -> None:
    rows = [
        _canonical_row(account="DU1", sym="AAPL", sectype="STK", qty=100.0, avg_cost=150.0),
        _canonical_row(account="DU2", sym="MSFT", sectype="STK", qty=50.0, avg_cost=300.0),
    ]
    sup = _supervisor_with_positions(rows)
    # Stub the (engine-requiring) view builder so the cached-view seam stays
    # JVM-free; it just returns an opaque sentinel the snapshot helper ignores.
    monkeypatch.setattr(
        positions_mod, "_canonical_position_view", lambda _sup: object()
    )

    tbl = snapshot_positions(sup)
    assert tbl.schema.equals(POSITION.to_arrow_schema())
    assert tbl.num_rows == 2

    filtered = snapshot_positions(sup, account="DU2")
    assert filtered.num_rows == 1
    assert filtered.to_pylist()[0]["Sym"] == "MSFT"


# ---------------------------------------------------------------------------
# (Finding misc-e) _VIEW_CACHE eviction via WeakKeyDictionary
# ---------------------------------------------------------------------------


def test_view_cache_evicts_when_session_is_garbage_collected() -> None:
    """(Finding misc-e) _VIEW_CACHE auto-evicts when the session object is GC'd.

    The previous dict[int, Any] keyed by id(session) was never evicted: sessions
    that were discarded left stale entries that grew without bound, and a NEW
    session that reused the same memory address would receive the old cached view.

    The WeakKeyDictionary fix auto-evicts on GC — confirmed here by deleting the
    only strong reference to the session and forcing a collection.
    """
    from ibda.adapters.ibkr.positions import _VIEW_CACHE

    class _FakeSession:
        """Opaque stand-in for a deephaven-ib IbSessionTws."""

    session = _FakeSession()
    session_ref: weakref.ref[_FakeSession] = weakref.ref(session)

    # Directly insert a sentinel into the cache (bypasses the JVM-backed view build).
    _VIEW_CACHE[session] = "sentinel_view"
    assert session in _VIEW_CACHE, "entry must be present while session is alive"

    # Release the only strong reference and force a GC cycle.
    del session
    gc.collect()

    # Session must now be dead.
    assert session_ref() is None, "session should have been GC'd after del + gc.collect()"

    # The WeakKeyDictionary must have auto-evicted the entry.
    # We can't reference `session` again, so we check via values() — the sentinel
    # should no longer be reachable from the cache.
    live_values = list(_VIEW_CACHE.values())
    assert "sentinel_view" not in live_values, (
        "_VIEW_CACHE still holds a stale entry after the session was GC'd; "
        "eviction via WeakKeyDictionary is not working"
    )
