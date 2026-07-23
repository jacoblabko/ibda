"""Pure (no-JVM) tests for position-marks enrichment from accounts_portfolio.

Covers the wiring layer in ``live.py``:
- ``_enrich_position_marks`` is called (and updates ``tables["position"]``) when
  accounts_portfolio is non-empty and the position source was accounts_positions.
- ``_enrich_position_marks`` is a no-op when accounts_portfolio is absent or empty.
- No enrichment fires when POSITION_PORTFOLIO_SPEC was the chosen source (marks
  are already present on that path).

The actual Deephaven join inside ``enrich_position_with_marks`` (views.py) requires
the JVM — see ``ibda/tests_jvm/test_position_marks_enrichment.py`` for the
contract test that exercises the real table operations.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

from ibda.adapters.ibkr.live import _enrich_position_marks
from ibda.adapters.ibkr.specs import CANONICAL_SPECS


# ---------------------------------------------------------------------------
# Minimal fake raw-table stub
# ---------------------------------------------------------------------------


class _FakeRawTable:
    """Minimal stand-in for a deephaven raw table (no JVM required)."""

    def __init__(self, size: int) -> None:
        self.size = size


class _FakeSupervisor:
    def __init__(self, tables: dict[str, Any]) -> None:
        self._tables = tables

    def raw_table(self, name: str) -> Any:
        if name not in self._tables:
            raise KeyError(name)
        return self._tables[name]


# ---------------------------------------------------------------------------
# _enrich_position_marks wiring tests
# ---------------------------------------------------------------------------


def test_enrich_calls_enrich_function_and_updates_tables_when_portfolio_non_empty() -> None:
    """_enrich_position_marks must call enrich_position_with_marks and update tables."""
    position_sentinel = object()
    enriched_sentinel = object()

    supervisor = _FakeSupervisor({"accounts_portfolio": _FakeRawTable(2129)})
    tables: dict[str, Any] = {"position": position_sentinel}

    with patch(
        "ibda.adapters.ibkr.live.enrich_position_with_marks",
        return_value=enriched_sentinel,
    ) as mock_enrich:
        _enrich_position_marks(supervisor, tables)

    mock_enrich.assert_called_once()
    args = mock_enrich.call_args
    # First positional arg is the position view, second is the portfolio raw table.
    assert args[0][0] is position_sentinel, "position_view arg must be the original position table"
    assert args[0][1].size == 2129, "portfolio_raw arg must be the accounts_portfolio table"
    assert tables["position"] is enriched_sentinel, "tables['position'] must be updated after enrichment"


def test_enrich_skips_when_portfolio_absent() -> None:
    """_enrich_position_marks must not call enrich when accounts_portfolio raises KeyError."""
    # Supervisor with NO accounts_portfolio at all
    supervisor = _FakeSupervisor({})
    original = object()
    tables: dict[str, Any] = {"position": original}

    with patch(
        "ibda.adapters.ibkr.live.enrich_position_with_marks"
    ) as mock_enrich:
        _enrich_position_marks(supervisor, tables)

    mock_enrich.assert_not_called()
    assert tables["position"] is original, "position table must be unchanged when portfolio absent"


def test_enrich_skips_when_portfolio_empty() -> None:
    """_enrich_position_marks must skip enrichment when accounts_portfolio has 0 rows."""
    supervisor = _FakeSupervisor({"accounts_portfolio": _FakeRawTable(0)})
    original = object()
    tables: dict[str, Any] = {"position": original}

    with patch(
        "ibda.adapters.ibkr.live.enrich_position_with_marks"
    ) as mock_enrich:
        _enrich_position_marks(supervisor, tables)

    mock_enrich.assert_not_called()
    assert tables["position"] is original, "position table must be unchanged when portfolio empty"


def test_enrich_survives_exception_from_enrich_function() -> None:
    """A failure inside enrich_position_with_marks must not propagate — position view unchanged."""
    supervisor = _FakeSupervisor({"accounts_portfolio": _FakeRawTable(100)})
    original = object()
    tables: dict[str, Any] = {"position": original}

    with patch(
        "ibda.adapters.ibkr.live.enrich_position_with_marks",
        side_effect=RuntimeError("deephaven error"),
    ):
        _enrich_position_marks(supervisor, tables)  # must not raise

    assert tables["position"] is original, "position table must be unchanged after failed enrichment"


# ---------------------------------------------------------------------------
# connect_live wiring: enrichment fires only on the accounts_positions path
# ---------------------------------------------------------------------------


def _run_connect_live_with_portfolio(portfolio_size: int) -> dict[str, list[Any]]:
    """Run connect_live (fully stubbed) and return per-call records for _enrich_position_marks."""
    from ibda.adapters.ibkr import live as live_mod

    enrich_calls: list[tuple[Any, dict[str, Any]]] = []

    def _fake_enrich(supervisor: Any, tables: dict[str, Any]) -> None:
        enrich_calls.append((supervisor, dict(tables)))

    # Supervisor that has both accounts_positions (non-empty) and accounts_portfolio.
    # Both non-empty → _resolve_position_spec picks accounts_positions (default spec).
    raw_tables: dict[str, Any] = {
        spec.raw_table: object() for spec in CANONICAL_SPECS.values()
    }
    raw_tables["accounts_positions"] = _FakeRawTable(2127)
    raw_tables["accounts_portfolio"] = _FakeRawTable(portfolio_size)

    class FakeSupervisor:
        _account = "DU_TEST"
        _session: Any = None
        _last_error: str = ""

        def connect(self, **kwargs: Any) -> None:
            pass

        def discover_account(self, fallback: str | None = None) -> str:
            return "DU_TEST"

        def wait_for_positions_settled(self, **kwargs: Any) -> int:
            return 5

        def raw_table(self, name: str) -> Any:
            return raw_tables.get(name, object())

        def snapshot_raw_rows(self, name: str) -> list[dict[str, Any]]:
            return []

        def canonical_account_snapshot(self) -> list[dict[str, Any]]:
            return [{"Account": "DU_TEST"}]

        def is_connected(self) -> bool:
            return True

    with (
        patch.object(live_mod, "start_server", return_value=None),
        patch.object(live_mod, "IbkrSupervisor", return_value=FakeSupervisor()),
        patch.object(live_mod, "apply_canonical_view", side_effect=lambda r, s, sc, join_raw=None: r),
        patch.object(live_mod, "build_account_view", return_value=object()),
        patch.object(live_mod, "build_cash_balance_view", return_value=object()),
        patch.object(live_mod, "_enrich_position_marks", side_effect=_fake_enrich),
    ):
        live_mod.connect_live(client_id=77, read_only=True)

    return {"enrich_calls": enrich_calls}


def test_connect_live_calls_enrichment_when_portfolio_non_empty() -> None:
    """connect_live must call _enrich_position_marks when using accounts_positions source."""
    records = _run_connect_live_with_portfolio(portfolio_size=2129)
    assert len(records["enrich_calls"]) == 1, (
        "Expected _enrich_position_marks to be called exactly once; "
        f"got {len(records['enrich_calls'])} calls"
    )


def test_connect_live_skips_enrichment_on_portfolio_spec_path() -> None:
    """When POSITION_PORTFOLIO_SPEC is used, _enrich_position_marks must NOT be called.

    POSITION_PORTFOLIO_SPEC is chosen when accounts_positions is empty; marks are
    already present from accounts_portfolio so no join is needed.
    """
    from ibda.adapters.ibkr import live as live_mod

    enrich_calls: list[Any] = []

    raw_tables: dict[str, Any] = {
        spec.raw_table: object() for spec in CANONICAL_SPECS.values()
    }
    # Empty accounts_positions → _resolve_position_spec picks POSITION_PORTFOLIO_SPEC.
    raw_tables["accounts_positions"] = _FakeRawTable(0)
    raw_tables["accounts_portfolio"] = _FakeRawTable(2129)

    class FakeSupervisor:
        _account = "DU_TEST"
        _session: Any = None
        _last_error: str = ""

        def connect(self, **kwargs: Any) -> None:
            pass

        def discover_account(self, fallback: str | None = None) -> str:
            return "DU_TEST"

        def wait_for_positions_settled(self, **kwargs: Any) -> int:
            return 5

        def raw_table(self, name: str) -> Any:
            return raw_tables.get(name, object())

        def snapshot_raw_rows(self, name: str) -> list[dict[str, Any]]:
            return []

        def canonical_account_snapshot(self) -> list[dict[str, Any]]:
            return [{"Account": "DU_TEST"}]

        def is_connected(self) -> bool:
            return True

    with (
        patch.object(live_mod, "start_server", return_value=None),
        patch.object(live_mod, "IbkrSupervisor", return_value=FakeSupervisor()),
        patch.object(live_mod, "apply_canonical_view", side_effect=lambda r, s, sc, join_raw=None: r),
        patch.object(live_mod, "build_account_view", return_value=object()),
        patch.object(live_mod, "build_cash_balance_view", return_value=object()),
        patch.object(live_mod, "_enrich_position_marks") as mock_enrich,
    ):
        live_mod.connect_live(client_id=77, read_only=True)

    mock_enrich.assert_not_called()
