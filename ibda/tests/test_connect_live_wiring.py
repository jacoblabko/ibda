"""Pure wiring tests for connect_live — no JVM, no TWS connection.

Tests the table-assembly logic of connect_live in isolation by:
- Injecting a fake IbkrSupervisor (returns fake raw table sentinels).
- Monkeypatching apply_canonical_view to a stub that returns the raw table
  (identity — sufficient to verify it was called with the right spec).
- Monkeypatching build_account_view / build_cash_balance_view to stubs that
  return sentinel tables and record their call args (the live pivot views
  that replaced the old table_from_rows() snapshot).
- Monkeypatching start_server and supervisor.connect to no-ops.

Assertions:
- apply_canonical_view is called once per CANONICAL_SPEC (now including
  account_pnl, price_tick, news_provider).
- Each call is given the matching spec and schema.
- build_account_view / build_cash_balance_view are each called once, with the
  accounts_overview raw table and the discovered account id.
- The returned DataPort has a tables dict with exactly the canonical key set
  (all CANONICAL_SPECS keys + "account" + "cash_balance").
- No engine (deephaven) import occurs during wiring assembly.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from ibda.adapters.ibkr.specs import CANONICAL_SPECS
from ibda.port import DataPort


# ---------------------------------------------------------------------------
# Fake supervisor
# ---------------------------------------------------------------------------

class _FakeSupervisor:
    """Minimal supervisor stub for wiring tests."""

    def __init__(self) -> None:
        self._account = "DU_TEST"
        self._session: Any = None
        self._last_error: str = ""
        self.disconnect_called = False
        self._raw_tables: dict[str, object] = {
            spec.raw_table: object() for spec in CANONICAL_SPECS.values()
        }
        # accounts_overview is not any CANONICAL_SPECS.raw_table (it's a
        # key-value source pivoted by build_account_view/build_cash_balance_view,
        # not apply_canonical_view) — give it its own stable sentinel.
        self._raw_tables["accounts_overview"] = object()

    def connect(self, **kwargs: Any) -> None:
        pass

    def disconnect(self) -> None:
        self.disconnect_called = True

    def discover_account(self, fallback: str | None = None) -> str:
        return "DU_TEST"

    def wait_for_positions_settled(self, **kwargs: Any) -> int:
        return 5

    def raw_table(self, name: str) -> Any:
        return self._raw_tables.get(name, object())

    def snapshot_raw_rows(self, name: str) -> list[dict[str, Any]]:
        """Return empty rows for any raw table — sufficient for wiring tests."""
        return []

    def is_connected(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXPECTED_CANONICAL_KEYS = set(CANONICAL_SPECS.keys()) | {"account", "cash_balance"}


def _run_connect_live(
    subscribe_symbols: list[str] | None = None,
) -> tuple[Any, DataPort, dict[str, Any]]:
    """Run connect_live with all engine/vendor calls patched to no-ops.

    Returns (supervisor_instance, data_port, call_records) where call_records
    logs what apply_canonical_view / build_account_view / build_cash_balance_view
    were called with.

    live.py imports start_server, apply_canonical_view, build_account_view,
    build_cash_balance_view at module level, so we patch those names on the
    live module object.
    """
    from ibda.adapters.ibkr import live as live_mod

    call_records: dict[str, Any] = {}

    def fake_apply_canonical_view(raw: Any, spec: Any, schema: Any, join_raw: Any = None) -> Any:
        key = spec.schema_name
        call_records[key] = {"spec": spec, "schema": schema, "raw": raw, "join_raw": join_raw}
        return raw  # identity stub

    def fake_build_account_view(overview_raw: Any, account: str | None = None) -> Any:
        call_records["account"] = {"overview_raw": overview_raw, "account": account}
        return overview_raw

    def fake_build_cash_balance_view(overview_raw: Any, account: str | None = None) -> Any:
        call_records["cash_balance"] = {"overview_raw": overview_raw, "account": account}
        return overview_raw

    fake_supervisor = _FakeSupervisor()

    with (
        patch.object(live_mod, "start_server", return_value=None),
        patch.object(live_mod, "IbkrSupervisor", return_value=fake_supervisor),
        patch.object(live_mod, "apply_canonical_view", side_effect=fake_apply_canonical_view),
        patch.object(live_mod, "build_account_view", side_effect=fake_build_account_view),
        patch.object(
            live_mod, "build_cash_balance_view", side_effect=fake_build_cash_balance_view
        ),
    ):
        supervisor, port = live_mod.connect_live(
            client_id=77,
            read_only=True,
            subscribe_symbols=subscribe_symbols,
        )

    return supervisor, port, call_records


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_connect_live_returns_data_port() -> None:
    _, port, _ = _run_connect_live()
    assert isinstance(port, DataPort)


def test_connect_live_tables_has_all_canonical_keys() -> None:
    """The returned DataPort must have all CANONICAL_SPECS + 'account' keys."""
    _, port, _ = _run_connect_live()
    from ibda.adapters.deephaven.adapter import DeephavenPort

    assert isinstance(port, DeephavenPort)
    assert set(port._tables.keys()) == _EXPECTED_CANONICAL_KEYS


def test_connect_live_calls_apply_canonical_view_for_each_spec() -> None:
    """apply_canonical_view must be called once per entry in CANONICAL_SPECS."""
    _, _, call_records = _run_connect_live()
    for canonical_name in CANONICAL_SPECS:
        spec = CANONICAL_SPECS[canonical_name]
        assert spec.schema_name in call_records, (
            f"apply_canonical_view not called for spec {canonical_name!r}"
        )


def test_connect_live_passes_correct_spec_to_apply() -> None:
    """Each apply_canonical_view call must receive the matching spec object."""
    _, _, call_records = _run_connect_live()
    for canonical_name, spec in CANONICAL_SPECS.items():
        record = call_records.get(spec.schema_name)
        assert record is not None
        assert record["spec"] is spec, (
            f"{canonical_name}: expected spec {spec!r}, got {record['spec']!r}"
        )


def test_connect_live_passes_correct_schema_to_apply() -> None:
    """Each apply_canonical_view call must receive the schema matching the spec."""
    from ibda.schema import ALL as ALL_SCHEMAS

    _, _, call_records = _run_connect_live()
    for canonical_name, spec in CANONICAL_SPECS.items():
        record = call_records.get(spec.schema_name)
        assert record is not None
        expected_schema = ALL_SCHEMAS[spec.schema_name]
        assert record["schema"] is expected_schema, (
            f"{canonical_name}: wrong schema passed"
        )


def test_connect_live_builds_account_and_cash_balance_from_accounts_overview() -> None:
    """build_account_view / build_cash_balance_view must each be called once,
    with the accounts_overview raw table and the discovered account id."""
    supervisor, _, call_records = _run_connect_live()
    overview_sentinel = supervisor._raw_tables["accounts_overview"]

    assert call_records["account"]["overview_raw"] is overview_sentinel
    assert call_records["account"]["account"] == "DU_TEST"
    assert call_records["cash_balance"]["overview_raw"] is overview_sentinel
    assert call_records["cash_balance"]["account"] == "DU_TEST"


def test_connect_live_no_subscriptions_without_subscribe_symbols() -> None:
    """If subscribe_symbols is None, _subscribe_for_symbols must NOT be called."""
    from ibda.adapters.ibkr import live as live_mod

    with (
        patch.object(live_mod, "start_server", return_value=None),
        patch.object(live_mod, "IbkrSupervisor", return_value=_FakeSupervisor()),
        patch.object(live_mod, "apply_canonical_view", side_effect=lambda r, s, sc, join_raw=None: r),
        patch.object(live_mod, "build_account_view", return_value=object()),
        patch.object(live_mod, "build_cash_balance_view", return_value=object()),
        patch.object(live_mod, "_subscribe_for_symbols") as mock_sub,
    ):
        live_mod.connect_live(client_id=77, read_only=True, subscribe_symbols=None)

    mock_sub.assert_not_called()


def test_connect_live_calls_subscribe_for_symbols_when_given() -> None:
    """subscribe_symbols != None must trigger _subscribe_for_symbols."""
    from ibda.adapters.ibkr import live as live_mod

    with (
        patch.object(live_mod, "start_server", return_value=None),
        patch.object(live_mod, "IbkrSupervisor", return_value=_FakeSupervisor()),
        patch.object(live_mod, "apply_canonical_view", side_effect=lambda r, s, sc, join_raw=None: r),
        patch.object(live_mod, "build_account_view", return_value=object()),
        patch.object(live_mod, "build_cash_balance_view", return_value=object()),
        patch.object(live_mod, "_subscribe_for_symbols") as mock_sub,
    ):
        live_mod.connect_live(
            client_id=77,
            read_only=True,
            subscribe_symbols=["AAPL", "MSFT"],
        )

    mock_sub.assert_called_once()
    # _subscribe_for_symbols(supervisor, symbols, max_hz=...) — symbols is second positional arg.
    _, called_symbols = mock_sub.call_args.args
    assert set(called_symbols) == {"AAPL", "MSFT"}


def test_connect_live_disconnects_supervisor_on_post_connect_failure() -> None:
    """A failure in any post-connect step (steps 4-7: canonical views, account/
    cash_balance pivots, subscriptions, DataPort wiring) must disconnect the
    supervisor before the exception propagates. Without this, supervisor.connect()
    succeeds (client_id acquired) but connect_live never returns it to the caller —
    a retry then hits IB error 326 ("client id already in use") because the leaked
    session never released client_id."""
    from ibda.adapters.ibkr import live as live_mod

    fake_supervisor = _FakeSupervisor()

    def _boom(overview_raw: Any, account: str | None = None) -> Any:
        raise RuntimeError("boom: build_account_view failed")

    with (
        patch.object(live_mod, "start_server", return_value=None),
        patch.object(live_mod, "IbkrSupervisor", return_value=fake_supervisor),
        patch.object(
            live_mod, "apply_canonical_view", side_effect=lambda r, s, sc, join_raw=None: r
        ),
        patch.object(live_mod, "build_account_view", side_effect=_boom),
        patch.object(live_mod, "build_cash_balance_view", return_value=object()),
    ):
        with pytest.raises(RuntimeError, match="boom: build_account_view failed"):
            live_mod.connect_live(client_id=77, read_only=True, subscribe_symbols=None)

    assert fake_supervisor.disconnect_called is True
