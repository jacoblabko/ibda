"""Pure tests for ibda.adapters.ibkr.account — no JVM, no TWS.

Tests that snapshot_account_overview logs a WARNING and sets supervisor._last_error
on any read failure (FIX 6), rather than silently returning an empty DataFrame.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ibda.adapters.ibkr.account import snapshot_account_overview
from ibda.adapters.ibkr.supervisor import IbkrSupervisor


class _FakeSession:
    def __init__(self, table_obj: object) -> None:
        self.tables: dict[str, object] = {"accounts_overview": table_obj}

    def is_connected(self) -> bool:
        return True


def _make_sup(
    rows: list[dict[str, Any]],
    raise_on_raw_table: bool = False,
    raise_on_snapshot: bool = False,
) -> IbkrSupervisor:
    """Build a supervisor with injected snapshot + raw_table behaviour."""
    table_obj = object()
    session = _FakeSession(table_obj)

    class _BadRawTable:
        def raw_table(self, name: str) -> Any:
            raise RuntimeError("raw_table error")

    def snapshot_fn(raw: Any) -> list[dict[str, Any]]:
        if raise_on_snapshot:
            raise RuntimeError("snapshot error")
        return rows

    sup = IbkrSupervisor(snapshot_rows_fn=snapshot_fn)
    sup._session = session

    if raise_on_raw_table:
        # Patch raw_table to raise.
        def bad_raw_table(name: str) -> Any:
            raise RuntimeError("raw_table error")
        sup.raw_table = bad_raw_table  # type: ignore[method-assign]

    return sup


def test_snapshot_account_overview_happy_path() -> None:
    rows = [
        {"Account": "DU1", "Key": "NetLiquidation", "DoubleValue": 100.0, "Currency": "USD"},
    ]
    sup = _make_sup(rows)
    df = snapshot_account_overview(sup)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1


def test_snapshot_account_overview_empty_rows_returns_empty_df() -> None:
    sup = _make_sup([])
    df = snapshot_account_overview(sup)
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_snapshot_account_overview_read_error_returns_empty_df(
    caplog: Any,
) -> None:
    """A read failure returns an empty DataFrame — never raises (FIX 6)."""
    sup = _make_sup([], raise_on_snapshot=True)
    with caplog.at_level(logging.WARNING):
        df = snapshot_account_overview(sup)

    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_snapshot_account_overview_read_error_sets_last_error(
    caplog: Any,
) -> None:
    """A read failure sets supervisor._last_error (FIX 6)."""
    sup = _make_sup([], raise_on_snapshot=True)
    with caplog.at_level(logging.WARNING):
        snapshot_account_overview(sup)

    assert "snapshot error" in sup._last_error


def test_snapshot_account_overview_read_error_logs_warning(
    caplog: Any,
) -> None:
    """A read failure logs at WARNING level (FIX 6)."""
    sup = _make_sup([], raise_on_snapshot=True)
    with caplog.at_level(logging.WARNING):
        snapshot_account_overview(sup)

    assert any(
        r.levelno == logging.WARNING and "accounts_overview" in r.message
        for r in caplog.records
    )
