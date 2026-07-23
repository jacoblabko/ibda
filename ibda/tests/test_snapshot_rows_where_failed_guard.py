"""Pure tests for the ``snapshot_rows_where`` failed-source guard — no JVM.

Covers the robustness fix for the permanent "failed" engine state a shared
streaming source (notably ``bars_historical``) can enter mid-session (first
observed 2026-07-14, against a live session). Once failed, every
``raw_table.where(predicate)`` call would otherwise raise
``deephaven.dherror.DHError`` forever; ``snapshot_rows_where`` must pre-check
``is_failed`` and return ``[]`` without calling ``.where``.
"""
from __future__ import annotations

import sys
import types
from typing import Any

import pandas as pd
import pytest

from ibda.adapters.deephaven.views import snapshot_rows_where


def _fake_deephaven_pandas(monkeypatch: pytest.MonkeyPatch, df: pd.DataFrame) -> None:
    """Install a fake `deephaven.pandas` module so the lazy, engine-confined
    `from deephaven.pandas import to_pandas` import succeeds without a real
    Deephaven JVM/engine (importing the real `deephaven` package outside a
    running Server raises `RuntimeError`)."""
    fake_pandas_mod = types.ModuleType("deephaven.pandas")
    fake_pandas_mod.to_pandas = lambda table: df  # type: ignore[attr-defined]
    fake_deephaven_pkg = types.ModuleType("deephaven")
    monkeypatch.setitem(sys.modules, "deephaven", fake_deephaven_pkg)
    monkeypatch.setitem(sys.modules, "deephaven.pandas", fake_pandas_mod)


class _FailedTable:
    """Fake table that reports a failed engine state; `.where` must never be called."""

    is_failed = True

    def where(self, predicate: str) -> Any:
        raise AssertionError("where() must not be called when is_failed is True")


class _HealthyTable:
    """Fake table that is not failed; `.where` should be attempted."""

    is_failed = False

    def __init__(self) -> None:
        self.where_calls: list[str] = []

    def where(self, predicate: str) -> "_HealthyTable":
        self.where_calls.append(predicate)
        return self


class _NoIsFailedAttrTable:
    """Fake table with no `is_failed` attribute at all (defensive getattr path)."""

    def where(self, predicate: str) -> "_NoIsFailedAttrTable":
        raise RuntimeError("boom")


def test_snapshot_rows_where_returns_empty_and_skips_where_when_failed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    table = _FailedTable()

    result = snapshot_rows_where(table, "RequestId == 1")

    assert result == []


def test_snapshot_rows_where_logs_warning_when_failed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    table = _FailedTable()

    with caplog.at_level("WARNING", logger="ibda.adapters.deephaven.views"):
        snapshot_rows_where(table, "RequestId == 1")

    assert any(
        "failed state" in record.message for record in caplog.records
    )


def test_snapshot_rows_where_proceeds_to_where_when_not_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _HealthyTable()
    _fake_deephaven_pandas(monkeypatch, pd.DataFrame([{"x": 1}]))

    result = snapshot_rows_where(table, "RequestId == 1")

    assert table.where_calls == ["RequestId == 1"]
    assert result == [{"x": 1}]


def test_snapshot_rows_where_defensive_getattr_when_is_failed_attr_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A table with no `is_failed` attribute must not raise AttributeError from
    the guard itself; it should fall through to attempt `.where` (which we let
    raise its own, unrelated error here -- proving the guard didn't swallow it)."""
    table = _NoIsFailedAttrTable()
    _fake_deephaven_pandas(monkeypatch, pd.DataFrame())

    with pytest.raises(RuntimeError, match="boom"):
        snapshot_rows_where(table, "RequestId == 1")
