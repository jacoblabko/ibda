"""Tests for ibda.adapters.deephaven.builders — pure (no JVM).

These tests verify the ISO-string formatting used by _to_j_instant without
booting a Deephaven JVM.  The deephaven.time module is patched in sys.modules
so the lazy `from deephaven.time import to_j_instant` inside _to_j_instant uses
a fake that captures its argument.

An instant was previously truncated to whole seconds (format string
"%Y-%m-%dT%H:%M:%S UTC"); the fix added "%f" to preserve microseconds.  These
tests pin that behavior.
"""
from __future__ import annotations

import sys
import types
from datetime import datetime, timezone

import pytest


def _make_fake_dh_time(captured: list[str]) -> types.ModuleType:
    """Return a fake deephaven.time module whose to_j_instant records its argument."""
    fake = types.ModuleType("deephaven.time")

    def _fake_to_j_instant(s: str) -> str:
        captured.append(s)
        return s

    fake.to_j_instant = _fake_to_j_instant  # type: ignore[attr-defined]
    return fake


def test_to_j_instant_preserves_microseconds(monkeypatch: pytest.MonkeyPatch) -> None:
    """_to_j_instant passes a string containing microseconds to deephaven.

    Before the fix the format string was "%Y-%m-%dT%H:%M:%S UTC" (whole-second
    truncation).  After the fix it is "%Y-%m-%dT%H:%M:%S.%f UTC", preserving the
    microsecond component so the value round-trips to TIMESTAMP_NS without loss.
    """
    captured: list[str] = []

    # Inject before the function body runs; the lazy `from deephaven.time import …`
    # inside _to_j_instant will use these entries from sys.modules.
    monkeypatch.setitem(sys.modules, "deephaven", types.ModuleType("deephaven"))
    monkeypatch.setitem(sys.modules, "deephaven.time", _make_fake_dh_time(captured))

    from ibda.adapters.deephaven.builders import _to_j_instant

    dt = datetime(2026, 6, 2, 10, 31, 0, 123456, tzinfo=timezone.utc)  # 123456 µs
    _to_j_instant(dt)

    assert len(captured) == 1, "to_j_instant must be called exactly once"
    iso: str = captured[0]
    assert "123456" in iso, (
        f"microseconds (123456) missing from ISO string passed to deephaven: {iso!r}; "
        "this indicates whole-second truncation is back"
    )
    assert "." in iso, (
        f"no decimal point in ISO string — microseconds not formatted: {iso!r}"
    )


def test_to_j_instant_naive_datetime_treated_as_utc(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tz-naive datetime is treated as UTC (not silently double-offset)."""
    captured: list[str] = []

    monkeypatch.setitem(sys.modules, "deephaven", types.ModuleType("deephaven"))
    monkeypatch.setitem(sys.modules, "deephaven.time", _make_fake_dh_time(captured))

    from ibda.adapters.deephaven.builders import _to_j_instant

    # Naive datetime — builder should attach UTC before formatting.
    dt_naive = datetime(2026, 6, 2, 10, 31, 0, 0)
    _to_j_instant(dt_naive)

    assert len(captured) == 1
    iso = captured[0]
    # UTC suffix must be present so deephaven doesn't misinterpret as local time.
    assert "UTC" in iso, f"UTC suffix missing from ISO string for naive input: {iso!r}"
