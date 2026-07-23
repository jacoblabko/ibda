"""Pure tests for connection_health — no JVM, no TWS (fake supervisor with canned errors rows)."""
from __future__ import annotations

from typing import Any

from ibda.adapters.ibkr.diagnostics import ConnectionHealth, connection_health


class _FakeSupervisor:
    def __init__(self, rows: list[dict[str, Any]], connected: bool = True) -> None:
        self._rows = rows
        self._connected = connected

    def snapshot_raw_rows(self, name: str) -> list[dict[str, Any]]:
        return self._rows if name == "errors" else []

    def is_connected(self) -> bool:
        return self._connected


def _err(code: int, error: str, desc: str = "") -> dict[str, Any]:
    return {"ErrorCode": code, "Error": error, "ErrorDescription": desc or error}


def test_health_all_informational_is_healthy() -> None:
    rows = [
        _err(2104, "Market data farm connection is OK:usfarm"),
        _err(2106, "HMDS data farm connection is OK:ushmds"),
        _err(2158, "Sec-def data farm connection is OK:secdefil"),
    ]
    h = connection_health(_FakeSupervisor(rows))
    assert isinstance(h, ConnectionHealth)
    assert h.connected is True
    assert h.genuine_errors == []
    assert h.connection_lost is False
    assert h.farms.get("usfarm") is True
    assert h.tier_counts.get("INFORMATIONAL", 0) == 3


def test_health_surfaces_genuine_error() -> None:
    rows = [_err(326, "Unable to connect as the client id is already in use")]
    h = connection_health(_FakeSupervisor(rows, connected=False))
    assert h.connected is False
    assert any(g["code"] == 326 for g in h.genuine_errors)
    assert h.tier_counts.get("GENUINE_ERROR", 0) == 1


def test_health_connection_lost_then_restored() -> None:
    # 1100 lost, then 1102 restored → not currently lost.
    rows = [_err(1100, "Connectivity between IB and TWS has been lost."),
            _err(1102, "Connectivity between IB and TWS has been restored.")]
    h = connection_health(_FakeSupervisor(rows))
    assert h.connection_lost is False


def test_health_connection_lost_no_restore() -> None:
    rows = [_err(1100, "Connectivity between IB and TWS has been lost.")]
    h = connection_health(_FakeSupervisor(rows))
    assert h.connection_lost is True


def test_health_farm_broken() -> None:
    rows = [_err(2103, "Market data farm connection is broken:usfarm")]
    h = connection_health(_FakeSupervisor(rows))
    assert h.farms.get("usfarm") is False
    assert h.tier_counts.get("CONNECTIVITY_DEGRADED", 0) == 1


def test_health_handles_missing_errors_table_gracefully() -> None:
    class _Empty:
        def snapshot_raw_rows(self, name: str) -> list[dict[str, Any]]:
            raise KeyError(name)

        def is_connected(self) -> bool:
            return True

    h = connection_health(_Empty())
    assert h.connected is True
    assert h.genuine_errors == []
    assert h.tier_counts == {} or sum(h.tier_counts.values()) == 0
