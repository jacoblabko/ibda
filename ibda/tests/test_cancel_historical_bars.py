"""Pure tests for ibda.adapters.ibkr.marketdata.cancel_historical_bars — no JVM, no TWS.

cancel_historical_bars bypasses ``Request.cancel()`` (unusable for the
``keep_up_to_date=True`` line opened by ``request_intraday_bars_ticking``,
since deephaven-ib builds that Request with ``cancel_func=None``) and instead
calls ``cancelHistoricalData`` directly on the raw ibapi client reachable at
``supervisor._session._client``. It must be best-effort: never raise, always
log a WARNING and return ``None`` on any failure, including a missing
session/client.
"""
from __future__ import annotations

import logging
from typing import Any

import pytest

from ibda.adapters.ibkr.marketdata import cancel_historical_bars
from ibda.adapters.ibkr.supervisor import IbkrSupervisor


class _FakeClient:
    def __init__(self, *, raises: bool = False) -> None:
        self._raises = raises
        self.cancel_calls: list[int] = []

    def cancelHistoricalData(self, req_id: int) -> None:
        if self._raises:
            raise RuntimeError("boom: cancelHistoricalData failed")
        self.cancel_calls.append(req_id)


class _FakeSession:
    def __init__(self, client: Any) -> None:
        self._client = client


def _supervisor_with_session(session: Any) -> IbkrSupervisor:
    sup = IbkrSupervisor()
    sup._session = session
    return sup


def test_cancel_historical_bars_calls_cancel_historical_data_once() -> None:
    client = _FakeClient()
    sup = _supervisor_with_session(_FakeSession(client))

    cancel_historical_bars(sup, 4242)

    assert client.cancel_calls == [4242]


def test_cancel_historical_bars_swallows_client_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = _FakeClient(raises=True)
    sup = _supervisor_with_session(_FakeSession(client))

    with caplog.at_level(logging.WARNING, logger="ibda.adapters.ibkr.marketdata"):
        cancel_historical_bars(sup, 4242)

    assert any(
        record.levelno == logging.WARNING and "cancelHistoricalData failed" in record.message
        for record in caplog.records
    )


@pytest.mark.parametrize(
    "session",
    [
        None,
        _FakeSession(None),
    ],
    ids=["session_none", "client_none"],
)
def test_cancel_historical_bars_missing_session_or_client_never_raises(
    session: Any, caplog: pytest.LogCaptureFixture
) -> None:
    sup = _supervisor_with_session(session)

    with caplog.at_level(logging.WARNING, logger="ibda.adapters.ibkr.marketdata"):
        cancel_historical_bars(sup, 4242)

    assert any(record.levelno == logging.WARNING for record in caplog.records)
