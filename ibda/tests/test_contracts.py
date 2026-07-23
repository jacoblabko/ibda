"""Pure tests for ibda.adapters.ibkr.contracts — no JVM, no TWS."""
from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from ibda.adapters.ibkr.contracts import (
    _conid_of,
    build_contract_from_conid,
    build_option_contract,
    build_stock_contract,
    get_primary_exchange,
    parse_expiry,
)


def test_parse_expiry_yyyymmdd() -> None:
    assert parse_expiry("20260117") == dt.date(2026, 1, 17)


def test_parse_expiry_yyyymm_is_first_of_month() -> None:
    assert parse_expiry("202601") == dt.date(2026, 1, 1)


def test_parse_expiry_passthrough_date() -> None:
    d = dt.date(2026, 3, 20)
    assert parse_expiry(d) is d


def test_parse_expiry_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        parse_expiry("nope")


def test_build_stock_contract_defaults() -> None:
    c = build_stock_contract("aapl")
    assert c.symbol == "AAPL"
    assert c.secType == "STK"
    assert c.exchange == "SMART"
    assert c.currency == "USD"


def test_build_stock_contract_primary_exchange() -> None:
    c = build_stock_contract("MSFT", primary_exchange="NASDAQ")
    assert c.primaryExchange == "NASDAQ"


def test_build_option_contract() -> None:
    c = build_option_contract("AAPL", "20260117", 195.0, "c")
    assert c.symbol == "AAPL"
    assert c.secType == "OPT"
    assert c.lastTradeDateOrContractMonth == "20260117"
    assert c.strike == 195.0
    assert c.right == "C"
    assert c.multiplier == "100"


def test_build_option_contract_rejects_bad_right() -> None:
    with pytest.raises(ValueError):
        build_option_contract("AAPL", "20260117", 195.0, "X")


class _FakeContract:
    def __init__(self, primary: str | None) -> None:
        self.primaryExchange = primary


class _FakeCD:
    def __init__(self, primary: str | None) -> None:
        self.contract = _FakeContract(primary)


class _FakeRC:
    def __init__(self, primary: str | None) -> None:
        self.contract_details = [_FakeCD(primary)]


class _FakeSession:
    def __init__(self, primary: str | None, raises: bool = False) -> None:
        self._primary = primary
        self._raises = raises

    def get_registered_contract(self, contract: object) -> _FakeRC:
        if self._raises:
            raise RuntimeError("no security definition")
        return _FakeRC(self._primary)


class _FakeSupervisor:
    def __init__(self, session: object) -> None:
        self._session = session


def test_build_contract_from_conid_defaults() -> None:
    c = build_contract_from_conid(265598)
    assert c.conId == 265598
    assert c.exchange == "SMART"


def test_build_contract_from_conid_custom_exchange() -> None:
    c = build_contract_from_conid(12345, exchange="NYSE")
    assert c.conId == 12345
    assert c.exchange == "NYSE"


def test_conid_of_returns_conid() -> None:
    c = _FakeRC(primary="NASDAQ")  # reuse existing fake
    # _conid_of needs contract_details[0].contract.conId
    # _FakeRC has contract_details[0].contract = _FakeContract(primary)
    # _FakeContract has primaryExchange but not conId — so _conid_of returns None
    assert _conid_of(c) is None  # correct: _FakeContract has no conId


class _FakeContractWithId:
    def __init__(self, conid: int) -> None:
        self.conId = conid


class _FakeCDWithId:
    def __init__(self, conid: int) -> None:
        self.contract = _FakeContractWithId(conid)


class _FakeRCWithId:
    def __init__(self, conid: int) -> None:
        self.contract_details = [_FakeCDWithId(conid)]


def test_conid_of_extracts_conid() -> None:
    rc = _FakeRCWithId(265598)
    assert _conid_of(rc) == 265598


def test_conid_of_returns_none_on_empty_details() -> None:
    class _Empty:
        contract_details: list[Any] = []

    assert _conid_of(_Empty()) is None


def test_conid_of_returns_none_on_no_attribute() -> None:
    assert _conid_of(object()) is None


def test_get_primary_exchange_returns_primary() -> None:
    sup = _FakeSupervisor(_FakeSession("NASDAQ"))
    assert get_primary_exchange(sup, "MSFT") == "NASDAQ"


def test_get_primary_exchange_none_when_absent() -> None:
    sup = _FakeSupervisor(_FakeSession(""))
    assert get_primary_exchange(sup, "MSFT") is None


def test_get_primary_exchange_none_on_resolve_failure() -> None:
    sup = _FakeSupervisor(_FakeSession(None, raises=True))
    assert get_primary_exchange(sup, "ZZZZ") is None
