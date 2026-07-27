"""Pure tests for ibda.adapters.ibkr.optionchain — no JVM, no TWS.

A fake session resolves a conId and records the reqSecDefOptParams call; canned
securities_def_option_params rows are fed through a real IbkrSupervisor so the
facade's snapshot_raw_rows poll works end-to-end.
"""
from __future__ import annotations

import datetime as dt
from typing import Any

import re

from ibda.adapters.ibkr.optionchain import (
    OptionChain,
    latest_option_greeks,
    option_chain,
    subscribe_option_greeks,
)
from ibda.adapters.ibkr.supervisor import IbkrSupervisor


def _to_int(value: Any) -> int | None:
    """Test helper: coerce a value to int, returning None on failure."""
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _apply_int_eq_predicate(rows: list[dict[str, Any]], predicate: str) -> list[dict[str, Any]]:
    """Evaluate a simple ``Column == int_value`` predicate for test stubs."""
    m = re.match(r"^(\w+)\s*==\s*(-?\d+)$", predicate.strip())
    if not m:
        return rows
    col, val_str = m.group(1), m.group(2)
    val = int(val_str)
    return [r for r in rows if _to_int(r.get(col)) == val]


class _FakeContract:
    conId = 265598


class _FakeCD:
    contract = _FakeContract()


class _FakeRC:
    contract_details = [_FakeCD()]

    def is_multi(self) -> bool:
        return False


class _FakeMultiRC:
    """A registered contract that resolved to >1 ContractDetails (e.g. SPY's
    SMART option chain resolving to both '2SPY' and 'SPY') -- Fix D's
    ``rc.is_multi()`` guard must skip this rather than let it fan out into
    N ``reqMktData`` calls / N market-data lines."""

    contract_details = [_FakeCD(), _FakeCD()]

    def is_multi(self) -> bool:
        return True


class _FakeClient:
    def __init__(self) -> None:
        self.req_calls: list[dict[str, Any]] = []

    def request_sec_def_opt_params(
        self,
        underlying_symbol: str,
        underlying_con_id: int,
        fut_fop_exchange: str = "",
        underlying_sec_type: str = "STK",
    ) -> int:
        self.req_calls.append({"sym": underlying_symbol, "conid": underlying_con_id})
        return 77


class _FakeSession:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._client = _FakeClient()
        self._rows = rows
        # Stub table objects; the injected snapshot_rows_where_fn ignores the raw
        # table arg, but supervisor.raw_table() is called unconditionally first.
        self.tables: dict[str, Any] = {
            "securities_def_option_params": object(),
        }

    def get_registered_contract(self, contract: object) -> _FakeRC:
        return _FakeRC()


def _supervisor_with_rows(rows: list[dict[str, Any]]) -> IbkrSupervisor:
    session = _FakeSession(rows)

    def snapshot_rows_where_fn(raw_table: Any, predicate: str) -> list[dict[str, Any]]:
        all_rows = rows if True else []  # always the chain rows for this stub
        return _apply_int_eq_predicate(all_rows, predicate)

    sup = IbkrSupervisor.from_session(
        session, snapshot_rows_where_fn=snapshot_rows_where_fn
    )
    # snapshot_raw_rows is still patched for back-compat with any caller that
    # uses it directly (e.g. subscribe_option_greeks path).
    sup.snapshot_raw_rows = lambda name: rows if name == "securities_def_option_params" else []  # type: ignore[method-assign]
    return sup


def test_option_chain_parses_smart_and_by_exchange() -> None:
    rows = [
        {
            "RequestId": 77,
            "Exchange": "SMART",
            "UnderlyingConId": 265598,
            "TradingClass": "AAPL",
            "Multiplier": "100",
            "Expirations": "20260116,20260117",
            "Strikes": "190.0,195.0,200.0",
        },
        {
            "RequestId": 77,
            "Exchange": "CBOE",
            "UnderlyingConId": 265598,
            "TradingClass": "AAPL",
            "Multiplier": "100",
            "Expirations": "20260117",
            "Strikes": "195.0",
        },
        {
            "RequestId": 99,
            "Exchange": "PHLX",
            "UnderlyingConId": 1,
            "TradingClass": "X",
            "Multiplier": "100",
            "Expirations": "20260117",
            "Strikes": "5.0",
        },
    ]
    result = option_chain(_supervisor_with_rows(rows), "AAPL", timeout_s=0.5)

    # On success option_chain returns an OptionChain, not an error dict.
    assert isinstance(result, OptionChain), f"expected OptionChain, got error dict: {result}"
    assert result.symbol == "AAPL"
    assert result.underlying_con_id == 265598
    assert len(result.by_exchange) == 2
    assert result.smart is not None
    assert result.smart.exchange == "SMART"
    assert result.smart.expirations == [dt.date(2026, 1, 16), dt.date(2026, 1, 17)]
    assert result.smart.strikes == [190.0, 195.0, 200.0]


def test_option_chain_timeout_returns_error_dict() -> None:
    """A timeout must return a structured error dict, not an empty OptionChain."""
    result = option_chain(_supervisor_with_rows([]), "AAPL", timeout_s=0.3)
    assert isinstance(result, dict), f"expected error dict on timeout, got {type(result)}"
    assert result["error"] == "chain_timeout"
    assert result["symbol"] == "AAPL"
    assert "underlying_con_id" in result
    assert "message" in result


# ---------------------------------------------------------------------------
# Structured error cases — symbol_not_found and request_failed
# ---------------------------------------------------------------------------


class _FailingResolveSession:
    """Session where get_registered_contract always raises → symbol_not_found."""

    def __init__(self) -> None:
        self._client = _FakeClient()
        self.tables: dict[str, Any] = {"securities_def_option_params": object()}

    def get_registered_contract(self, contract: object) -> _FakeRC:
        raise RuntimeError("No security definition has been found for the request")


class _EmptyContractDetails:
    """Stub RC with empty contract_details so _conid_of returns None."""

    def __init__(self) -> None:
        self.contract_details: list[Any] = []


class _NullConIdSession:
    """Session whose RC has no contract details → _conid_of returns None → symbol_not_found."""

    def __init__(self) -> None:
        self._client = _FakeClient()
        self.tables: dict[str, Any] = {"securities_def_option_params": object()}

    def get_registered_contract(self, contract: object) -> _EmptyContractDetails:
        return _EmptyContractDetails()


class _FailingRequestClient:
    """Client whose request_sec_def_opt_params always raises → request_failed."""

    def request_sec_def_opt_params(
        self,
        underlying_symbol: str,
        underlying_con_id: int,
        fut_fop_exchange: str = "",
        underlying_sec_type: str = "STK",
    ) -> int:
        raise RuntimeError("API connection lost")


class _FailingRequestSession:
    """Session where resolution succeeds but the chain request always raises → request_failed."""

    def __init__(self) -> None:
        self._client: _FailingRequestClient = _FailingRequestClient()
        self.tables: dict[str, Any] = {"securities_def_option_params": object()}

    def get_registered_contract(self, contract: object) -> _FakeRC:
        return _FakeRC()  # resolution succeeds; the request itself will fail


def test_option_chain_symbol_not_found_on_resolution_exception() -> None:
    """get_registered_contract raising must return a symbol_not_found structured error."""
    sup = IbkrSupervisor.from_session(_FailingResolveSession())
    result = option_chain(sup, "ZZZZ")
    assert isinstance(result, dict), f"expected error dict, got {type(result)}"
    assert result["error"] == "symbol_not_found"
    assert result["symbol"] == "ZZZZ"
    assert "message" in result


def test_option_chain_symbol_not_found_on_null_conid() -> None:
    """RC with empty contract_details (conId=None) must return symbol_not_found."""
    sup = IbkrSupervisor.from_session(_NullConIdSession())
    result = option_chain(sup, "ZZZZ")
    assert isinstance(result, dict), f"expected error dict, got {type(result)}"
    assert result["error"] == "symbol_not_found"
    assert result["symbol"] == "ZZZZ"
    assert "message" in result


def test_option_chain_request_failed_on_exception() -> None:
    """request_sec_def_opt_params raising must return a request_failed structured error."""
    sup = IbkrSupervisor.from_session(_FailingRequestSession())
    result = option_chain(sup, "AAPL")
    assert isinstance(result, dict), f"expected error dict, got {type(result)}"
    assert result["error"] == "request_failed"
    assert result["symbol"] == "AAPL"
    assert "underlying_con_id" in result
    assert "message" in result


# ---------------------------------------------------------------------------
# Task 5 — per-contract Greeks helpers
# ---------------------------------------------------------------------------


class _GreeksClient:
    pass


class _GreeksSession:
    def __init__(self, rc: Any = None) -> None:
        self._client = _GreeksClient()
        self.market_data_calls: list[Any] = []
        self.registered_contracts: list[Any] = []
        self._rc = rc if rc is not None else _FakeRC()
        # Stub table objects; the injected snapshot_rows_where_fn ignores the raw
        # table arg, but supervisor.raw_table() is called unconditionally first.
        self.tables: dict[str, Any] = {
            "ticks_option_computation": object(),
        }

    def get_registered_contract(self, contract: object) -> Any:
        self.registered_contracts.append(contract)
        return self._rc

    def request_market_data(self, rc: object, snapshot: bool = False) -> None:
        self.market_data_calls.append((rc, snapshot))


def _supervisor_with_greeks(rows: list[dict[str, Any]]) -> IbkrSupervisor:
    def snapshot_rows_where_fn(raw_table: Any, predicate: str) -> list[dict[str, Any]]:
        return _apply_int_eq_predicate(rows, predicate)

    sup = IbkrSupervisor.from_session(
        _GreeksSession(), snapshot_rows_where_fn=snapshot_rows_where_fn
    )
    sup.snapshot_raw_rows = lambda name: rows if name == "ticks_option_computation" else []  # type: ignore[method-assign]
    return sup


def test_subscribe_option_greeks_resolves_and_requests() -> None:
    session = _GreeksSession()
    sup = IbkrSupervisor.from_session(session)
    conid = subscribe_option_greeks(sup, "AAPL", "20260117", 195.0, "C")
    assert conid == 265598
    assert len(session.market_data_calls) == 1


def test_subscribe_option_greeks_passes_trading_class_to_built_contract() -> None:
    """Fix D: an explicit ``trading_class`` (e.g. an already-parsed
    ``OptionParams.trading_class``) must reach the built ``Contract`` passed to
    ``get_registered_contract`` -- this is what lets IBKR disambiguate a
    multi-trading-class underlying (SPY) down to a single ``ContractDetails``."""
    session = _GreeksSession()
    sup = IbkrSupervisor.from_session(session)
    conid = subscribe_option_greeks(
        sup, "SPY", "20260117", 450.0, "C", trading_class="SPY"
    )
    assert conid == 265598
    assert len(session.registered_contracts) == 1
    assert session.registered_contracts[0].tradingClass == "SPY"


def test_subscribe_option_greeks_skips_ambiguous_multi_contract() -> None:
    """Fix D: when the registered contract resolves to >1 ContractDetails
    (``rc.is_multi() is True`` -- e.g. SPY's SMART chain without a
    ``trading_class``, live-verified 2026-07-16), ``subscribe_option_greeks``
    must return ``None`` and NEVER call ``request_market_data`` -- mirroring
    ``marketdata.py``'s ``request_intraday_bars`` guard -- rather than letting
    the ambiguous contract fan out into N market-data lines for one call."""
    session = _GreeksSession(rc=_FakeMultiRC())
    sup = IbkrSupervisor.from_session(session)
    conid = subscribe_option_greeks(sup, "SPY", "20260117", 450.0, "C")
    assert conid is None
    assert session.market_data_calls == []


def test_latest_option_greeks_picks_matching_conid() -> None:
    rows = [
        {"ContractId": 265598, "Delta": 0.4, "ImpliedVol": 0.31},
        {"ContractId": 265598, "Delta": 0.5, "ImpliedVol": 0.30},
        {"ContractId": 999, "Delta": 0.1, "ImpliedVol": 0.9},
    ]
    sup = _supervisor_with_greeks(rows)
    g = latest_option_greeks(sup, 265598)
    assert g is not None
    assert g["Delta"] == 0.5


# ---------------------------------------------------------------------------
# Structured error modes (symbol_not_found, request_failed)
#
# Before the fix all failure paths (resolve error, no conId, request failure,
# timeout) returned empty / raised silently — callers could not distinguish them.
# After the fix option_chain returns a structured dict {"error": <code>, ...}
# matching the contract that snapshot_quote already uses.
# ---------------------------------------------------------------------------


class _ResolveErrorSession:
    """Session stub where get_registered_contract always raises."""

    def __init__(self) -> None:
        self.tables: dict[str, Any] = {"securities_def_option_params": object()}
        self._client = _FakeClient()

    def get_registered_contract(self, contract: object) -> Any:
        raise RuntimeError("No security definition has been found for the request")


class _NoConIdSession:
    """Session stub where RC resolves but has empty contract_details (→ conId=None)."""

    def __init__(self) -> None:
        self.tables: dict[str, Any] = {"securities_def_option_params": object()}
        self._client = _FakeClient()

    def get_registered_contract(self, contract: object) -> Any:
        class _EmptyRC:
            def __init__(self) -> None:
                self.contract_details: list[Any] = []

        return _EmptyRC()


class _RequestFailedClient:
    """Client stub where request_sec_def_opt_params raises."""

    def request_sec_def_opt_params(
        self,
        underlying_symbol: str = "",
        underlying_con_id: int = 0,
        fut_fop_exchange: str = "",
        underlying_sec_type: str = "STK",
    ) -> int:
        raise RuntimeError("reqSecDefOptParams timed out")


class _RequestFailedSession:
    """Session stub where RC resolves correctly but the chain request raises."""

    def __init__(self) -> None:
        self.tables: dict[str, Any] = {"securities_def_option_params": object()}
        self._client = _RequestFailedClient()

    def get_registered_contract(self, contract: object) -> _FakeRC:
        return _FakeRC()


def _supervisor_from_session(session: Any) -> IbkrSupervisor:
    """Build a minimal IbkrSupervisor from a test session stub."""
    return IbkrSupervisor.from_session(
        session,
        snapshot_rows_fn=lambda _: [],
        snapshot_rows_where_fn=lambda _, __: [],
    )


def test_option_chain_symbol_not_found_on_resolve_error() -> None:
    """get_registered_contract raising → {"error": "symbol_not_found"}.

    Before the fix, the exception was caught but only an empty result was returned
    (no structured error key).  Callers could not tell resolve failure from timeout.
    """
    sup = _supervisor_from_session(_ResolveErrorSession())
    result = option_chain(sup, "ZZZZ", timeout_s=0.1)
    assert isinstance(result, dict), f"expected error dict, got {type(result)}"
    assert result["error"] == "symbol_not_found", (
        f"wrong error code: {result.get('error')!r}"
    )
    assert result["symbol"] == "ZZZZ"
    assert "message" in result


def test_option_chain_symbol_not_found_when_no_conid() -> None:
    """RC with empty contract_details → {"error": "symbol_not_found"}.

    _conid_of returns None when contract_details is empty.  option_chain must
    return a structured error, not empty/None.
    """
    sup = _supervisor_from_session(_NoConIdSession())
    result = option_chain(sup, "AAPL", timeout_s=0.1)
    assert isinstance(result, dict)
    assert result["error"] == "symbol_not_found", (
        f"wrong error code: {result.get('error')!r}"
    )
    assert result["symbol"] == "AAPL"
    assert "message" in result


def test_option_chain_request_failed_on_request_error() -> None:
    """request_sec_def_opt_params raising → {"error": "request_failed"}.

    Before the fix, the exception was swallowed and the poll loop returned empty.
    After the fix, a distinct "request_failed" error dict is returned so callers
    can distinguish a TWS-side request failure from a timeout.
    """
    sup = _supervisor_from_session(_RequestFailedSession())
    result = option_chain(sup, "AAPL", timeout_s=0.1)
    assert isinstance(result, dict)
    assert result["error"] == "request_failed", (
        f"wrong error code: {result.get('error')!r}"
    )
    assert result["symbol"] == "AAPL"
    assert "underlying_con_id" in result   # conId was resolved before the failure
    assert "message" in result
