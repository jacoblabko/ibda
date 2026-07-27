"""Pure tests for the ATM-option subscribe helper — no JVM, no TWS.

``nearest_expiry`` / ``atm_strike`` are pure functions tested directly.
``subscribe_atm_option`` is tested by monkeypatching the module-level
``option_chain`` / ``subscribe_option_greeks`` collaborators it composes, so no
real supervisor/session is needed.
"""
from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

import ibda.adapters.ibkr.optionchain as oc
from ibda.adapters.ibkr.optionchain import (
    OptionChain,
    OptionParams,
    atm_strike,
    atm_strike_candidates,
    nearest_expiry,
    nearest_monthly_expiry,
    subscribe_atm_option,
)

# ---------------------------------------------------------------------------
# nearest_expiry
# ---------------------------------------------------------------------------


def test_nearest_expiry_picks_earliest_qualifying_not_absolute_earliest() -> None:
    today = dt.date(2026, 1, 1)
    # 10 DTE (too near), 30 DTE, 60 DTE -- min_dte=20 should skip the 10-DTE one.
    expirations = ["20260111", "20260131", "20260302"]
    result = nearest_expiry(expirations, today, min_dte=20)
    assert result == "20260131"


def test_nearest_expiry_none_when_all_too_near() -> None:
    today = dt.date(2026, 1, 1)
    expirations = ["20260105", "20260110"]
    assert nearest_expiry(expirations, today, min_dte=20) is None


def test_nearest_expiry_skips_malformed_token() -> None:
    today = dt.date(2026, 1, 1)
    expirations = ["NOTADATE", "20260201"]
    assert nearest_expiry(expirations, today, min_dte=20) == "20260201"


def test_nearest_expiry_exact_boundary_dte_qualifies() -> None:
    today = dt.date(2026, 1, 1)
    # Exactly 20 days out.
    expirations = ["20260121"]
    assert nearest_expiry(expirations, today, min_dte=20) == "20260121"


def test_nearest_expiry_none_when_no_tokens_parse() -> None:
    today = dt.date(2026, 1, 1)
    assert nearest_expiry(["NOTADATE", "ALSO-BAD"], today, min_dte=20) is None


# ---------------------------------------------------------------------------
# nearest_monthly_expiry
# ---------------------------------------------------------------------------


def test_nearest_monthly_expiry_picks_third_friday_and_skips_weekly() -> None:
    today = dt.date(2026, 1, 1)
    # 2026-01-23 is a Friday but NOT the 3rd Friday of January (that's 2026-01-16) --
    # a "weekly"-style Friday expiry -- and qualifies on DTE alone (22 days). It must
    # be skipped in favor of 2026-02-20, the 3rd Friday of February (a real monthly,
    # 50 days out), even though the weekly is nearer in time.
    expirations = ["20260123", "20260220"]
    assert nearest_monthly_expiry(expirations, today, min_dte=20) == "20260220"


def test_nearest_monthly_expiry_none_when_only_monthly_too_near() -> None:
    today = dt.date(2026, 1, 1)
    # 2026-01-16 IS the 3rd Friday of January, but only 15 DTE -- below min_dte=20.
    assert nearest_monthly_expiry(["20260116"], today, min_dte=20) is None


def test_nearest_monthly_expiry_none_when_no_monthly_present() -> None:
    today = dt.date(2026, 1, 1)
    # Neither token is a 3rd Friday.
    assert nearest_monthly_expiry(["20260109", "20260123"], today, min_dte=20) is None


def test_nearest_monthly_expiry_skips_malformed_token() -> None:
    today = dt.date(2026, 1, 1)
    assert nearest_monthly_expiry(["NOTADATE", "20260220"], today, min_dte=20) == "20260220"


# ---------------------------------------------------------------------------
# nearest_expiry — monthly preference over the plain "earliest qualifying" fallback
# ---------------------------------------------------------------------------


def test_nearest_expiry_prefers_monthly_over_nearer_non_monthly() -> None:
    today = dt.date(2026, 1, 1)
    # 20260123 (a non-monthly Friday, 22 DTE) is nearer than 20260220 (the monthly,
    # 50 DTE) -- the plain "earliest qualifying" rule would pick 20260123, but
    # nearest_expiry must prefer the monthly since one qualifies.
    expirations = ["20260123", "20260220"]
    assert nearest_expiry(expirations, today, min_dte=20) == "20260220"


def test_nearest_expiry_falls_back_to_any_expiry_when_no_monthly_qualifies() -> None:
    today = dt.date(2026, 1, 1)
    # No monthly at all in this list -- falls back to the original "earliest
    # qualifying" behavior (still exercised directly by the tests above this one).
    expirations = ["20260109", "20260123"]
    assert nearest_expiry(expirations, today, min_dte=20) == "20260123"


# ---------------------------------------------------------------------------
# atm_strike
# ---------------------------------------------------------------------------


def test_atm_strike_picks_closest() -> None:
    assert atm_strike([90.0, 100.0, 110.0], 101.0) == 100.0


def test_atm_strike_tie_resolves_to_lower() -> None:
    # 95 and 105 are equidistant from 100 -- lower strike wins.
    assert atm_strike([95.0, 105.0], 100.0) == 95.0


def test_atm_strike_empty_strikes_is_none() -> None:
    assert atm_strike([], 100.0) is None


def test_atm_strike_nonpositive_spot_is_none() -> None:
    assert atm_strike([90.0, 100.0], 0.0) is None
    assert atm_strike([90.0, 100.0], -5.0) is None


# ---------------------------------------------------------------------------
# atm_strike_candidates
# ---------------------------------------------------------------------------


def test_atm_strike_candidates_orders_nearest_first_with_lower_tiebreak() -> None:
    # distances from spot=100: 90->10, 95->5, 100->0, 105->5, 110->10, 120->20.
    # Ties (95/105 at 5; 90/110 at 10) resolve to the lower strike first.
    strikes = [90.0, 95.0, 100.0, 105.0, 110.0, 120.0]
    result = atm_strike_candidates(strikes, 100.0, k=10)
    assert result == [100.0, 95.0, 105.0, 90.0, 110.0, 120.0]


def test_atm_strike_candidates_truncates_to_k() -> None:
    strikes = [90.0, 95.0, 100.0, 105.0, 110.0, 120.0]
    result = atm_strike_candidates(strikes, 100.0, k=3)
    assert result == [100.0, 95.0, 105.0]


def test_atm_strike_candidates_empty_strikes_is_empty_list() -> None:
    assert atm_strike_candidates([], 100.0) == []


def test_atm_strike_candidates_nonpositive_spot_is_empty_list() -> None:
    assert atm_strike_candidates([90.0, 100.0], 0.0) == []
    assert atm_strike_candidates([90.0, 100.0], -5.0) == []


def test_atm_strike_candidates_default_k_is_five() -> None:
    strikes = [float(s) for s in range(50, 151, 5)]  # 21 strikes, 50..150 step 5
    result = atm_strike_candidates(strikes, 100.0)
    assert len(result) == 5


# ---------------------------------------------------------------------------
# subscribe_atm_option -- composition, with fakes/monkeypatches
# ---------------------------------------------------------------------------


def _chain(expirations: list[dt.date], strikes: list[float]) -> OptionChain:
    return OptionChain(
        symbol="AAPL",
        underlying_con_id=1,
        smart=OptionParams(
            exchange="SMART",
            trading_class="AAPL",
            multiplier="100",
            expirations=expirations,
            strikes=strikes,
        ),
        by_exchange=[],
    )


def test_subscribe_atm_option_none_on_chain_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        oc,
        "option_chain",
        lambda supervisor, symbol, **kw: {
            "error": "symbol_not_found", "symbol": symbol, "message": "x",
        },
    )
    result = subscribe_atm_option(object(), "AAPL", 100.0)
    assert result is None


def test_subscribe_atm_option_none_when_no_qualifying_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    today = dt.date(2026, 1, 1)
    chain = _chain(expirations=[dt.date(2026, 1, 5)], strikes=[100.0])
    monkeypatch.setattr(oc, "option_chain", lambda supervisor, symbol, **kw: chain)
    result = subscribe_atm_option(object(), "AAPL", 100.0, min_dte=20, today=today)
    assert result is None


def test_subscribe_atm_option_happy_path_returns_conid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    today = dt.date(2026, 1, 1)
    chain = _chain(expirations=[dt.date(2026, 2, 1)], strikes=[95.0, 100.0, 105.0])
    captured: dict[str, Any] = {}

    def _fake_subscribe_option_greeks(
        supervisor: Any, symbol: str, expiry: Any, strike: float, right: str,
        exchange: str = "SMART", trading_class: str | None = None,
    ) -> int:
        captured.update(
            symbol=symbol, expiry=expiry, strike=strike, right=right,
            exchange=exchange, trading_class=trading_class,
        )
        return 42

    monkeypatch.setattr(oc, "option_chain", lambda supervisor, symbol, **kw: chain)
    monkeypatch.setattr(oc, "subscribe_option_greeks", _fake_subscribe_option_greeks)

    result = subscribe_atm_option(object(), "AAPL", 101.0, min_dte=20, today=today)

    assert result == 42
    assert captured["symbol"] == "AAPL"
    assert captured["expiry"] == "20260201"
    assert captured["strike"] == 100.0
    assert captured["right"] == "C"
    assert captured["exchange"] == "SMART"
    # Fix D: the already-parsed OptionParams.trading_class ("AAPL", from _chain())
    # must reach subscribe_option_greeks so it can disambiguate a multi-trading-class
    # underlying instead of fanning out into >1 market-data line.
    assert captured["trading_class"] == "AAPL"


def test_subscribe_atm_option_never_raises_on_collaborator_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(supervisor: Any, symbol: str, **kw: Any) -> OptionChain:
        raise RuntimeError("boom")

    monkeypatch.setattr(oc, "option_chain", _raise)
    result = subscribe_atm_option(object(), "AAPL", 100.0)
    assert result is None


# ---------------------------------------------------------------------------
# subscribe_atm_option -- multi-candidate strike walk (the NVDA-2026-07-09 fix)
# ---------------------------------------------------------------------------


def test_subscribe_atm_option_walks_to_second_candidate_when_first_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The nearest strike (100.0, distance 1.0 from spot=101.0) fails to resolve
    (the union-list expiry/strike pairing does not guarantee it exists) -- the next
    nearest candidate (105.0, distance 4.0) must be tried next and succeed."""
    today = dt.date(2026, 1, 1)
    chain = _chain(expirations=[dt.date(2026, 2, 20)], strikes=[95.0, 100.0, 105.0])
    attempts: list[float] = []

    def _fake_subscribe_option_greeks(
        supervisor: Any, symbol: str, expiry: Any, strike: float, right: str,
        exchange: str = "SMART", trading_class: str | None = None,
    ) -> int | None:
        attempts.append(strike)
        return None if strike == 100.0 else 77

    monkeypatch.setattr(oc, "option_chain", lambda supervisor, symbol, **kw: chain)
    monkeypatch.setattr(oc, "subscribe_option_greeks", _fake_subscribe_option_greeks)

    result = subscribe_atm_option(object(), "AAPL", 101.0, min_dte=20, today=today)

    assert result == 77
    assert attempts == [100.0, 105.0]


def test_subscribe_atm_option_none_when_every_candidate_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    today = dt.date(2026, 1, 1)
    chain = _chain(expirations=[dt.date(2026, 2, 20)], strikes=[95.0, 100.0, 105.0])
    attempts: list[float] = []

    def _always_fail(
        supervisor: Any, symbol: str, expiry: Any, strike: float, right: str,
        exchange: str = "SMART", trading_class: str | None = None,
    ) -> int | None:
        attempts.append(strike)
        return None

    monkeypatch.setattr(oc, "option_chain", lambda supervisor, symbol, **kw: chain)
    monkeypatch.setattr(oc, "subscribe_option_greeks", _always_fail)

    result = subscribe_atm_option(object(), "AAPL", 101.0, min_dte=20, today=today)

    assert result is None
    # Nearest-first order at spot=101.0: 100.0 (d=1), 105.0 (d=4), 95.0 (d=6).
    assert attempts == [100.0, 105.0, 95.0]


def test_subscribe_atm_option_respects_max_strike_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """max_strike_candidates=1 must not try a second strike even if the first fails."""
    today = dt.date(2026, 1, 1)
    chain = _chain(expirations=[dt.date(2026, 2, 20)], strikes=[95.0, 100.0, 105.0])
    attempts: list[float] = []

    def _always_fail(
        supervisor: Any, symbol: str, expiry: Any, strike: float, right: str,
        exchange: str = "SMART", trading_class: str | None = None,
    ) -> int | None:
        attempts.append(strike)
        return None

    monkeypatch.setattr(oc, "option_chain", lambda supervisor, symbol, **kw: chain)
    monkeypatch.setattr(oc, "subscribe_option_greeks", _always_fail)

    result = subscribe_atm_option(
        object(), "AAPL", 101.0, min_dte=20, today=today, max_strike_candidates=1,
    )

    assert result is None
    assert attempts == [100.0]
