"""performance_from_sections reuses the same path as flex_performance.

Both funnel into the single shared core, :func:`ibda.analytics.performance.compute_performance`
— there is no separate re-derivation of Sharpe/Sortino/drawdown for the Flex path.
"""
from __future__ import annotations

from pathlib import Path

from ibda.adapters.ibkr.flex.arrow import flex_performance, performance_from_sections
from ibda.adapters.ibkr.flex.parse import parse_statement_or_raise

_FIXTURE = Path(__file__).parent / "fixtures" / "flex" / "report_full.xml"


def test_performance_from_sections_matches_flex_performance() -> None:
    xml = _FIXTURE.read_text(encoding="utf-8")
    sections = parse_statement_or_raise(xml)
    from_sections = performance_from_sections(sections, risk_free_annual=0.04)
    from_file = flex_performance(str(_FIXTURE), risk_free_annual=0.04)
    assert from_sections.sharpe_ratio == from_file.sharpe_ratio
    assert from_sections.sortino_ratio == from_file.sortino_ratio
    assert from_sections.annualized_return == from_file.annualized_return
    assert from_sections.ending_nav == from_file.ending_nav
    assert from_sections.num_periods == from_file.num_periods


def test_flex_performance_report_arg_is_positional() -> None:
    """flex_performance's first parameter is named ``report`` (not ``source``)."""
    perf = flex_performance(report=str(_FIXTURE), risk_free_annual=0.04)
    assert perf.num_periods > 0


def test_performance_from_sections_accepts_auto_risk_free() -> None:
    from ibda.rates import DEFAULT_RISK_FREE_ANNUAL

    xml = _FIXTURE.read_text(encoding="utf-8")
    sections = parse_statement_or_raise(xml)
    perf = performance_from_sections(sections, risk_free_annual="auto")
    assert perf.risk_free_annual == DEFAULT_RISK_FREE_ANNUAL


def test_an_account_present_only_in_cash_trips_the_multi_account_guard() -> None:
    """The guard must see the section that carries the flows, not just the NAV section.

    `performance_from_sections` derived its account set from NAV rows alone, so a report
    carrying an account in <CashTransactions> that is absent from
    EquitySummaryByReportDateInBase looked single-account: the guard was skipped and the
    second account's deposits were folded into the first's flow series, depressing its
    return with no error and no warning. Deriving the set from NAV *and* cash closes it
    without needing to first establish whether IBKR emits such a report — if it never
    does, the union is identical and this costs nothing.
    """
    import pytest

    xml = _FIXTURE.read_text(encoding="utf-8")
    sections = parse_statement_or_raise(xml)
    stray = dict(sections["cash"][0]) | {"account": "U9999999", "amount": 100_000.0,
                                         "type": "Deposits/Withdrawals"}
    sections = {**sections, "cash": [*sections["cash"], stray]}

    with pytest.raises(ValueError, match="covers multiple accounts"):
        performance_from_sections(sections, risk_free_annual=0.04)


def test_naming_the_account_still_works_when_cash_carries_a_stray() -> None:
    """The guard is a prompt to disambiguate, not a dead end — `account=` still resolves."""
    xml = _FIXTURE.read_text(encoding="utf-8")
    sections = parse_statement_or_raise(xml)
    stray = dict(sections["cash"][0]) | {"account": "U9999999", "amount": 100_000.0,
                                         "type": "Deposits/Withdrawals"}
    spiked = {**sections, "cash": [*sections["cash"], stray]}

    chosen = performance_from_sections(spiked, account="U0000000", risk_free_annual=0.04)
    clean = performance_from_sections(sections, account="U0000000", risk_free_annual=0.04)
    assert chosen.annualized_return == clean.annualized_return, (
        "selecting the account must exclude the stray deposit from its flow series"
    )
