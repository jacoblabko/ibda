"""JVM test: Flex loader -> canonical deephaven Tables -> served via port -> Arrow.

Verifies that flex_canonical_tables produces Tables that:
- conform to the EXECUTION and CASH schemas (via port snapshot + Schema.validate)
- have the expected row counts from the committed fixture

Run with: uv run pytest ibda/tests_jvm/test_flex_loader.py -q
"""
from __future__ import annotations

from pathlib import Path

import ibda
from ibda.adapters.ibkr.flex.loader import flex_canonical_tables
from ibda.adapters.ibkr.flex.parse import parse_statement
from ibda.schema import CASH, EXECUTION, NAV

# ibda/tests/fixtures/flex/report_full.xml, resolved from this test's location
_FIXTURE = (
    Path(__file__).resolve().parent.parent  # -> ibda/
    / "tests"
    / "fixtures"
    / "flex"
    / "report_full.xml"
)


def test_flex_served_through_port_conforms() -> None:
    sections = parse_statement(_FIXTURE.read_text())["sections"]
    tables = flex_canonical_tables(sections)
    port = ibda.connect(tables)

    execs = port.table("execution").snapshot()
    EXECUTION.validate(execs)
    assert execs.num_rows == 2

    cash = port.table("cash").snapshot()
    CASH.validate(cash)
    assert cash.num_rows == 3

    nav = port.table("nav").snapshot()
    NAV.validate(nav)
    assert nav.num_rows == 6


def test_flex_execution_new_columns_populated() -> None:
    """OrderId/Currency/OpenClose (review items #7/#8/#9) survive the full
    parse -> map -> table_from_rows -> port.snapshot() path.

    report_full.xml's SELL trade carries ibOrderID="9988771",
    openCloseIndicator="C", and currency="USD"; the BUY has no ibOrderID/
    openCloseIndicator (OrderId/OpenClose -> None) but does have currency="USD".
    """
    sections = parse_statement(_FIXTURE.read_text())["sections"]
    tables = flex_canonical_tables(sections)
    port = ibda.connect(tables)

    rows = port.table("execution").snapshot().to_pylist()
    sell = next(r for r in rows if r["Side"] == "SELL")
    buy = next(r for r in rows if r["Side"] == "BUY")

    assert sell["OrderId"] == 9988771
    assert buy["OrderId"] is None

    assert sell["OpenClose"] == "C"
    assert buy["OpenClose"] is None

    assert sell["Currency"] == "USD"
    assert buy["Currency"] == "USD"


def test_flex_nav_drives_performance_end_to_end() -> None:
    """Sharpe & friends compute straight off the port's nav table."""
    sections = parse_statement(_FIXTURE.read_text())["sections"]
    port = ibda.connect(flex_canonical_tables(sections))

    perf = ibda.performance_summary(port)
    assert perf.num_periods == 5
    assert perf.starting_nav == 1_000_000.0
    assert perf.ending_nav == 1_004_580.0
    assert perf.sharpe_ratio > 0
