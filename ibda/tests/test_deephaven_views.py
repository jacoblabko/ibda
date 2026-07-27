"""Pure tests for ``ibda.adapters.deephaven.views.snapshot_rows_where``'s
failed-source-table degradation -- no JVM, no TWS.

The deephaven-ib ``bars_historical`` streaming table can enter a permanent Deephaven engine
"failed" state mid-session. Once failed, every ``.where()`` call against it
raises ``io.deephaven.engine.exceptions.TableAlreadyFailedException``,
surfaced to Python as ``deephaven.dherror.DHError: table where operation
failed.``. ``snapshot_rows_where`` now checks ``Table.is_failed`` BEFORE
calling ``where()`` and degrades to an empty result instead of raising --
this is the low, shared layer under both the intraday-P&L recompute path and
(indirectly, via ``historical_bars``'s poll loop) the symbol-detail daily
price-history fetch.

A plain fake double (``_FakeTable``, exposing only ``is_failed`` and a
``where()`` that raises if ever called) is enough to prove the PRE-CHECK
branch: importing ``ibda.adapters.deephaven.views`` requires no JVM (its
``deephaven`` imports are all function-local/lazy), and the pre-check
short-circuits before any of those lazy imports run, so this exercises the
exact "table is already failed" scenario confirmed live -- no engine needed.

**There is deliberately no race-window test, because there is no race-window
branch.** If the table fails in the gap between the ``is_failed`` check and the
``where()`` call, ``snapshot_rows_where`` lets the ``DHError`` propagate, and
its docstring says so explicitly ("Any OTHER exception ... is NOT swallowed and
propagates as before"). That is a deliberate scoping decision, not an omission:
the guard exists to stop a *permanently* failed table from raising on every
~10 s recompute cycle, and a table that fails mid-call raises exactly once
before the next cycle's pre-check catches it. An earlier draft of this fix also
caught ``DHError`` and re-checked ``is_failed`` afterwards; that draft was not
taken, and these tests describe the shipped behaviour. Do not port race-window
assertions here without first porting the branch they test.
"""
from __future__ import annotations

from typing import Any

import pytest

from ibda.adapters.deephaven.views import snapshot_rows_where


class _FakeTable:
    """Minimal double exposing only what ``snapshot_rows_where`` touches."""

    def __init__(self, is_failed: bool) -> None:
        self.is_failed = is_failed

    def where(self, predicate: str) -> Any:
        raise AssertionError(
            "where() must not be called once is_failed is True -- "
            "snapshot_rows_where's pre-check must short-circuit first"
        )


def test_snapshot_rows_where_returns_empty_for_already_failed_table() -> None:
    """The exact live-confirmed scenario: a permanently-failed source table
    degrades to 0 rows instead of raising ``DHError`` -- the fix that keeps
    ``historical_bars``'s poll loop (and every other ``snapshot_rows_where``
    caller) from propagating a ``TableAlreadyFailedException`` into the UI.
    """
    rows = snapshot_rows_where(_FakeTable(is_failed=True), "ContractId == 12345")
    assert rows == []


def test_snapshot_rows_where_never_calls_where_on_a_failed_table() -> None:
    """Belt-and-suspenders on the same behavior: ``_FakeTable.where`` raises
    ``AssertionError`` if invoked, so a passing test proves the pre-check
    short-circuits before the (engine-requiring, exception-prone) ``where()``
    call is ever attempted.
    """
    table = _FakeTable(is_failed=True)
    rows = snapshot_rows_where(table, "RequestId == 1")
    assert rows == []


@pytest.mark.parametrize("predicate", ["ContractId == 1", "RequestId == 2 && ContractId == 3"])
def test_snapshot_rows_where_degrades_regardless_of_predicate(predicate: str) -> None:
    """The degrade path does not depend on the predicate's shape -- any
    filter expression against an already-failed table degrades the same way.
    """
    assert snapshot_rows_where(_FakeTable(is_failed=True), predicate) == []
