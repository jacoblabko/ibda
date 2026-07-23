"""JVM test: build_account_view / build_cash_balance_view — LIVE pivots over
accounts_overview (Pass B #1: fixes the static-snapshot-freeze bug).

Uses a DynamicTableWriter-backed accounts_overview-shaped table (same pattern
as test_watch_jvm.py) so rows can be appended AFTER the view is built —
proving the pivot re-ticks on new data rather than freezing at build time
(the behavior the old ``table_from_rows(ACCOUNT, canonical_account_snapshot())``
snapshot had).

Run with:
    uv run pytest ibda/tests_jvm/test_live_account_view.py -q
"""
from __future__ import annotations

import time
from typing import Any

import pytest

import ibda
from ibda.adapters.deephaven.views import build_account_view, build_cash_balance_view
from ibda.schema import ACCOUNT, CASH_BALANCE


def _make_overview_writer() -> Any:
    from deephaven import DynamicTableWriter, dtypes

    return DynamicTableWriter(
        {
            "RequestId": dtypes.long,
            "ReceiveTime": dtypes.Instant,
            "Account": dtypes.string,
            "Currency": dtypes.string,
            "Key": dtypes.string,
            "ModelCode": dtypes.string,
            "Value": dtypes.string,
            "DoubleValue": dtypes.double,
        }
    )


def _write_summary_rows(dtw: Any, account: str, t: Any) -> None:
    """Write the five account-level (unprefixed-key, Currency=BASE) rows."""
    dtw.write_row(1, t, account, "BASE", "NetLiquidation", "", "", 1_000_000.0)
    dtw.write_row(1, t, account, "BASE", "BuyingPower", "", "", 500_000.0)
    dtw.write_row(1, t, account, "BASE", "MaintMarginReq", "", "", 20_000.0)
    dtw.write_row(1, t, account, "BASE", "GrossPositionValue", "", "", 300_000.0)
    dtw.write_row(1, t, account, "BASE", "TotalCashValue", "", "", 400_000.0)


def _write_ledger_rows(dtw: Any, account: str, t: Any) -> None:
    """Write the per-currency ($LEDGER-*) rows for USD and EUR."""
    dtw.write_row(1, t, account, "USD", "$LEDGER-CashBalance", "", "", 100_000.0)
    dtw.write_row(1, t, account, "EUR", "$LEDGER-CashBalance", "", "", 5_000.0)
    dtw.write_row(1, t, account, "USD", "$LEDGER-NetLiquidationByCurrency", "", "", 100_000.0)
    dtw.write_row(1, t, account, "EUR", "$LEDGER-NetLiquidationByCurrency", "", "", 5_500.0)
    dtw.write_row(1, t, account, "USD", "$LEDGER-ExchangeRate", "", "", 1.0)
    dtw.write_row(1, t, account, "EUR", "$LEDGER-ExchangeRate", "", "", 1.08)


def _poll_until(predicate: Any, timeout_s: float = 20.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.2)
    return False


# ---------------------------------------------------------------------------
# build_account_view
# ---------------------------------------------------------------------------


def test_account_view_conforms_to_schema() -> None:
    from deephaven.time import to_j_instant

    dtw = _make_overview_writer()
    t0 = to_j_instant("2026-07-08T15:00:00 UTC")
    _write_summary_rows(dtw, "DU1", t0)

    viewed = build_account_view(dtw.table, account="DU1")
    port = ibda.connect({"account": viewed})

    def _has_row() -> bool:
        return bool(port.table("account").snapshot().num_rows == 1)

    assert _poll_until(_has_row), "account view never produced a row"
    arrow = port.table("account").snapshot()
    ACCOUNT.validate(arrow)


def test_account_view_pivots_metric_values() -> None:
    from deephaven.time import to_j_instant

    dtw = _make_overview_writer()
    t0 = to_j_instant("2026-07-08T15:00:00 UTC")
    _write_summary_rows(dtw, "DU1", t0)

    viewed = build_account_view(dtw.table, account="DU1")
    port = ibda.connect({"account": viewed})

    def _has_row() -> bool:
        return bool(port.table("account").snapshot().num_rows == 1)

    assert _poll_until(_has_row)
    row = port.table("account").snapshot().to_pylist()[0]
    assert row["Account"] == "DU1"
    assert row["NetLiquidation"] == pytest.approx(1_000_000.0)
    assert row["BuyingPower"] == pytest.approx(500_000.0)
    assert row["MaintMargin"] == pytest.approx(20_000.0)
    assert row["GrossPositionValue"] == pytest.approx(300_000.0)
    assert row["Currency"] == "BASE"
    assert row["TotalCashValue"] == pytest.approx(400_000.0)


def test_account_view_is_live_not_frozen() -> None:
    """The KEY regression test for Pass B #1: appending an UPDATED
    NetLiquidation row after the view is built must change what snapshot()
    returns — proving this is a ticking view, not a snapshot frozen at
    connect time (the bug being fixed)."""
    from deephaven.time import to_j_instant

    dtw = _make_overview_writer()
    t0 = to_j_instant("2026-07-08T15:00:00 UTC")
    _write_summary_rows(dtw, "DU1", t0)

    viewed = build_account_view(dtw.table, account="DU1")
    port = ibda.connect({"account": viewed})

    def _has_initial_value() -> bool:
        arrow = port.table("account").snapshot()
        return bool(
            arrow.num_rows == 1
            and arrow.to_pylist()[0]["NetLiquidation"] == pytest.approx(1_000_000.0)
        )

    assert _poll_until(_has_initial_value), "initial NetLiquidation never arrived"

    # Write an updated NetLiquidation for the SAME account — this must
    # supersede the earlier value in the live last_by([]) pivot.
    t1 = to_j_instant("2026-07-08T15:05:00 UTC")
    dtw.write_row(1, t1, "DU1", "BASE", "NetLiquidation", "", "", 1_250_000.0)

    def _has_updated_value() -> bool:
        arrow = port.table("account").snapshot()
        return bool(
            arrow.num_rows == 1
            and arrow.to_pylist()[0]["NetLiquidation"] == pytest.approx(1_250_000.0)
        )

    assert _poll_until(_has_updated_value), (
        "account view did not re-tick on a new NetLiquidation row — "
        "it behaved like a frozen snapshot"
    )


def test_account_view_filters_by_account() -> None:
    """Rows for a different account must not leak into the requested account's view."""
    from deephaven.time import to_j_instant

    dtw = _make_overview_writer()
    t0 = to_j_instant("2026-07-08T15:00:00 UTC")
    _write_summary_rows(dtw, "DU1", t0)
    _write_summary_rows(dtw, "DU2", t0)

    viewed = build_account_view(dtw.table, account="DU1")
    port = ibda.connect({"account": viewed})

    def _has_row() -> bool:
        return bool(port.table("account").snapshot().num_rows == 1)

    assert _poll_until(_has_row)
    row = port.table("account").snapshot().to_pylist()[0]
    assert row["Account"] == "DU1"


# ---------------------------------------------------------------------------
# build_cash_balance_view
# ---------------------------------------------------------------------------


def test_cash_balance_view_conforms_to_schema() -> None:
    from deephaven.time import to_j_instant

    dtw = _make_overview_writer()
    t0 = to_j_instant("2026-07-08T15:00:00 UTC")
    _write_ledger_rows(dtw, "DU1", t0)

    viewed = build_cash_balance_view(dtw.table, account="DU1")
    port = ibda.connect({"cash_balance": viewed})

    def _has_rows() -> bool:
        return bool(port.table("cash_balance").snapshot().num_rows == 2)

    assert _poll_until(_has_rows), "cash_balance view never produced both currency rows"
    arrow = port.table("cash_balance").snapshot()
    CASH_BALANCE.validate(arrow)


def test_cash_balance_view_per_currency_values() -> None:
    from deephaven.time import to_j_instant

    dtw = _make_overview_writer()
    t0 = to_j_instant("2026-07-08T15:00:00 UTC")
    _write_ledger_rows(dtw, "DU1", t0)

    viewed = build_cash_balance_view(dtw.table, account="DU1")
    port = ibda.connect({"cash_balance": viewed})

    def _has_rows() -> bool:
        return bool(port.table("cash_balance").snapshot().num_rows == 2)

    assert _poll_until(_has_rows)
    rows = {r["Currency"]: r for r in port.table("cash_balance").snapshot().to_pylist()}
    assert rows["USD"]["CashBalance"] == pytest.approx(100_000.0)
    assert rows["USD"]["NetLiquidation"] == pytest.approx(100_000.0)
    assert rows["USD"]["ExchangeRate"] == pytest.approx(1.0)
    assert rows["EUR"]["CashBalance"] == pytest.approx(5_000.0)
    assert rows["EUR"]["NetLiquidation"] == pytest.approx(5_500.0)
    assert rows["EUR"]["ExchangeRate"] == pytest.approx(1.08)


def test_cash_balance_view_is_live() -> None:
    """A new currency row appended after the view is built must appear —
    proving the per-currency ledger view ticks too."""
    from deephaven.time import to_j_instant

    dtw = _make_overview_writer()
    t0 = to_j_instant("2026-07-08T15:00:00 UTC")
    _write_ledger_rows(dtw, "DU1", t0)

    viewed = build_cash_balance_view(dtw.table, account="DU1")
    port = ibda.connect({"cash_balance": viewed})

    def _has_two_rows() -> bool:
        return bool(port.table("cash_balance").snapshot().num_rows == 2)

    assert _poll_until(_has_two_rows)

    # Add a third currency (GBP) after the view was built.
    dtw.write_row(1, t0, "DU1", "GBP", "$LEDGER-CashBalance", "", "", 250.0)
    dtw.write_row(1, t0, "DU1", "GBP", "$LEDGER-NetLiquidationByCurrency", "", "", 275.0)

    def _has_three_rows() -> bool:
        return bool(port.table("cash_balance").snapshot().num_rows == 3)

    assert _poll_until(_has_three_rows), "cash_balance view did not re-tick on a new currency row"
