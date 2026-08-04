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


# ---------------------------------------------------------------------------
# Cross-account attribution
#
# Every other test in this file passes account="DU1", so none of them could observe
# the defect: build_account_view reduced each of its five metrics with `last_by([])`
# — no group-by — and joined them on a constant `_k = 1`. With more than one account
# present, each metric independently took whichever account pushed that key last, and
# the result was emitted as ONE row labelled with the NetLiquidation row's Account.
#
# That path is reachable by design, not by accident: supervisor account discovery
# yields "" when accounts_managed is empty or unreadable, live.py passes it through,
# and build_account_view treats "" as falsy and skips the filter deliberately (it
# mirrors the old canonical_account_snapshot degradation). So the unfiltered,
# multi-account case is exactly the degraded-discovery case.
#
# Same defect, same file, as the enrich_position_with_marks fix: `last_by` keyed on
# too few columns, so a value was attributed to the wrong holder.
# ---------------------------------------------------------------------------


def test_two_accounts_do_not_have_their_metrics_mixed_into_one_row() -> None:
    """Unfiltered, each account must get its own row with its own five metrics."""
    from deephaven.time import to_j_instant

    dtw = _make_overview_writer()
    t0 = to_j_instant("2026-07-08T15:00:00 UTC")

    # U_A is written first, U_B second, so U_B is "last" for every key. Under the old
    # constant-key join the single emitted row took U_A's Account label (whatever the
    # NetLiquidation branch carried) beside U_B's BuyingPower and MaintMargin.
    dtw.write_row(1, t0, "U_A", "BASE", "NetLiquidation", "", "", 1_000_000.0)
    dtw.write_row(1, t0, "U_A", "BASE", "BuyingPower", "", "", 500_000.0)
    dtw.write_row(1, t0, "U_A", "BASE", "MaintMarginReq", "", "", 20_000.0)
    dtw.write_row(1, t0, "U_A", "BASE", "GrossPositionValue", "", "", 300_000.0)
    dtw.write_row(1, t0, "U_A", "BASE", "TotalCashValue", "", "", 400_000.0)

    dtw.write_row(1, t0, "U_B", "BASE", "NetLiquidation", "", "", 7.0)
    dtw.write_row(1, t0, "U_B", "BASE", "BuyingPower", "", "", 8.0)
    dtw.write_row(1, t0, "U_B", "BASE", "MaintMarginReq", "", "", 9.0)
    dtw.write_row(1, t0, "U_B", "BASE", "GrossPositionValue", "", "", 10.0)
    dtw.write_row(1, t0, "U_B", "BASE", "TotalCashValue", "", "", 11.0)

    viewed = build_account_view(dtw.table)  # no account filter — the degraded path
    port = ibda.connect({"account": viewed})

    def _has_both() -> bool:
        return bool(port.table("account").snapshot().num_rows == 2)

    assert _poll_until(_has_both), (
        "expected one row per account; got "
        f"{port.table('account').snapshot().num_rows}"
    )

    rows = {r["Account"]: r for r in port.table("account").snapshot().to_pylist()}
    assert set(rows) == {"U_A", "U_B"}

    # The whole point: every metric on a row belongs to that row's account.
    assert rows["U_A"]["NetLiquidation"] == 1_000_000.0
    assert rows["U_A"]["BuyingPower"] == 500_000.0
    assert rows["U_A"]["MaintMargin"] == 20_000.0
    assert rows["U_A"]["GrossPositionValue"] == 300_000.0
    assert rows["U_A"]["TotalCashValue"] == 400_000.0

    assert rows["U_B"]["NetLiquidation"] == 7.0
    assert rows["U_B"]["BuyingPower"] == 8.0
    assert rows["U_B"]["MaintMargin"] == 9.0
    assert rows["U_B"]["GrossPositionValue"] == 10.0
    assert rows["U_B"]["TotalCashValue"] == 11.0
