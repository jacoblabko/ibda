"""JVM regression test for historical_bars's RequestId scoping.

Proves the exact live defect, observed against a real
Deephaven server: deephaven-ib's ``bars_historical`` table is SHARED across every
``request_bars_historical`` call for the life of a session, so a poll filtered by
``ContractId`` ALONE unions in bars from any other duration/bar_size request
already outstanding on the same contract. This test builds a ``bars_historical``
fixture carrying TWO RequestIds on the SAME ContractId — one simulating an
earlier "1 day" (daily-bar) request already outstanding, one simulating THIS
call's own "5 mins" (intraday) request — and proves ``historical_bars`` returns
ONLY the rows of the request it itself issued, not the union.

``dhib.Duration``/``dhib.BarSize``/``dhib.BarDataType``/``dhib.MarketDataType``
are REAL (imported from the real, JVM-backed ``deephaven_ib`` package — these
are pure value/enum objects and construct fine with no live TWS connection).
Only the TWS session itself (``get_registered_contract`` /
``request_bars_historical``) is faked, mirroring
``ibda/tests_jvm/test_read_intraday_bars_jvm.py``'s pattern.

Run with:
    uv run pytest ibda/tests_jvm/test_historical_bars_request_scoping_jvm.py -q
"""

from __future__ import annotations

from typing import Any

from ibda.adapters.ibkr.supervisor import IbkrSupervisor


# ---------------------------------------------------------------------------
# Table factory
# ---------------------------------------------------------------------------


def _bars_historical_table() -> Any:
    """One ContractId (555), two RequestIds:

    - RequestId=10 (daily-bar request, already outstanding): 3 rows, closes
      100.0/101.0/102.0.
    - RequestId=20 (THIS test's 5-min-bar request): 4 rows, closes
      102.1/102.2/102.3/102.4.

    Timestamps are deliberately interleaved (not grouped by RequestId) so a
    passing test cannot be an artifact of row order.
    """
    from deephaven import new_table
    from deephaven.column import datetime_col, double_col, long_col
    from deephaven.time import to_j_instant

    request_ids = [10, 20, 10, 20, 20, 10, 20]
    contract_ids = [555] * 7
    timestamps = [
        to_j_instant("2026-07-07T20:00:00 UTC"),   # RequestId=10 (daily)
        to_j_instant("2026-07-09T13:30:00 UTC"),   # RequestId=20 (intraday)
        to_j_instant("2026-07-08T20:00:00 UTC"),   # RequestId=10
        to_j_instant("2026-07-09T13:35:00 UTC"),   # RequestId=20
        to_j_instant("2026-07-09T13:40:00 UTC"),   # RequestId=20
        to_j_instant("2026-07-09T20:00:00 UTC"),   # RequestId=10
        to_j_instant("2026-07-09T13:45:00 UTC"),   # RequestId=20
    ]
    closes = [100.0, 102.1, 101.0, 102.2, 102.3, 102.0, 102.4]

    return new_table([
        long_col("RequestId", request_ids),
        long_col("ContractId", contract_ids),
        datetime_col("Timestamp", timestamps),
        double_col("Open", closes),
        double_col("High", closes),
        double_col("Low", closes),
        double_col("Close", closes),
        double_col("Volume", [100.0] * 7),
    ])


class _FakeRequest:
    def __init__(self, request_id: int) -> None:
        self.request_id = request_id


class _FakeSession:
    """Simulates a session with an outstanding daily-bar request (RequestId=10)
    that is now ALSO issuing a NEW 5-min-bar request (RequestId=20) for the same
    contract — exactly the scenario that produced the union-of-bar-sizes defect.
    """

    def __init__(self, dh_table: Any, issued_request_id: int) -> None:
        self.tables = {"bars_historical": dh_table}
        self._issued_request_id = issued_request_id
        self.requested_kwargs: dict[str, Any] = {}

    def get_registered_contract(self, contract: Any) -> object:
        return object()

    def request_bars_historical(self, rc: Any, **kwargs: Any) -> list[_FakeRequest]:
        self.requested_kwargs = kwargs
        return [_FakeRequest(request_id=self._issued_request_id)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_historical_bars_scopes_to_own_request_id_not_union(
    monkeypatch: Any,
) -> None:
    """The RequestId-scoping regression test: two different bar_size requests on
    the SAME ContractId must not union in historical_bars's result — only the
    rows of the request THIS call issued (RequestId=20) come back."""
    from ibda.adapters.ibkr import marketdata

    dh_table = _bars_historical_table()
    session = _FakeSession(dh_table, issued_request_id=20)
    sup = IbkrSupervisor.from_session(session)

    monkeypatch.setattr(marketdata, "_conid_of", lambda rc: 555)
    monkeypatch.setattr(marketdata, "_set_delayed_frozen", lambda sup: None)

    table = marketdata.historical_bars(
        sup, "AAPL", duration="1 D", bar_size="5 mins",
        timeout_s=2.0, poll_interval_s=0.05,
    )

    assert table.num_rows == 4, (
        f"expected 4 rows (RequestId=20 only); got {table.num_rows} — "
        "ContractId-only filtering would union in the 3 daily-bar rows (RequestId=10)"
    )
    # Sorted by arrival order in the fixture (append-only table) -- values, not
    # order, are what this test cares about: exactly the 5-min-request closes.
    assert sorted(table.column("Close").to_pylist()) == [102.1, 102.2, 102.3, 102.4]

    # Real deephaven-ib request_bars_historical objects were passed through
    # (duration/bar_size as .value-bearing wrapper objects, not raw strings).
    assert session.requested_kwargs["duration"].value == "1 D"
    assert session.requested_kwargs["bar_size"].value == "5 mins"
    # historical_bars now takes an explicit keep_up_to_date parameter (default True,
    # forwarded verbatim) -- a later, independently-discovered fix for
    # the same persistent-subscription concern this test's docstring used to note as
    # "out of scope". The default preserves this function's long-standing
    # implicit-True behavior, so daily-bar benchmark fetches still keep ticking
    # unless a caller explicitly passes keep_up_to_date=False.
    assert session.requested_kwargs["keep_up_to_date"] is True


def test_historical_bars_daily_request_unaffected_by_a_later_intraday_request(
    monkeypatch: Any,
) -> None:
    """The inverse direction: a request for the DAILY bars (RequestId=10) must
    return only ITS rows even though a 5-min request (RequestId=20) on the same
    ContractId already sits in the table."""
    from ibda.adapters.ibkr import marketdata

    dh_table = _bars_historical_table()
    session = _FakeSession(dh_table, issued_request_id=10)
    sup = IbkrSupervisor.from_session(session)

    monkeypatch.setattr(marketdata, "_conid_of", lambda rc: 555)
    monkeypatch.setattr(marketdata, "_set_delayed_frozen", lambda sup: None)

    table = marketdata.historical_bars(
        sup, "AAPL", duration="1 Y", bar_size="1 day",
        timeout_s=2.0, poll_interval_s=0.05,
    )

    assert table.num_rows == 3, (
        f"expected 3 rows (RequestId=10 only); got {table.num_rows} — "
        "ContractId-only filtering would union in the 4 intraday-bar rows (RequestId=20)"
    )
    assert sorted(table.column("Close").to_pylist()) == [100.0, 101.0, 102.0]


def test_historical_bars_contract_id_only_would_have_produced_the_union_bug(
    monkeypatch: Any,
) -> None:
    """Sanity check on the fixture itself: proves this table genuinely reproduces
    the pre-fix defect (a ContractId-only filter returns all 7 rows), so the
    passing assertions above are not vacuous."""
    from ibda.adapters.deephaven.views import snapshot_rows_where

    dh_table = _bars_historical_table()
    contract_only = snapshot_rows_where(dh_table, "ContractId == 555")
    assert len(contract_only) == 7, (
        "fixture no longer reproduces the union defect — ContractId-only filtering "
        f"should return all 7 rows (both RequestIds); got {len(contract_only)}"
    )
