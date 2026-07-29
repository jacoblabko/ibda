from __future__ import annotations
import datetime as dt

import pytest

from ibda.analytics.roundtrips import Fill, aggregate_round_trips, reconstruct_round_trips

UTC = dt.timezone.utc
def _t(mins: int) -> dt.datetime: return dt.datetime(2026, 6, 30, 14, mins, 0, tzinfo=UTC)
def _f(
    side: str,
    qty: float,
    price: float,
    ts: dt.datetime,
    *,
    rp: float = 0.0,
    comm: float = -1.0,
    sym: str = "AAPL",
    strat: str = "s1",
    mult: float = 1.0,
    seq: int = 0,
) -> Fill:
    return Fill(sym=sym, side=side, qty=qty, price=price, ts=ts, realized_pnl=rp,
                commission=comm, multiplier=mult, strategy=strat, seq=seq)

def test_simple_long_round_trip() -> None:
    r = reconstruct_round_trips([_f("BUY",100,10.0,_t(0)), _f("SELL",100,11.0,_t(5),rp=100.0)])
    assert r.warnings == []
    assert len(r.round_trips) == 1
    rt = r.round_trips[0]
    assert rt.direction == "long" and rt.is_open is False and rt.suspect is False
    assert rt.qty == 100 and rt.avg_entry_price == 10.0 and rt.avg_exit_price == 11.0
    assert rt.realized_pnl == 100.0 and rt.commission == -2.0
    assert rt.hold_seconds == 300.0
    assert rt.entry_ts == _t(0) and rt.exit_ts == _t(5) and rt.strategy == "s1"

def test_partial_closes_group_into_one_round_trip() -> None:
    fills = [_f("BUY",100,10.0,_t(0)), _f("SELL",40,11.0,_t(3),rp=40.0),
             _f("SELL",60,12.0,_t(6),rp=120.0)]
    r = reconstruct_round_trips(fills)
    assert len(r.round_trips) == 1
    rt = r.round_trips[0]
    assert rt.qty == 100 and rt.realized_pnl == 160.0 and rt.is_open is False
    assert rt.avg_exit_price is not None
    assert abs(rt.avg_exit_price - 11.6) < 1e-9

def test_flip_through_zero_splits_into_two() -> None:
    fills = [_f("BUY",50,10.0,_t(0)), _f("SELL",80,11.0,_t(5),rp=25.0,comm=-8.0),
             _f("BUY",30,10.5,_t(9),rp=15.0,comm=-3.0)]
    r = reconstruct_round_trips(fills)
    assert len(r.round_trips) == 2
    long_rt, short_rt = r.round_trips
    assert long_rt.direction == "long" and long_rt.qty == 50 and long_rt.realized_pnl == 25.0
    assert abs(long_rt.commission - (-1.0 + -5.0)) < 1e-9
    assert short_rt.direction == "short" and short_rt.qty == 30 and short_rt.realized_pnl == 15.0
    assert short_rt.is_open is False
    assert abs(short_rt.commission - (-3.0 + -3.0)) < 1e-9

def test_still_open_position_is_flagged_open() -> None:
    r = reconstruct_round_trips([_f("BUY",100,10.0,_t(0)), _f("SELL",40,11.0,_t(3),rp=40.0)])
    assert len(r.round_trips) == 1
    rt = r.round_trips[0]
    assert rt.is_open is True and rt.exit_ts is None and rt.hold_seconds is None
    assert rt.realized_pnl == 40.0

def test_multi_symbol_isolation() -> None:
    fills = [_f("BUY",10,10.0,_t(0),sym="AAA"), _f("BUY",10,20.0,_t(1),sym="BBB"),
             _f("SELL",10,11.0,_t(2),sym="AAA",rp=10.0), _f("SELL",10,19.0,_t(3),sym="BBB",rp=-10.0)]
    r = reconstruct_round_trips(fills)
    syms = sorted(rt.sym for rt in r.round_trips)
    assert syms == ["AAA","BBB"] and len(r.round_trips) == 2

def test_suspect_when_realized_pnl_on_an_opening_fill() -> None:
    r = reconstruct_round_trips([_f("SELL",50,11.0,_t(0),rp=30.0), _f("BUY",50,10.0,_t(5),rp=0.0)])
    assert any(w["sym"] == "AAPL" and w["reason"] == "unmatched_close" for w in r.warnings)
    assert all(rt.suspect for rt in r.round_trips if rt.sym == "AAPL")

def test_same_timestamp_fills_use_seq_order() -> None:
    # SELL (the close, rp=100) appears FIRST in the input list but has a HIGHER
    # seq than the BUY, both at the same ts. seq ordering must fix what input-list
    # order alone would corrupt (a close processed before its open).
    fills = [_f("SELL",100,11.0,_t(0),rp=100.0,seq=1), _f("BUY",100,10.0,_t(0),seq=0)]
    r = reconstruct_round_trips(fills)
    assert len(r.round_trips) == 1
    rt = r.round_trips[0]
    assert rt.direction == "long" and rt.realized_pnl == 100.0
    assert rt.suspect is False
    assert r.warnings == []

def test_aggregate_basic_win_loss_stats() -> None:
    r = reconstruct_round_trips([
        _f("BUY",10,10.0,_t(0)), _f("SELL",10,12.0,_t(2),rp=20.0),      # +20 win (AAPL)
        _f("BUY",10,10.0,_t(3),sym="BBB",strat="s2"),
        _f("SELL",10,9.0,_t(4),sym="BBB",strat="s2",rp=-10.0),          # -10 loss (BBB)
    ])
    j = aggregate_round_trips(r, group_by="symbol")
    agg = j.aggregates
    assert agg["count"] == 2 and agg["wins"] == 1 and agg["losses"] == 1
    assert abs(agg["win_rate"] - 0.5) < 1e-9
    assert abs(agg["total_realized_pnl"] - 10.0) < 1e-9
    assert abs(agg["profit_factor"] - 2.0) < 1e-9
    assert abs(agg["expectancy"] - 5.0) < 1e-9
    assert "AAPL" in j.by_group and "BBB" in j.by_group
    assert j.by_group["AAPL"]["wins"] == 1 and j.by_group["BBB"]["losses"] == 1

def test_aggregate_group_by_strategy_and_excludes_open() -> None:
    r = reconstruct_round_trips([
        _f("BUY",10,10.0,_t(0),strat="s1"), _f("SELL",10,12.0,_t(2),strat="s1",rp=20.0),
        _f("BUY",10,10.0,_t(3),sym="BBB",strat="s2"),   # still open -> excluded from win/loss
    ])
    j = aggregate_round_trips(r, group_by="strategy")
    assert j.aggregates["count"] == 1
    assert len(j.open_round_trips) == 1
    assert set(j.by_group.keys()) == {"s1"}
    assert j.warnings == r.warnings

def test_aggregate_profit_factor_no_losses_is_none() -> None:
    r = reconstruct_round_trips([_f("BUY",10,10.0,_t(0)), _f("SELL",10,12.0,_t(2),rp=20.0)])
    j = aggregate_round_trips(r, group_by="symbol")
    assert j.aggregates["profit_factor"] is None

def test_aggregate_rejects_bad_group_by() -> None:
    import pytest
    r = reconstruct_round_trips([])
    with pytest.raises(ValueError, match="group_by"):
        aggregate_round_trips(r, group_by="venue")

def test_journal_result_as_dict_is_json_safe() -> None:
    r = reconstruct_round_trips([
        _f("BUY",10,10.0,_t(0)), _f("SELL",10,12.0,_t(2),rp=20.0),
        _f("BUY",10,10.0,_t(3),sym="BBB"),  # left open
    ])
    j = aggregate_round_trips(r, group_by="symbol")
    d = j.as_dict()
    assert isinstance(d["round_trips"], list) and len(d["round_trips"]) == 1
    assert isinstance(d["round_trips"][0]["entry_ts"], str)
    assert isinstance(d["open_round_trips"], list) and len(d["open_round_trips"]) == 1
    assert d["open_round_trips"][0]["exit_ts"] is None
    assert d["aggregates"]["count"] == 1
    assert "AAPL" in d["by_group"]
    assert d["warnings"] == []

def test_journal_result_render_is_human_readable() -> None:
    r = reconstruct_round_trips([
        _f("BUY",10,10.0,_t(0)), _f("SELL",10,12.0,_t(2),rp=20.0),
        _f("BUY",10,10.0,_t(3),sym="BBB",strat="s2"),
        _f("SELL",10,9.0,_t(4),sym="BBB",strat="s2",rp=-10.0),
    ])
    j = aggregate_round_trips(r, group_by="symbol")
    text = j.render()
    assert "Round-trip journal" in text
    assert "Win rate" in text
    assert "Total realized P&L" in text
    assert "AAPL" in text and "BBB" in text

def test_journal_result_render_notes_open_and_warnings() -> None:
    r = reconstruct_round_trips([_f("SELL",50,11.0,_t(0),rp=30.0), _f("BUY",50,10.0,_t(5),rp=0.0)])
    j = aggregate_round_trips(r, group_by="symbol")
    text = j.render()
    assert "Warnings" in text


# ---------------------------------------------------------------------------
# Net-of-commission stats (net = realized_pnl + commission; commission negative=cost)
# ---------------------------------------------------------------------------


def test_gross_win_becomes_net_loss_after_commission() -> None:
    """A round-trip that is gross-positive but net-negative after commission must
    count as a loss under net_win_rate while remaining a win under gross win_rate."""
    fills = [
        _f("BUY", 10, 10.0, _t(0), comm=-1.0),
        _f("SELL", 10, 10.5, _t(2), rp=5.0, comm=-10.0),
    ]
    r = reconstruct_round_trips(fills)
    rt = r.round_trips[0]
    assert rt.realized_pnl == 5.0 and rt.commission == -11.0  # gross win, heavy commission

    j = aggregate_round_trips(r, group_by="symbol")
    agg = j.aggregates

    # Gross view: a win.
    assert agg["count"] == 1
    assert agg["wins"] == 1 and agg["losses"] == 0
    assert agg["win_rate"] == pytest.approx(1.0)
    assert agg["total_realized_pnl"] == pytest.approx(5.0)

    # Net view: a loss (5.0 + (-11.0) == -6.0).
    assert agg["net_wins"] == 0 and agg["net_losses"] == 1
    assert agg["net_win_rate"] == pytest.approx(0.0)
    assert agg["total_net_pnl"] == pytest.approx(-6.0)
    assert agg["net_expectancy"] == pytest.approx(-6.0)
    assert agg["net_avg_loss"] == pytest.approx(-6.0)
    assert agg["net_avg_win"] is None


def test_net_profit_factor_and_expectancy_multi_round_trip() -> None:
    r = reconstruct_round_trips([
        _f("BUY", 10, 10.0, _t(0), comm=-1.0),
        _f("SELL", 10, 12.0, _t(2), rp=20.0, comm=-1.0),            # gross +20, net +18: win
        _f("BUY", 10, 10.0, _t(3), sym="BBB", strat="s2", comm=-1.0),
        _f("SELL", 10, 9.0, _t(4), sym="BBB", strat="s2", rp=-10.0, comm=-1.0),  # net -12: loss
    ])
    j = aggregate_round_trips(r, group_by="symbol")
    agg = j.aggregates
    assert agg["total_net_pnl"] == pytest.approx(18.0 + -12.0)
    assert agg["net_wins"] == 1 and agg["net_losses"] == 1
    assert agg["net_profit_factor"] == pytest.approx(18.0 / 12.0)
    assert agg["net_expectancy"] == pytest.approx((18.0 - 12.0) / 2)


def test_stats_empty_includes_net_fields() -> None:
    from ibda.analytics.roundtrips import _stats

    stats = _stats([])
    assert stats["net_win_rate"] is None
    assert stats["total_net_pnl"] == 0.0
    assert stats["net_profit_factor"] is None
    assert stats["net_expectancy"] is None


def test_render_shows_gross_and_net_labels() -> None:
    r = reconstruct_round_trips([
        _f("BUY", 10, 10.0, _t(0), comm=-1.0),
        _f("SELL", 10, 10.5, _t(2), rp=5.0, comm=-10.0),
    ])
    j = aggregate_round_trips(r, group_by="symbol")
    text = j.render()
    assert "Net win rate" in text
    assert "Total net P&L" in text
    assert "Net profit factor" in text
    assert "Net expectancy" in text


# ---------------------------------------------------------------------------
# Realized P&L booked on a partially-closed position must not vanish
# ---------------------------------------------------------------------------


def test_partial_close_realized_pnl_is_reported_not_dropped() -> None:
    """A partial close books real money even though the trade is still open.

    `aggregate_round_trips` excludes open round-trips — correctly, since win-rate /
    profit-factor / expectancy need completed trades. But its docstring justified that with
    "their P&L is unrealized", which is false: `_add_exit` accumulates IBKR's
    `fifoPnlRealized` on every partial close.

    BUY 100 @10 then SELL 50 @15 books $250 of realized P&L and $2 of commission on a
    round-trip that is still open. Without the dedicated fields, that money is reachable
    only by reading the open round-trip object and `aggregates` reports 0.0 — so a journal
    cannot reconcile against the account's realized P&L for any book holding a
    partially-closed position.
    """
    r = reconstruct_round_trips([
        _f("BUY", 100, 10.0, _t(0)),
        _f("SELL", 50, 15.0, _t(5), rp=250.0),
    ])
    rt = r.round_trips[0]
    assert rt.is_open is True
    assert rt.realized_pnl == 250.0, "the partial close booked real money"

    j = aggregate_round_trips(r)
    # The completed-trade statistics correctly see nothing: no trade has finished.
    assert j.aggregates["count"] == 0
    assert j.aggregates["total_realized_pnl"] == 0.0
    # ...but the money is surfaced rather than dropped.
    assert j.open_realized_pnl == 250.0
    assert j.open_commission == -2.0


def test_open_totals_are_zero_when_no_position_is_partially_closed() -> None:
    """A fully-closed book reports nothing on the open channel — no double-counting."""
    j = aggregate_round_trips(reconstruct_round_trips([
        _f("BUY", 100, 10.0, _t(0)),
        _f("SELL", 100, 11.0, _t(5), rp=100.0),
    ]))
    assert j.aggregates["total_realized_pnl"] == 100.0
    assert j.open_realized_pnl == 0.0
    assert j.open_commission == 0.0


def test_open_totals_sum_across_several_partially_closed_positions() -> None:
    """Two names each partially closed — both contribute."""
    j = aggregate_round_trips(reconstruct_round_trips([
        _f("BUY", 100, 10.0, _t(0), sym="AAPL"),
        _f("SELL", 50, 15.0, _t(1), rp=250.0, sym="AAPL"),
        _f("BUY", 80, 20.0, _t(2), sym="MSFT"),
        _f("SELL", 40, 22.0, _t(3), rp=80.0, sym="MSFT"),
    ]))
    assert j.aggregates["count"] == 0
    assert j.open_realized_pnl == 330.0
