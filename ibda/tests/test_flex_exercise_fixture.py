"""Guard the option-exercise fixture's published numbers against silent drift.

``report_exercise.xml`` exists to be run by readers: the IBKR Quant article prints a
clone-and-run command against it and quotes its output verbatim. Nothing else in the
suite touches the fixture, so before this file a change to round-trip reconstruction
could alter those numbers with every test still green and the article quietly wrong.

What is asserted here is the article's claim, not merely the code's current behavior:

1. The five round trips and the $37,537.37 total, and the per-symbol split beneath it.
   Cross-checked against the fixture's own ``FIFOPerformanceSummaryInBase``, which IBKR
   computes independently of the ``Trades`` rows the journal reads.
2. The AVAV exercise split: $34,004.85 on the stock leg, $2,739.33 on the option leg.
   The exercise itself realizes nothing on the 19 contracts it consumes, because their
   premium is already inside the stock's cost basis.
3. That deriving the stock leg from trade prices instead of ``fifoPnlRealized`` reads
   $48,443.57, which is $14,438.72 too high. This is the article's headline figure and
   the reason the fixture is shaped the way it is.
4. In the public repo, that ``examples/roundtrip_pnl_ibda.py`` still prints the exact
   block the article shows. Skipped in the monorepo, which has no ``examples/``.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from ibda.adapters.ibkr.flex.arrow import flex_arrow_tables
from ibda.adapters.ibkr.flex.parse import parse_statement_or_raise
from ibda.analytics.roundtrips import Fill, aggregate_round_trips, reconstruct_round_trips

_FIXTURE = Path(__file__).parent / "fixtures" / "flex" / "report_exercise.xml"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLE = _REPO_ROOT / "examples" / "roundtrip_pnl_ibda.py"

#: The stock and option legs of the AVAV exercise, as IBKR books them.
_AVAV_STK = "AVAV"
_AVAV_OPT = "AVAV  260702C00142000"

#: IBKR's exercise stamp, ``16:20:00`` US/Eastern in the raw Flex ``dateTime``, as the
#: canonical schema carries it. Executions normalize to UTC, so the tell to match on is
#: 20:20 rather than the 16:20 a reader sees in the XML.
_EXERCISE_UTC = "20:20:00"

#: Cent-scale tolerance. The fixture states each symbol's P&L twice — once as a sum of
#: ``fifoPnlRealized`` across trade rows, once in ``FIFOPerformanceSummaryInBase`` — and
#: IBKR rounds the summary more coarsely, so the two disagree by up to ~4e-6. That is
#: four ten-thousandths of a cent and cannot move a published figure; anything larger is
#: real drift.
_TOL = 1e-4

#: Exactly what the article prints, byte for byte.
_PUBLISHED = """\
5 closed round-trips   total realized P&L $37,537.37
  AVAV     $   34,004.85  (1 trips)
  AVAV  260702C00142000 $    2,739.33  (1 trips)
  TGT      $      890.16  (1 trips)
  JPM      $      -96.97  (2 trips)
"""


def _fills() -> list[Fill]:
    """Rebuild the ledger the way ``examples/roundtrip_pnl_ibda.py`` does."""
    sections = parse_statement_or_raise(_FIXTURE.read_text(encoding="utf-8"))
    execs = flex_arrow_tables(sections)["execution"].to_pandas()
    return [
        Fill(
            sym=str(row.Sym), side=str(row.Side),
            qty=float(row.Qty), price=float(row.Price),
            ts=row.Timestamp.to_pydatetime(), realized_pnl=float(row.RealizedPnl or 0.0),
            commission=float(row.Commission or 0.0),
            multiplier=float(row.Multiplier) if row.Multiplier else 1.0,
            strategy=str(row.OrderRef) if row.OrderRef else "unattributed", seq=i,
        )
        for i, row in enumerate(execs.itertuples()) if row.Sym
    ]


def _by_symbol() -> dict[str, dict[str, float]]:
    journal = aggregate_round_trips(reconstruct_round_trips(_fills()), group_by="symbol")
    return {sym: dict(stats) for sym, stats in journal.by_group.items()}


def _aggregates() -> dict[str, float]:
    journal = aggregate_round_trips(reconstruct_round_trips(_fills()), group_by="symbol")
    return dict(journal.aggregates)


# ---------------------------------------------------------------------------
# The totals the article quotes
# ---------------------------------------------------------------------------


def test_five_closed_round_trips() -> None:
    """Four symbols, five trips: JPM round-trips twice, everything else once."""
    assert _aggregates()["count"] == 5


def test_total_realized_pnl() -> None:
    """The $37,537.37 headline. Rounds to the cent the article prints."""
    assert f"${_aggregates()['total_realized_pnl']:,.2f}" == "$37,537.37"


@pytest.mark.parametrize(
    "sym,realized,trips",
    [
        (_AVAV_STK, 34004.85089, 1),
        (_AVAV_OPT, 2739.32811, 1),
        ("TGT", 890.15794, 1),
        ("JPM", -96.96502, 2),
    ],
)
def test_per_symbol_split(sym: str, realized: float, trips: int) -> None:
    """Each line of the published table, to the precision the fixture carries."""
    stats = _by_symbol()[sym]
    assert stats["count"] == trips
    assert abs(stats["total_realized_pnl"] - realized) < _TOL


def test_symbols_reconcile_to_the_total() -> None:
    """The four legs sum to the headline; no symbol is dropped or double-counted."""
    total = sum(s["total_realized_pnl"] for s in _by_symbol().values())
    assert abs(total - _aggregates()["total_realized_pnl"]) < _TOL


def test_journal_matches_the_brokers_own_summary() -> None:
    """Cross-check against ``FIFOPerformanceSummaryInBase``.

    IBKR computes that section itself, so it is an independent statement of the same
    P&L rather than a restatement of the ``Trades`` rows the journal reads. If the two
    ever disagree, the journal is wrong and the fixture is not.
    """
    xml = _FIXTURE.read_text(encoding="utf-8")
    broker = {
        attrs["symbol"]: float(attrs["realizedTotal"])
        for attrs in (
            dict(re.findall(r'(\w+)="([^"]*)"', row))
            for row in re.findall(r"<FIFOPerformanceSummaryUnderlying[^>]*/>", xml)
        )
    }
    journal = {sym: stats["total_realized_pnl"] for sym, stats in _by_symbol().items()}
    assert set(journal) == set(broker)
    for sym, realized in broker.items():
        assert abs(journal[sym] - realized) < _TOL, f"{sym} disagrees with the broker"


# ---------------------------------------------------------------------------
# The exercise itself: why the fixture exists
# ---------------------------------------------------------------------------


def test_exercise_realizes_nothing_on_the_consumed_contracts() -> None:
    """The 19 exercised contracts book $0.00, priced at zero on a 16:20:00 stamp.

    The option leg's whole $2,739.33 comes from the single contract genuinely sold on
    the open market five days earlier, not from the exercise.
    """
    exercised = [
        f for f in _fills()
        if f.sym == _AVAV_OPT and f.ts.strftime("%H:%M:%S") == _EXERCISE_UTC
    ]
    assert len(exercised) == 1
    assert exercised[0].qty == 19.0
    assert exercised[0].price == 0.0
    assert exercised[0].realized_pnl == 0.0


def test_trade_price_derivation_overstates_the_stock_leg() -> None:
    """The article's $14,438.72.

    Re-deriving the stock leg from proceeds and commissions, the way a journal that
    ignores ``fifoPnlRealized`` would, reads $48,443.57 against the broker's
    $34,004.85. The gap is the option premium that rolled into the stock's basis at
    exercise: real money, sitting on the wrong leg, invisible in the total.
    """
    stock = [f for f in _fills() if f.sym == _AVAV_STK]
    signed = sum(
        (f.qty * f.price if f.side == "SELL" else -f.qty * f.price) + f.commission
        for f in stock
    )
    broker = _by_symbol()[_AVAV_STK]["total_realized_pnl"]
    assert f"${signed:,.2f}" == "$48,443.57"
    assert f"${signed - broker:,.2f}" == "$14,438.72"


# ---------------------------------------------------------------------------
# The command as published
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _EXAMPLE.exists(),
    reason="examples/ ships in the public ibda repo only; the monorepo has no copy",
)
def test_published_command_prints_the_published_output() -> None:
    """Run the article's command and compare stdout to the block it shows.

    This is the assertion that fails if the script's formatting drifts rather than its
    arithmetic. Everything above would stay green through a changed format string.
    """
    proc = subprocess.run(
        [sys.executable, str(_EXAMPLE), str(_FIXTURE)],
        cwd=_REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == _PUBLISHED
