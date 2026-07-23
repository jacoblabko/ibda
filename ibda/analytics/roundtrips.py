"""
ibda/analytics/roundtrips.py

Pure round-trip reconstruction over a fill ledger. No IO, no engine, no TWS (same
purity contract as ibda/analytics/performance.py). A round-trip is a flat->open->
flat cycle for one symbol. Realized P&L is taken from each fill's IBKR-authoritative
fifoPnlRealized (Fill.realized_pnl); we group and bucket, we do not recompute lot P&L.
This grouping/bucketing rule was specified as a standalone design (dated 2026-07-01) before
implementation; the docstrings in this module are the complete, current statement of that
contract.

Ordering contract: input fills must be in execution order. Flex ``dateTime`` is
second-resolution, so multiple fills commonly share the same ``ts``; when timestamps
tie, ``Fill.seq`` breaks the tie. Callers should pass a monotonic execution-sequence
index (e.g. Flex report row order) via ``seq`` rather than relying on list order alone.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class Fill:
    """A single execution (partial or full) feeding round-trip reconstruction."""

    sym: str
    side: str            # "BUY" | "SELL"
    qty: float           # > 0 (absolute)
    price: float
    ts: datetime         # tz-aware UTC
    realized_pnl: float  # fifoPnlRealized for this fill; 0.0 if it realized nothing
    commission: float    # ib_commission; negative = cost
    multiplier: float    # contract multiplier (1.0 for stocks)
    strategy: str        # OrderRef / orderReference, or "unattributed"
    seq: int = 0         # monotonic execution-sequence index; breaks ts ties


@dataclass(frozen=True)
class RoundTrip:
    """A reconstructed flat->open->flat (or still-open) position cycle for one symbol."""

    sym: str
    direction: str            # "long" | "short"
    entry_ts: datetime
    exit_ts: datetime | None
    hold_seconds: float | None
    qty: float                # total opened quantity (absolute)
    avg_entry_price: float
    avg_exit_price: float | None
    realized_pnl: float
    commission: float
    return_pct: float | None  # realized_pnl / (qty * avg_entry_price * multiplier)
    strategy: str
    is_open: bool
    # per-round-trip echo of a symbol gap seen at/after this round-trip; see
    # ReconstructResult.warnings for the authoritative symbol-level signal
    suspect: bool


@dataclass(frozen=True)
class ReconstructResult:
    """Output of :func:`reconstruct_round_trips`: all round-trips plus symbol-level warnings."""

    round_trips: list[RoundTrip]
    # Authoritative symbol-level gap signal (one entry per symbol with an
    # unmatched-close detected anywhere in its fill sequence). ``RoundTrip.suspect``
    # is a per-round-trip echo: it is set only on round-trips finalized at or after
    # the gap was seen, so an earlier, clean round-trip on the same symbol may show
    # ``suspect=False`` even though the symbol still appears here.
    warnings: list[dict[str, str]]


@dataclass
class _OpenRT:
    sym: str
    direction: str
    entry_ts: datetime
    strategy: str
    multiplier: float
    entry_qty: float = 0.0
    _entry_notional: float = 0.0
    exit_qty: float = 0.0
    _exit_notional: float = 0.0
    realized_pnl: float = 0.0
    commission: float = 0.0
    exit_ts: datetime | None = None


def _signed(f: Fill) -> float:
    return f.qty if f.side.upper() == "BUY" else -f.qty


def _add_entry(rt: _OpenRT, price: float, qty: float, commission: float) -> None:
    rt.entry_qty += qty
    rt._entry_notional += price * qty
    rt.commission += commission


def _add_exit(rt: _OpenRT, price: float, qty: float, realized: float, commission: float) -> None:
    rt.exit_qty += qty
    rt._exit_notional += price * qty
    rt.realized_pnl += realized
    rt.commission += commission


def _finalize(rt: _OpenRT, *, is_open: bool, suspect: bool) -> RoundTrip:
    avg_entry = rt._entry_notional / rt.entry_qty if rt.entry_qty else 0.0
    avg_exit = (rt._exit_notional / rt.exit_qty) if rt.exit_qty else None
    hold = ((rt.exit_ts - rt.entry_ts).total_seconds()
            if (rt.exit_ts is not None and not is_open) else None)
    entry_notional = rt.entry_qty * avg_entry * rt.multiplier
    return_pct = (rt.realized_pnl / entry_notional) if (entry_notional and not is_open) else None
    return RoundTrip(
        sym=rt.sym, direction=rt.direction, entry_ts=rt.entry_ts,
        exit_ts=(None if is_open else rt.exit_ts), hold_seconds=hold,
        qty=rt.entry_qty, avg_entry_price=avg_entry, avg_exit_price=avg_exit,
        realized_pnl=rt.realized_pnl, commission=rt.commission,
        return_pct=return_pct, strategy=rt.strategy, is_open=is_open, suspect=suspect,
    )


def reconstruct_round_trips(fills: Sequence[Fill]) -> ReconstructResult:
    """Reconstruct round-trips from an execution-ordered fill ledger, per symbol.

    Fills are grouped by ``sym`` and, within each symbol, sorted by ``(ts, seq)`` —
    ``seq`` breaks ties when Flex's second-resolution ``dateTime`` puts multiple fills
    at the same timestamp (see the module docstring's ordering contract). Each symbol's
    position then walks flat -> open -> flat: the first fill from flat opens a new
    round-trip; same-direction fills add to it; an opposite-direction fill reduces or
    closes it (partial close accumulates against the same round-trip; an exact or
    over-sized close finalizes it). A fill that closes the existing position and still
    has quantity left over flips through zero — it is split so the closing quantity
    (and a pro-rated share of its commission) finalizes the old round-trip, and the
    remainder (with the rest of the commission) opens a new one in the flipped
    direction. A symbol with a nonzero residual position after its last fill produces
    one trailing **open** round-trip (``is_open=True``, ``exit_ts=None``).

    A symbol is flagged ``suspect`` (both in ``ReconstructResult.warnings`` and echoed
    on every round-trip finalized at or after the gap) when a fill contributes nonzero
    ``realized_pnl`` while *opening* or *adding to* a position — IBKR's
    ``fifoPnlRealized`` should only be nonzero on a reducing fill, so this signals an
    unmatched close (e.g. the fill ledger doesn't start from a flat position).

    Args:
        fills: fills for one or more symbols, in execution order. Ties in ``ts`` are
            broken by ``Fill.seq`` — pass a monotonic execution-sequence index (e.g.
            Flex report row order) rather than relying on list order alone.

    Returns:
        A :class:`ReconstructResult` with every closed and still-open round-trip
        (in per-symbol chronological order, grouped by symbol) and the symbol-level
        ``warnings`` for any detected unmatched-close gap.
    """
    by_sym: dict[str, list[Fill]] = {}
    for f in fills:
        by_sym.setdefault(f.sym, []).append(f)

    round_trips: list[RoundTrip] = []
    warnings: list[dict[str, str]] = []
    _EPS = 1e-9

    for sym, sym_fills in by_sym.items():
        ordered = sorted(sym_fills, key=lambda f: (f.ts, f.seq))
        pos = 0.0
        rt: _OpenRT | None = None
        sym_suspect = False

        for f in ordered:
            s = _signed(f)
            if abs(pos) < _EPS:                              # OPEN a new round-trip
                if abs(f.realized_pnl) > _EPS:
                    sym_suspect = True
                rt = _OpenRT(sym=sym, direction=("long" if s > 0 else "short"),
                             entry_ts=f.ts, strategy=f.strategy, multiplier=f.multiplier)
                _add_entry(rt, f.price, f.qty, f.commission)
                pos = s
            elif (pos > 0) == (s > 0):                       # ADD to the position
                if abs(f.realized_pnl) > _EPS:
                    sym_suspect = True
                assert rt is not None
                _add_entry(rt, f.price, f.qty, f.commission)
                pos += s
            else:                                            # REDUCE / CLOSE / FLIP
                assert rt is not None
                if f.qty <= abs(pos) + _EPS:                 # partial or exact close
                    _add_exit(rt, f.price, f.qty, f.realized_pnl, f.commission)
                    pos += s
                    if abs(pos) < _EPS:
                        rt.exit_ts = f.ts
                        round_trips.append(_finalize(rt, is_open=False, suspect=sym_suspect))
                        rt = None
                else:                                        # FLIP through zero
                    closing_qty = abs(pos)
                    opening_qty = f.qty - closing_qty
                    close_frac = closing_qty / f.qty
                    _add_exit(rt, f.price, closing_qty, f.realized_pnl, f.commission * close_frac)
                    rt.exit_ts = f.ts
                    round_trips.append(_finalize(rt, is_open=False, suspect=sym_suspect))
                    rt = _OpenRT(sym=sym, direction=("long" if s > 0 else "short"),
                                 entry_ts=f.ts, strategy=f.strategy, multiplier=f.multiplier)
                    _add_entry(rt, f.price, opening_qty, f.commission * (1.0 - close_frac))
                    pos += s

        if rt is not None:                                   # residual -> open round-trip
            round_trips.append(_finalize(rt, is_open=True, suspect=sym_suspect))
        if sym_suspect:
            warnings.append({"sym": sym, "reason": "unmatched_close"})

    return ReconstructResult(round_trips=round_trips, warnings=warnings)


@dataclass(frozen=True)
class JournalResult:
    """Output of :func:`aggregate_round_trips`: grouped round-trip stats and totals."""

    round_trips: list[RoundTrip]
    open_round_trips: list[RoundTrip]
    aggregates: dict[str, Any]
    by_group: dict[str, dict[str, Any]]
    warnings: list[dict[str, str]]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict (round-trip datetimes rendered as ISO-8601 strings)."""

        def _rt_json(rt: RoundTrip) -> dict[str, Any]:
            out: dict[str, Any] = asdict(rt)
            out["entry_ts"] = rt.entry_ts.isoformat()
            out["exit_ts"] = rt.exit_ts.isoformat() if rt.exit_ts is not None else None
            return out

        return {
            "round_trips": [_rt_json(rt) for rt in self.round_trips],
            "open_round_trips": [_rt_json(rt) for rt in self.open_round_trips],
            "aggregates": dict(self.aggregates),
            "by_group": {k: dict(v) for k, v in self.by_group.items()},
            "warnings": [dict(w) for w in self.warnings],
        }

    def render(self) -> str:
        """Render a plain-text round-trip journal report suitable for sharing.

        Shows both the gross (``fifoPnlRealized``-only) figures and the
        commission-inclusive ``net_*`` figures side by side.
        """

        def pct(x: Any) -> str:
            return "n/a" if x is None else f"{x * 100:+.1f}%"

        def num(x: Any) -> str:
            return "n/a" if x is None else f"{x:.2f}"

        def money(x: Any) -> str:
            return "n/a" if x is None else f"${x:,.2f}"

        agg = self.aggregates
        lines = [
            "Round-trip journal",
            f"  Closed round-trips   {agg.get('count', 0)}   "
            f"(open: {len(self.open_round_trips)})",
            f"  Win rate             {pct(agg.get('win_rate'))}"
            f"   ({agg.get('wins', 0)} wins / {agg.get('losses', 0)} losses)   [gross]",
            f"  Net win rate         {pct(agg.get('net_win_rate'))}"
            f"   ({agg.get('net_wins', 0)} wins / {agg.get('net_losses', 0)} losses)   [net]",
            f"  Total realized P&L   {money(agg.get('total_realized_pnl'))}   [gross]",
            f"  Total net P&L        {money(agg.get('total_net_pnl'))}   [net of commission]",
            f"  Total commission     {money(agg.get('total_commission'))}",
            f"  Avg win / loss       {money(agg.get('avg_win'))} / {money(agg.get('avg_loss'))}"
            "   [gross]",
            f"  Net avg win / loss   {money(agg.get('net_avg_win'))} / "
            f"{money(agg.get('net_avg_loss'))}   [net]",
            f"  Profit factor        {num(agg.get('profit_factor'))}   [gross]",
            f"  Net profit factor    {num(agg.get('net_profit_factor'))}   [net]",
            f"  Expectancy           {money(agg.get('expectancy'))}   [gross]",
            f"  Net expectancy       {money(agg.get('net_expectancy'))}   [net]",
        ]
        if self.by_group:
            lines.append("")
            lines.append("  By group:")
            for key, stats in self.by_group.items():
                lines.append(
                    f"    {key:<12} count={stats.get('count', 0):<4} "
                    f"win_rate={pct(stats.get('win_rate'))}  "
                    f"pnl={money(stats.get('total_realized_pnl'))}"
                )
        if self.warnings:
            lines.append("")
            lines.append(
                f"  Warnings: {len(self.warnings)} symbol(s) with an unmatched close "
                "(see .warnings for detail)."
            )
        lines.append("")
        lines.append(
            "  Win rate = fraction of closed round-trips with positive realized P&L;"
        )
        lines.append(
            "  profit factor = gross wins / |gross losses|; expectancy = mean P&L per trade."
        )
        lines.append(
            "  [gross] uses fifoPnlRealized only; [net] adds commission "
            "(net = realized_pnl + commission)."
        )
        return "\n".join(lines)


def _net_pnl(rt: RoundTrip) -> float:
    """Round-trip P&L net of commission. ``commission`` is negative = cost, so
    net = realized_pnl + commission (the fixed cross-cutting sign convention)."""
    return rt.realized_pnl + rt.commission


def _stats(closed: list[RoundTrip]) -> dict[str, Any]:
    """Win/loss/expectancy stats over *closed* round-trips, both gross and net of commission.

    The un-prefixed fields (``win_rate``, ``avg_win``, ``profit_factor``, ``expectancy``,
    ``total_realized_pnl``, …) are **gross** of commission — they use ``RoundTrip.realized_pnl``
    (IBKR's ``fifoPnlRealized``) directly, never netted against ``commission``. The ``net_*``
    fields are the commission-inclusive analogues (``realized_pnl + commission``; commission
    is negative = cost), so a round-trip that is gross-positive but net-negative after
    commission counts as a loss under ``net_win_rate`` while remaining a win under the
    gross ``win_rate``.
    """
    n = len(closed)
    if n == 0:
        return {"count": 0, "wins": 0, "losses": 0, "win_rate": None, "avg_win": None,
                "avg_loss": None, "profit_factor": None, "expectancy": None,
                "total_realized_pnl": 0.0, "total_commission": 0.0,
                "avg_hold_seconds": None, "largest_win": None, "largest_loss": None,
                "net_wins": 0, "net_losses": 0, "net_win_rate": None, "net_avg_win": None,
                "net_avg_loss": None, "net_profit_factor": None, "net_expectancy": None,
                "total_net_pnl": 0.0}
    wins = [rt for rt in closed if rt.realized_pnl > 0]
    losses = [rt for rt in closed if rt.realized_pnl < 0]
    gross_win = sum(rt.realized_pnl for rt in wins)
    gross_loss = sum(rt.realized_pnl for rt in losses)   # <= 0
    holds = [rt.hold_seconds for rt in closed if rt.hold_seconds is not None]
    total_pnl = sum(rt.realized_pnl for rt in closed)

    net_pnls = [_net_pnl(rt) for rt in closed]
    net_wins = [np for np in net_pnls if np > 0]
    net_losses = [np for np in net_pnls if np < 0]
    net_gross_win = sum(net_wins)
    net_gross_loss = sum(net_losses)   # <= 0
    total_net_pnl = sum(net_pnls)

    return {
        "count": n, "wins": len(wins), "losses": len(losses),
        "win_rate": len(wins) / n,
        "avg_win": (gross_win / len(wins)) if wins else None,
        "avg_loss": (gross_loss / len(losses)) if losses else None,
        "profit_factor": (gross_win / abs(gross_loss)) if gross_loss != 0 else None,
        "expectancy": total_pnl / n,
        "total_realized_pnl": total_pnl,
        "total_commission": sum(rt.commission for rt in closed),
        "avg_hold_seconds": (sum(holds) / len(holds)) if holds else None,
        "largest_win": max((rt.realized_pnl for rt in wins), default=None),
        "largest_loss": min((rt.realized_pnl for rt in losses), default=None),
        # --- net-of-commission analogues (net = realized_pnl + commission) ---
        "net_wins": len(net_wins), "net_losses": len(net_losses),
        "net_win_rate": len(net_wins) / n,
        "net_avg_win": (net_gross_win / len(net_wins)) if net_wins else None,
        "net_avg_loss": (net_gross_loss / len(net_losses)) if net_losses else None,
        "net_profit_factor": (net_gross_win / abs(net_gross_loss)) if net_gross_loss != 0 else None,
        "net_expectancy": total_net_pnl / n,
        "total_net_pnl": total_net_pnl,
    }


def aggregate_round_trips(result: ReconstructResult, *, group_by: str = "symbol") -> JournalResult:
    """Group closed round-trips and compute win-rate / profit-factor / expectancy etc.

    Each returned stats dict (``aggregates`` and each ``by_group`` entry) carries both
    the gross (commission-excluded, ``fifoPnlRealized``-only) fields and the ``net_*``
    commission-inclusive analogues — see :func:`_stats`.

    group_by ∈ {"symbol","strategy"}. Open round-trips are excluded from all aggregates
    (their P&L is unrealized) and returned separately.
    """
    if group_by not in ("symbol", "strategy"):
        raise ValueError(f"group_by must be 'symbol' or 'strategy', got {group_by!r}")
    closed = [rt for rt in result.round_trips if not rt.is_open]
    open_rts = [rt for rt in result.round_trips if rt.is_open]
    key: Callable[[RoundTrip], str] = (
        (lambda rt: rt.sym) if group_by == "symbol" else (lambda rt: rt.strategy)
    )
    groups: dict[str, list[RoundTrip]] = {}
    for rt in closed:
        groups.setdefault(key(rt), []).append(rt)
    by_group = {k: _stats(v) for k, v in sorted(groups.items())}
    return JournalResult(round_trips=closed, open_round_trips=open_rts,
                         aggregates=_stats(closed), by_group=by_group, warnings=result.warnings)
