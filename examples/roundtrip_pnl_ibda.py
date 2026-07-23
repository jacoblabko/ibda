"""Round-trip realized P&L (FIFO) from Flex fills, via ibda's round-trip reconstruction.
Run: uv run python examples/roundtrip_pnl_ibda.py <flex_activity.xml>  (offline; no live connection).

Uses the engine-free canonical path (ibda.adapters.ibkr.flex.arrow.flex_arrow_tables):
Flex XML -> canonical Arrow `execution` table, no Deephaven/JVM needed, matching this
script's "offline, no live connection" nature. `ibda.load_flex_file` is the other public
entry point onto the same canonical schema, but it starts an embedded Deephaven engine
(a real Server + JVM boot) -- unnecessary weight for a one-shot read of a Flex file.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when run as a plain script (examples/ is sys.path[0],
# not the repo root, for a script nested one level under the repo root).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ibda.adapters.ibkr.flex.arrow import flex_arrow_tables  # noqa: E402
from ibda.adapters.ibkr.flex.parse import parse_statement_or_raise  # noqa: E402
from ibda.analytics.roundtrips import Fill, aggregate_round_trips, reconstruct_round_trips  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: roundtrip_pnl_ibda.py <flex_activity.xml>")

    with open(sys.argv[1], encoding="utf-8") as f:
        xml = f.read()
    sections = parse_statement_or_raise(xml)
    execs = flex_arrow_tables(sections)["execution"].to_pandas()

    # Canonical `execution` table has everything Fill needs: Sym/Side/Qty/Price/
    # Timestamp, RealizedPnl (IBKR's own fifoPnlRealized), Commission/Multiplier, and
    # OrderRef (Flex orderReference) for strategy attribution -- all via the mapper.
    fills = [
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

    journal = aggregate_round_trips(reconstruct_round_trips(fills), group_by="symbol")
    agg = journal.aggregates
    print(f"{agg['count']} closed round-trips   total realized P&L ${agg['total_realized_pnl']:,.2f}")
    top = sorted(journal.by_group.items(),
                 key=lambda kv: abs(kv[1]["total_realized_pnl"]), reverse=True)[:5]
    for sym, stats in top:
        print(f"  {sym:<8} ${stats['total_realized_pnl']:>12,.2f}  ({stats['count']} trips)")


if __name__ == "__main__":
    main()
