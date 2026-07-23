"""Command-line Sharpe / performance calculator: ``python -m ibda``.

The shareable, zero-code path. Point it at a Flex Activity XML report and get a
plain-text performance summary (Sharpe, Sortino, returns, volatility, drawdown)::

    python -m ibda activity.xml
    python -m ibda activity.xml --risk-free 0.05
    python -m ibda activity.xml --account U1234567 --json

No compute engine is started — it is pure parse → arithmetic. The Flex query must
include the daily equity summary (``EquitySummaryByReportDateInBase``).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from ibda.rates import DEFAULT_PERIODS_PER_YEAR


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m ibda",
        description=(
            "Compute the Sharpe ratio and basic performance/risk metrics from an "
            "IBKR Flex Activity XML report. The report must include the daily "
            "equity summary (EquitySummaryByReportDateInBase)."
        ),
    )
    p.add_argument("report", help="path to a Flex Activity XML report")
    p.add_argument(
        "--risk-free",
        default="auto",
        metavar="RATE",
        help=(
            "annual risk-free rate as a decimal, e.g. 0.05 for 5%%, or 'auto' "
            "(default) for the documented offline 3-month US Treasury proxy "
            "(ibda.rates.DEFAULT_RISK_FREE_ANNUAL; no network call)"
        ),
    )
    p.add_argument(
        "--periods",
        type=int,
        default=DEFAULT_PERIODS_PER_YEAR,
        metavar="N",
        help="periods per year for annualization (default 252 trading days)",
    )
    p.add_argument(
        "--account",
        default=None,
        metavar="ID",
        help="account id to select when the report covers multiple accounts",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="emit JSON instead of the human-readable report",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code (0 ok, 2 on a handled error)."""
    args = _build_parser().parse_args(argv)

    import ibda
    from ibda.errors import FlexParseError
    from ibda.rates import resolve_risk_free

    try:
        risk_free, rf_source = resolve_risk_free(args.risk_free)
    except ValueError:
        print(f"error: invalid --risk-free value {args.risk_free!r}", file=sys.stderr)
        return 2
    # Provenance goes to stderr so stdout stays clean (e.g. for --json piping).
    print(f"risk-free: {risk_free * 100:.2f}% ({rf_source})", file=sys.stderr)

    try:
        perf = ibda.flex_performance(
            args.report,
            risk_free_annual=risk_free,
            periods_per_year=args.periods,
            account=args.account,
        )
    except (FlexParseError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(perf.as_dict(), indent=2))
    else:
        print(perf.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
