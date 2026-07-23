"""ibda.analytics — derived analytics over canonical NAV data.

Performance (Sharpe/Sortino/Calmar/drawdown), time-series (rolling/periodic),
benchmark-relative metrics (beta/alpha/tracking-error), and round-trip trade-journal
reconstruction (win-rate/profit-factor/expectancy). Pure module: no engine, no
vendor SDK — only stdlib + pyarrow + ibda schema primitives.
"""

from __future__ import annotations

from ibda.analytics.benchmark import (
    RelativeSummary,
    relative_metrics,
    relative_summary,
    rolling_relative,
)
from ibda.analytics.performance import (
    PerformanceSummary,
    compute_performance,
    daily_returns,
    external_flows_from_cash,
    performance_summary,
    sharpe_ratio,
)
from ibda.analytics.roundtrips import (
    Fill,
    JournalResult,
    ReconstructResult,
    RoundTrip,
    aggregate_round_trips,
    reconstruct_round_trips,
)
from ibda.analytics.timeseries import periodic_returns, rolling_performance

__all__ = [
    # Performance
    "PerformanceSummary",
    "compute_performance",
    "daily_returns",
    "external_flows_from_cash",
    "performance_summary",
    "sharpe_ratio",
    # Time-series
    "periodic_returns",
    "rolling_performance",
    # Benchmark-relative
    "RelativeSummary",
    "relative_metrics",
    "relative_summary",
    "rolling_relative",
    # Round-trips / trade journal
    "Fill",
    "RoundTrip",
    "ReconstructResult",
    "JournalResult",
    "reconstruct_round_trips",
    "aggregate_round_trips",
]
