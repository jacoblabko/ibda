"""ibda — engine-hidden, vendor-neutral data-access API.

Connect once, get canonical data as Apache Arrow via a small, stable surface::

    port = ibda.connect(tables)
    arrow = port.table("position").snapshot()
    stream = port.table("quote").subscribe(my_callback)
    ibda.watch(port.table("quote"), my_cb, predicate="Last > 150")  # no-poll alert

Load a Flex XML report directly::

    port = ibda.load_flex_file("reports/activity.xml")
    arrow = port.table("execution").snapshot()
    df = arrow.to_pandas()

Importing this package never imports a compute engine or a vendor SDK.  The
engine is loaded lazily the first time a data-producing function is called
(e.g. ``connect``, ``load_flex_xml``, ``load_flex_file``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import pyarrow as pa

from ibda import errors, rates, schema

if TYPE_CHECKING:
    # Type-checking only: avoids an eager runtime import of the vendor-confined
    # supervisor module (which imports deephaven_ib) from the connect_live() wrapper
    # below. `from __future__ import annotations` keeps the annotation lazy at runtime.
    from ibda.adapters.ibkr.supervisor import IbkrSupervisor
from ibda._factory import connect
from ibda.analytics.benchmark import (
    RelativeSummary,
    relative_metrics,
    relative_summary,
    rolling_relative,
)
from ibda.analytics.performance import (
    PerformanceSummary,
    compute_performance,
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
from ibda.port import DataPort
from ibda.result import Result, Stream
from ibda.watch import watch


def load_flex_xml(xml: str) -> DataPort:
    """Parse a Flex XML string and return a connected :class:`DataPort`.

    Convenience wrapper — see
    :func:`ibda.adapters.ibkr.flex.loader.load_flex_xml` for full docs.
    The engine (Deephaven) is imported and started lazily on first call.

    Args:
        xml: raw Flex Activity XML string.

    Returns:
        A connected :class:`DataPort`.

    Raises:
        ibda.errors.FlexParseError: if the XML is malformed, not yet ready, or rejected.
    """
    from ibda.adapters.ibkr.flex.loader import (  # noqa: PLC0415 — lazy: keep engine out of import
        load_flex_xml as _load_flex_xml,
    )

    return cast(DataPort, _load_flex_xml(xml))


def load_flex_file(path: str) -> DataPort:
    """Read a Flex XML file path and return a connected, queryable :class:`DataPort`.

    **Boots the Deephaven engine** (starts an in-process JVM) on first call — real
    weight for a one-shot read. If you just want the canonical tables as plain
    :class:`pyarrow.Table` with no engine, use ``parse_statement_or_raise(xml)`` +
    :func:`flex_arrow_tables` instead (see the ``ibda/README.md`` "Recipe 1" pairing).
    Use `load_flex_file` when you specifically want `DataPort`'s query surface
    (``.filter``, ``.group_by``, ``.as_of_join``, live ``.subscribe``, ...).

    Convenience wrapper — see
    :func:`ibda.adapters.ibkr.flex.loader.load_flex_file` for full docs.

    Args:
        path: path to a Flex Activity XML file.

    Returns:
        A connected :class:`DataPort`.

    Raises:
        ibda.errors.FlexParseError: if the XML cannot be parsed or is not ready.
        OSError: if the file cannot be read.
    """
    from ibda.adapters.ibkr.flex.loader import (  # noqa: PLC0415 — lazy: keep engine out of import
        load_flex_file as _load_flex_file,
    )

    return cast(DataPort, _load_flex_file(path))


def flex_arrow_tables(sections: Mapping[str, Any]) -> dict[str, pa.Table]:
    """Map already-parsed Flex sections to canonical :class:`pyarrow.Table` objects — no engine.

    Takes **parsed sections** (a ``Mapping``), not a file path or raw XML string —
    call :func:`ibda.adapters.ibkr.flex.parse.parse_statement_or_raise` on the XML
    first to get *sections*, then pass its result here. This pairing
    (``parse_statement_or_raise`` -> ``flex_arrow_tables``) is the engine-free way to
    get canonical Arrow tables from a Flex report; see ``ibda/README.md`` "Recipe 1"
    for the full copy-pasteable example. If you have a path and want a live,
    queryable :class:`DataPort` instead (with the trade-off of booting the Deephaven
    engine), use :func:`load_flex_file` / :func:`load_flex_xml`.

    Convenience wrapper — see
    :func:`ibda.adapters.ibkr.flex.arrow.flex_arrow_tables` for full docs. That
    module is itself engine-free (pure parse -> map -> Arrow), but this
    top-level re-export still imports it lazily, matching :func:`load_flex_file`
    and :func:`load_flex_xml`, so importing ``ibda`` never has an import-time
    opinion about what the Flex adapter pulls in.

    Args:
        sections: parsed Flex sections, as returned by
            :func:`ibda.adapters.ibkr.flex.parse.parse_statement_or_raise`
            (or the lower-level :func:`~ibda.adapters.ibkr.flex.parse.parse_statement`).

    Returns:
        ``{"execution": ..., "cash": ...}`` always, plus ``"nav"`` when the
        report included an ``EquitySummaryByReportDateInBase`` section.
    """
    from ibda.adapters.ibkr.flex.arrow import (  # noqa: PLC0415 — lazy: keep flex parse out of import
        flex_arrow_tables as _flex_arrow_tables,
    )

    return _flex_arrow_tables(sections)


def connect_live(
    *,
    host: str = "127.0.0.1",
    port: int = 7497,
    client_id: int = 77,
    read_only: bool = True,
    settle_secs: int = 5,
    stable_secs: int = 10,
    max_settle_secs: int = 120,
    subscribe_symbols: list[str] | None = None,
    max_hz: float = 5.0,
    engine_port: int = 10000,
) -> tuple[IbkrSupervisor, DataPort]:
    """Connect to TWS, build canonical tables, and return a live DataPort.

    Convenience wrapper — see :func:`ibda.adapters.ibkr.live.connect_live` for the
    full step-by-step docs (start the Deephaven server, connect to TWS, wait for
    positions to settle, build canonical live views, optionally subscribe
    quotes/bars). The engine (Deephaven) and vendor SDK are imported and started
    lazily on first call, exactly like :func:`load_flex_file` — never at
    ``import ibda`` time.

    Args:
        host: TWS/Gateway host (default ``127.0.0.1``).
        port: TWS API port (default 7497 = paper TWS; use 7496 for live TWS).
        client_id: IBKR API client ID.
        read_only: True for analytics-only sessions.
        settle_secs: seconds to sleep after the TWS handshake before verifying
            connectivity.
        stable_secs: consecutive stable-count seconds for positions settlement.
        max_settle_secs: hard timeout on position settlement.
        subscribe_symbols: optional list of instrument symbols to subscribe to
            quotes and bars. If None, no market-data subscriptions are issued.
        max_hz: maximum market-data subscription rate (requests/s).
        engine_port: port for the Deephaven in-process server (default 10000).

    Returns:
        The live ``IbkrSupervisor`` (for health checks, account refresh, etc.)
        and the fully wired :class:`DataPort`.
    """
    from ibda.adapters.ibkr.live import (  # noqa: PLC0415 — lazy: keep engine out of import
        connect_live as _connect_live,
    )

    return _connect_live(
        host=host,
        port=port,
        client_id=client_id,
        read_only=read_only,
        settle_secs=settle_secs,
        stable_secs=stable_secs,
        max_settle_secs=max_settle_secs,
        subscribe_symbols=subscribe_symbols,
        max_hz=max_hz,
        engine_port=engine_port,
    )


def flex_performance(
    report: str,
    *,
    risk_free_annual: str | float = rates.DEFAULT_RISK_FREE_ANNUAL,
    periods_per_year: int = rates.DEFAULT_PERIODS_PER_YEAR,
    account: str | None = None,
) -> PerformanceSummary:
    """Compute Sharpe + basic performance/risk metrics from a Flex XML report.

    The simplest entry point — a Flex Activity XML file path (or raw XML string)
    in, a :class:`PerformanceSummary` out, with **no compute engine started**.
    Deposits/withdrawals are stripped automatically from the cash transactions.

    The Flex query must include the daily equity summary
    (``EquitySummaryByReportDateInBase``). See
    :func:`ibda.adapters.ibkr.flex.arrow.flex_performance` for full docs.

    Example::

        import ibda
        perf = ibda.flex_performance("activity.xml", risk_free_annual=0.05)
        print(perf.render())          # human-readable report
        print(perf.sharpe_ratio)      # or just the number

        # or let the documented offline Treasury-yield proxy resolve automatically:
        perf = ibda.flex_performance("activity.xml", risk_free_annual="auto")

    Args:
        report: path to a Flex Activity XML file, or the raw XML string.
        risk_free_annual: annual simple risk-free rate, or ``"auto"`` for the
            documented offline Treasury-yield proxy (:data:`ibda.rates.DEFAULT_RISK_FREE_ANNUAL`).
        periods_per_year: annualization factor (default 252 trading days).
        account: required only if the report covers more than one account.

    Returns:
        A :class:`PerformanceSummary`.

    Raises:
        ibda.errors.FlexParseError: if the XML is malformed, not yet ready, or rejected.
        OSError: if *report* is a path that cannot be read.
        ValueError: if the report has no daily NAV series, spans multiple accounts and
            none was chosen, or *risk_free_annual* is a non-numeric, non-``"auto"`` string.
    """
    from ibda.adapters.ibkr.flex.arrow import (  # noqa: PLC0415 — lazy: keep flex parse out of import
        flex_performance as _flex_performance,
    )

    return _flex_performance(
        report,
        risk_free_annual=risk_free_annual,
        periods_per_year=periods_per_year,
        account=account,
    )


__all__ = [
    # Connect / load
    "connect",
    "DataPort",
    "load_flex_file",
    "load_flex_xml",
    "flex_performance",
    "flex_arrow_tables",
    "connect_live",
    # Analytics — performance
    "PerformanceSummary",
    "compute_performance",
    "performance_summary",
    "sharpe_ratio",
    # Analytics — relative (benchmark)
    "RelativeSummary",
    "relative_metrics",
    "relative_summary",
    "rolling_relative",
    # Analytics — time-series
    "periodic_returns",
    "rolling_performance",
    # Analytics — round-trips / trade journal
    "Fill",
    "RoundTrip",
    "ReconstructResult",
    "JournalResult",
    "reconstruct_round_trips",
    "aggregate_round_trips",
    # Primitives
    "Result",
    "Stream",
    "watch",
    # Submodules
    "schema",
    "errors",
    "rates",
]
