"""ibda.adapters.ibkr.flex.arrow — engine-free Flex → Arrow + one-call metrics.

The lightweight counterpart to :mod:`ibda.adapters.ibkr.flex.loader`. Where
the loader builds live Deephaven tables (and starts the JVM), this module turns a
Flex report into plain :class:`pyarrow.Table` objects and computes performance
metrics with **no engine** — pure parse → map → Arrow → arithmetic.

That makes it the right entry point for the simple, shareable use case: a user
who just wants their Sharpe ratio from a downloaded Flex XML file, without
standing up the analytics platform.

    import ibda
    perf = ibda.flex_performance("activity.xml", risk_free_annual=0.05)
    print(perf.render())

ENGINE-FREE: imports only pyarrow + pure ibda modules. The boundary guard
permits this file because it never imports deephaven.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pyarrow as pa

from ibda.adapters.ibkr.flex.mapping import flex_sections_to_canonical
from ibda.adapters.ibkr.flex.parse import parse_statement_or_raise
from ibda.analytics.performance import (
    PerformanceSummary,
    compute_performance,
    external_flows_from_cash,
)
from ibda.rates import DEFAULT_PERIODS_PER_YEAR, DEFAULT_RISK_FREE_ANNUAL, resolve_risk_free
from ibda.schema import CASH, EXECUTION, NAV
from ibda.schema._core import DType, Schema


def arrow_table_from_rows(schema: Schema, rows: list[dict[str, Any]]) -> pa.Table:
    """Build a :class:`pyarrow.Table` from canonical row dicts, no engine.

    The pure-Arrow analogue of
    :func:`ibda.adapters.deephaven.builders.table_from_rows`. Columns and
    types come from *schema*; missing keys become nulls. An empty *rows* yields a
    correctly-typed zero-row table.

    A TIMESTAMP_NS column with a non-``datetime`` value is nulled rather than
    raising: ``pa.array(..., type=timestamp)`` otherwise raises
    :class:`pyarrow.lib.ArrowInvalid` on the first non-datetime value, whereas
    the parallel Deephaven builder
    (:func:`ibda.adapters.deephaven.builders.table_from_rows`) silently nulls
    it (``_to_j_instant(v) if isinstance(v, datetime) else None``). Not
    currently triggered by mapper output (mappers already drop bad rows before
    they reach here) — this is latent-parity hardening so the two producers
    behave identically if a caller ever feeds a raw, un-mapped row through.
    """
    columns: list[pa.Array] = []
    for col in schema.columns:
        values = [row.get(col.name) for row in rows]
        if col.dtype == DType.TIMESTAMP_NS:
            values = [v if isinstance(v, datetime) else None for v in values]
        columns.append(pa.array(values, type=col.dtype.to_arrow()))
    return cast(pa.Table, pa.table(dict(zip(schema.column_names, columns, strict=True)),
                                   schema=schema.to_arrow_schema()))


def flex_arrow_tables(sections: Mapping[str, Any]) -> dict[str, pa.Table]:
    """Map Flex sections to canonical :class:`pyarrow.Table` objects (no engine).

    Takes **parsed sections**, not a file path or raw XML — call
    :func:`ibda.adapters.ibkr.flex.parse.parse_statement_or_raise` on the XML first
    to get *sections*. That pairing (``parse_statement_or_raise`` then
    ``flex_arrow_tables``) is the whole engine-free recipe: no Deephaven, no JVM.
    The alternative that takes a path/XML string directly is
    :func:`ibda.adapters.ibkr.flex.loader.load_flex_file` /
    :func:`~ibda.adapters.ibkr.flex.loader.load_flex_xml` — but those boot the
    Deephaven engine to back a live, queryable :class:`~ibda.port.DataPort`.

    Returns ``{"execution": ..., "cash": ...}`` always, plus ``"nav"`` (the daily
    NAV series) when the report included an ``EquitySummaryByReportDateInBase``
    section.
    """
    canonical = flex_sections_to_canonical(sections)
    out: dict[str, pa.Table] = {
        "execution": arrow_table_from_rows(EXECUTION, canonical["execution"]),
        "cash": arrow_table_from_rows(CASH, canonical["cash"]),
    }
    if canonical["nav"]:
        out["nav"] = arrow_table_from_rows(NAV, canonical["nav"])
    return out


def _read_source(source: str | os.PathLike[str]) -> str:
    """Resolve *source* to Flex XML text — accepts a file path or raw XML string."""
    if not isinstance(source, str):
        return Path(source).read_text(encoding="utf-8")
    if source.lstrip()[:1] == "<":
        return source  # already XML
    return Path(source).read_text(encoding="utf-8")


def flex_performance(
    report: str | os.PathLike[str],
    *,
    risk_free_annual: str | float = DEFAULT_RISK_FREE_ANNUAL,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    account: str | None = None,
) -> PerformanceSummary:
    """Compute Sharpe and basic performance/risk metrics from a Flex report.

    The simplest possible path: a Flex Activity XML file (or XML string) in, a
    :class:`~ibda.analytics.performance.PerformanceSummary` out — no engine,
    no account setup. Deposits and withdrawals are stripped automatically using
    the report's cash transactions, so they are not mistaken for performance.
    Conceptually shorthand for ``performance_summary(load_flex_file(report))``, but
    engine-free: the Flex report is parsed straight to Arrow and handed to the same
    shared performance core (:func:`~ibda.analytics.performance.compute_performance`)
    rather than starting Deephaven.

    The Flex query **must include the daily equity summary**
    (``EquitySummaryByReportDateInBase``); that daily NAV series is what makes a
    Sharpe ratio computable.

    Args:
        report: path to a Flex Activity XML file, or the raw XML string.
        risk_free_annual: annual simple risk-free rate, or ``"auto"`` for the
            documented offline Treasury-yield proxy (:data:`ibda.rates.DEFAULT_RISK_FREE_ANNUAL`).
        periods_per_year: annualization factor (default 252 trading days).
        account: required only if the report covers more than one account — the
            account id to analyze.

    Returns:
        A :class:`~ibda.analytics.performance.PerformanceSummary`.

    Raises:
        ibda.errors.FlexParseError: if the XML is malformed, not ready, or rejected.
        OSError: if *report* is a path that cannot be read.
        ValueError: if the report has no daily NAV series, the named account is
            absent, it covers multiple accounts and none was specified, or
            *risk_free_annual* is a non-numeric, non-``"auto"`` string.
    """
    xml = _read_source(report)
    sections = parse_statement_or_raise(xml)
    return performance_from_sections(
        sections,
        risk_free_annual=risk_free_annual,
        periods_per_year=periods_per_year,
        account=account,
    )


def performance_from_sections(
    sections: Mapping[str, Any],
    *,
    risk_free_annual: str | float = DEFAULT_RISK_FREE_ANNUAL,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    account: str | None = None,
) -> PerformanceSummary:
    """Compute a PerformanceSummary from already-parsed Flex *sections* (no engine).

    Shared by :func:`flex_performance` (file/XML in) and a downstream performance
    analytic (sections fetched live via the Flex Web Service). Handles account selection and
    the missing-NAV error identically for both callers, then funnels into the same
    shared core as every other performance entry point
    (:func:`~ibda.analytics.performance.compute_performance`) — no re-derived math.

    Args:
        sections: parsed Flex sections (as returned by
            :func:`ibda.adapters.ibkr.flex.parse.parse_statement_or_raise`).
        risk_free_annual: annual simple risk-free rate, or ``"auto"`` for the
            documented offline Treasury-yield proxy (:data:`ibda.rates.DEFAULT_RISK_FREE_ANNUAL`).
        periods_per_year: annualization factor (default 252 trading days).
        account: required only if the report covers more than one account.

    Returns:
        A :class:`~ibda.analytics.performance.PerformanceSummary`.

    Raises:
        ValueError: if there is no daily NAV series, the named account is absent,
            the report covers multiple accounts and none was specified, or
            *risk_free_annual* is a non-numeric, non-``"auto"`` string.
    """
    rf, _rf_source = resolve_risk_free(risk_free_annual)
    canonical = flex_sections_to_canonical(sections)
    nav_rows = canonical["nav"]
    cash_rows = canonical["cash"]
    if not nav_rows:
        raise ValueError(
            "Flex report has no daily NAV series (EquitySummaryByReportDateInBase). "
            "Add that section to the Flex query — it is required to compute "
            "returns and a Sharpe ratio."
        )

    # Derived from NAV *and* cash, not NAV alone. The multi-account guard exists to stop one
    # account's flows being attributed to another, and cash is the side that carries the flows —
    # so deriving the account set from the NAV section only left the guard blind in exactly the
    # case it was written for. A report carrying an account in <CashTransactions> that is absent
    # from EquitySummaryByReportDateInBase would show len(present) == 1, skip the guard, and fold
    # that account's deposits into the other's flow series, silently depressing its return.
    #
    # Whether IBKR can actually emit such a report is still unobserved (every artifact under
    # artifacts/flex/ is single-account). That question no longer gates correctness: if it cannot,
    # this union equals the NAV set and nothing changes; if it can, the report is now rejected
    # with an actionable message instead of quietly producing a wrong number.
    present = sorted(
        {r["Account"] for r in nav_rows if r.get("Account")}
        | {r["Account"] for r in cash_rows if r.get("Account")}
    )
    if account is not None:
        nav_rows = [r for r in nav_rows if r.get("Account") == account]
        cash_rows = [r for r in cash_rows if r.get("Account") == account]
        if not nav_rows:
            raise ValueError(
                f"no NAV rows for account {account!r}; accounts in report: {present}"
            )
    elif len(present) > 1:
        raise ValueError(
            f"Flex report covers multiple accounts {present}; "
            "pass account=... to choose one."
        )

    nav_table = arrow_table_from_rows(NAV, nav_rows)
    flows = external_flows_from_cash(arrow_table_from_rows(CASH, cash_rows))
    return compute_performance(
        nav_table,
        flows=flows,
        risk_free_annual=rf,
        periods_per_year=periods_per_year,
    )
