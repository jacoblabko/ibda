"""ibda.adapters.ibkr.flex.loader — compose parse -> map -> build deephaven tables.

Entry point::

    from ibda.adapters.ibkr.flex.loader import flex_canonical_tables

    sections = parse_statement(xml_text)["sections"]
    tables = flex_canonical_tables(sections)
    # tables == {"execution": <dh Table>, "cash": <dh Table>}
    port = ibda.connect(tables)

The loader imports from both the pure mapping layer (adapters/ibkr) and the
engine builder (adapters/deephaven). It is NOT itself engine-confined, but it
does NOT directly import deephaven — it delegates to builders.table_from_rows,
which lives in the engine-confined adapter. The boundary guard allows this
because the deephaven import stays inside adapters/deephaven/.

Public convenience helpers::

    port = ibda.load_flex_file("reports/activity.xml")
    arrow = port.table("execution").snapshot()
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ibda.adapters.deephaven.builders import table_from_rows
from ibda.adapters.ibkr.flex.mapping import flex_sections_to_canonical
from ibda.adapters.ibkr.flex.parse import parse_statement_or_raise
from ibda.schema import CASH, EXECUTION, NAV


def flex_canonical_tables(sections: Mapping[str, Any]) -> dict[str, Any]:
    """Parse Flex sections into canonical deephaven Tables.

    Parameters
    ----------
    sections:
        The ``sections`` value from ``parse_statement``'s ``"ok"`` result.

    Returns
    -------
    ``{"execution": <Table>, "cash": <Table>, "nav": <Table>}`` — the ``"nav"``
    (daily NAV time series) table is included only when the Flex query carried an
    ``EquitySummaryByReportDateInBase`` section. It is omitted otherwise so that a
    report without a daily equity summary does not present an empty NAV table that
    would silently produce meaningless performance metrics.
    """
    canonical = flex_sections_to_canonical(sections)
    tables: dict[str, Any] = {
        "execution": table_from_rows(EXECUTION, canonical["execution"]),
        "cash": table_from_rows(CASH, canonical["cash"]),
    }
    nav_rows = canonical["nav"]
    if nav_rows:
        tables["nav"] = table_from_rows(NAV, nav_rows)
    return tables


def load_flex_xml(xml: str) -> Any:
    """Parse a Flex XML string and return a fully connected :class:`~ibda.port.DataPort`.

    This convenience function collapses the parse → canonical → connect chain
    into a single call.  The Deephaven engine is started (or reused) lazily
    when ``ibda.connect`` is first called.

    Parameters
    ----------
    xml:
        Raw Flex Web Service XML response string (the content of a
        ``GetStatement`` reply).

    Returns
    -------
    :class:`~ibda.port.DataPort`
        A live DataPort with ``"execution"`` and ``"cash"`` tables.

    Raises
    ------
    :class:`~ibda.errors.FlexParseError`
        If the XML is malformed (``status == "error"``), the report is not yet
        ready on the IBKR server (``status == "in_progress"``), or the request
        was rejected (``status == "fail"``).

    Example
    -------
    ::

        import pathlib

        import ibda
        port = ibda.load_flex_xml(pathlib.Path("activity.xml").read_text())
        arrow = port.table("execution").snapshot()
    """
    import ibda

    sections = parse_statement_or_raise(xml)
    tables = flex_canonical_tables(sections)
    return ibda.connect(tables)


def load_flex_file(path: str | os.PathLike[str]) -> Any:
    """Read a Flex XML file and return a connected :class:`~ibda.port.DataPort`.

    Takes a **file path** directly (reads *path* as UTF-8 text, delegates to
    :func:`load_flex_xml`) and **boots the Deephaven engine** (starts an
    in-process JVM) to back the returned `DataPort`'s query surface (`.filter`,
    `.group_by`, `.as_of_join`, live `.subscribe`, ...). If you only need the
    canonical tables as plain :class:`pyarrow.Table` with no engine, use
    :func:`ibda.adapters.ibkr.flex.parse.parse_statement_or_raise` +
    :func:`~ibda.adapters.ibkr.flex.arrow.flex_arrow_tables` instead — the
    engine-free alternative for the same Flex data.

    Parameters
    ----------
    path:
        Filesystem path to the Flex XML file.

    Returns
    -------
    :class:`~ibda.port.DataPort`
        A live DataPort with ``"execution"`` and ``"cash"`` tables.

    Raises
    ------
    :class:`~ibda.errors.FlexParseError`
        If the XML cannot be parsed or is not ready (forwarded from
        :func:`load_flex_xml`).
    :exc:`OSError`
        If the file cannot be read.

    Example
    -------
    ::

        import ibda
        port = ibda.load_flex_file("reports/activity.xml")
        arrow = port.table("execution").snapshot()
    """
    xml_text = Path(path).read_text(encoding="utf-8")
    return load_flex_xml(xml_text)
