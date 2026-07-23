"""Pure parsing of IBKR Flex Web Service XML. No network, no MCP imports.

Two entry points:
- parse_send_response(xml): the SendRequest reply (reference code / failure).
- parse_statement(xml): the GetStatement reply (full report, in-progress, or fail).

parse_statement returns one of:
  {"status": "ok", "sections": {...}, "unparsed_sections": [...]}
  {"status": "in_progress", "code": "1019", "message": ...}
  {"status": "fail", "code": ..., "message": ...}
  {"status": "error", "message": ...}            # malformed XML

Originally a standalone Flex parser, absorbed into this package when the Flex->canonical
adapter lane was built. Extended with exec_id, con_id, account, venue fields on each trade
dict so the canonical mapping (mapping.py) can produce non-null ExecId values.
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Any, cast

logger = logging.getLogger(__name__)

# Element tags we parse into sections; everything else is reported as unparsed.
_KNOWN_CONTAINERS = {
    "Trades", "CashTransactions", "CorporateActions", "Transfers",
    "ChangeInNAV", "FIFOPerformanceSummaryInBase", "EquitySummaryInBase",
}


def _to_float(v: str | None, field_name: str = "") -> float | None:
    """Convert a Flex attribute string to float, or None for absent/empty values.

    Logs a WARNING (including the field name and raw value) when *v* is a
    non-empty string that cannot be parsed as a float — distinguishing a genuine
    parse failure from a legitimately absent field (None / "").

    Parameters
    ----------
    v:
        The raw attribute string from the Flex XML element.
    field_name:
        The XML attribute name (e.g. ``"quantity"``), included in the warning
        so callers can identify which field triggered the coercion without
        searching the call stack.
    """
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        logger.warning(
            "_to_float: could not parse %r as float for field %r — returning None "
            "(downstream mapping will substitute 0.0 for None numeric fields)",
            v,
            field_name,
        )
        return None


def parse_send_response(xml: str) -> dict[str, Any]:
    """Parse a SendRequest reply."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        return {"status": "error", "message": f"malformed SendRequest XML: {exc}"}
    status = (root.findtext("Status") or "").strip()
    if status == "Success":
        return {
            "status": "Success",
            "reference_code": (root.findtext("ReferenceCode") or "").strip(),
            "url": (root.findtext("Url") or "").strip(),
        }
    return {
        "status": "Fail",
        "code": (root.findtext("ErrorCode") or "").strip(),
        "message": (root.findtext("ErrorMessage") or "").strip(),
    }


def parse_statement(xml: str) -> dict[str, Any]:
    """Parse a GetStatement reply (report / in-progress / fail)."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        return {"status": "error", "message": f"malformed statement XML: {exc}"}

    # A FlexStatementResponse here means in-progress or failure, not a report.
    if root.tag == "FlexStatementResponse":
        code = (root.findtext("ErrorCode") or "").strip()
        msg = (root.findtext("ErrorMessage") or "").strip()
        status = "in_progress" if code == "1019" else "fail"
        return {"status": status, "code": code, "message": msg}

    stmt = root.find(".//FlexStatement")
    if stmt is None:
        return {"status": "error", "message": "no FlexStatement element found"}

    sections = {
        "trades": _parse_trades(stmt),
        "cash": _parse_cash(stmt),
        "corporate_actions": _parse_corporate_actions(stmt),
        "pnl": _parse_pnl(stmt),
        "nav": _parse_nav(stmt),
    }
    unparsed = [
        child.tag for child in stmt
        if child.tag not in _KNOWN_CONTAINERS and len(child) > 0
    ]
    return {"status": "ok", "sections": sections, "unparsed_sections": unparsed}


def parse_statement_or_raise(xml: str) -> dict[str, Any]:
    """Parse a GetStatement reply and return its ``sections``, or raise.

    Convenience over :func:`parse_statement` that turns every non-``ok`` status
    into a :class:`~ibda.errors.FlexParseError` with an actionable message.
    Shared by the Deephaven loader and the engine-free Arrow path so both report
    identical errors.

    Raises
    ------
    :class:`~ibda.errors.FlexParseError`
        If the XML is malformed (``error``), not yet ready (``in_progress``), or
        the request was rejected (``fail``).
    """
    from ibda.errors import FlexParseError

    result = parse_statement(xml)
    status = result.get("status", "error")

    if status == "in_progress":
        code = result.get("code", "")
        message = result.get("message", "")
        raise FlexParseError(
            f"Flex report not yet ready (in_progress); code={code!r} message={message!r}. "
            "Retry after a few seconds."
        )
    if status != "ok":
        message = result.get("message", "")
        raise FlexParseError(f"Flex parse failed (status={status!r}): {message}")

    return cast("dict[str, Any]", result["sections"])


def _parse_trades(stmt: ET.Element) -> list[dict[str, Any]]:
    """Parse ``<Trades><Trade>`` elements, keeping only EXECUTION-level rows.

    Flex queries configured with multiple "levels of detail" (EXECUTION, ORDER,
    CLOSED_LOT, SUMMARY, ...) emit one ``<Trade>`` element PER level for the
    same underlying fill — including every level in the canonical mapping would
    silently double- (or triple-) count trades. The ``levelOfDetail`` attribute
    distinguishes them, but single-level Flex statements (the common case) omit
    the attribute entirely, so absence must be treated as "the only level
    present" rather than "unknown, drop it".

    The comparison is case-insensitive (``levelOfDetail`` is upper-cased before
    matching) — Flex does not document the attribute's casing as guaranteed.
    """
    trade_elements = stmt.findall(".//Trades/Trade")

    # Detect statements that mix more than one non-empty levelOfDetail value —
    # this is the case that would have silently double-counted trades before
    # this filter existed.
    distinct_levels = {
        level
        for t in trade_elements
        if (level := (t.attrib.get("levelOfDetail") or "").strip().upper())
    }
    if len(distinct_levels) > 1:
        logger.warning(
            "_parse_trades: statement mixes multiple levelOfDetail values %s — "
            "keeping only EXECUTION-level rows to avoid double-counting fills",
            sorted(distinct_levels),
        )

    out = []
    for t in trade_elements:
        a = t.attrib
        level = (a.get("levelOfDetail") or "").strip().upper()
        if level not in ("", "EXECUTION"):
            continue
        # exec_id: prefer ibExecID, fall back to tradeID, then empty string.
        exec_id = a.get("ibExecID") or a.get("tradeID") or ""
        out.append({
            "symbol": a.get("symbol"),
            "asset_category": a.get("assetCategory"),
            "trade_date": a.get("tradeDate"),
            "date_time": a.get("dateTime"),
            "buy_sell": a.get("buySell"),
            "quantity": _to_float(a.get("quantity"), "quantity"),
            "trade_price": _to_float(a.get("tradePrice"), "tradePrice"),
            "proceeds": _to_float(a.get("proceeds"), "proceeds"),
            "ib_commission": _to_float(a.get("ibCommission"), "ibCommission"),
            "net_cash": _to_float(a.get("netCash"), "netCash"),
            "fifo_pnl_realized": _to_float(a.get("fifoPnlRealized"), "fifoPnlRealized"),
            "currency": a.get("currency"),
            # Fields needed downstream by the canonical mapping (mapping.py):
            "exec_id": exec_id,
            "con_id": a.get("conid") or "",
            "account": a.get("accountId") or "",
            "venue": a.get("exchange") or "",
            "multiplier": a.get("multiplier"),
            "order_reference": a.get("orderReference"),
            "open_close": a.get("openCloseIndicator"),
            # ibOrderID is IB's native order id; orderID is the older/plain
            # attribute some Flex versions emit instead. Prefer ibOrderID.
            "order_id": a.get("ibOrderID") or a.get("orderID") or "",
        })

    # Distinct from the mixed-levels warning above: a SINGLE-level statement
    # whose only level is non-EXECUTION (e.g. an ORDER-only Flex query) has
    # len(distinct_levels) == 1, so the mixed-levels warning above never fires
    # — yet every Trade element gets filtered out, silently yielding an empty
    # execution table. Warn whenever levelOfDetail values were present at all
    # but none of them survived the EXECUTION-only filter.
    if distinct_levels and not out:
        logger.warning(
            "_parse_trades: statement has levelOfDetail values %s but none is "
            "'EXECUTION' — all %d trade element(s) dropped, yielding an empty "
            "execution table",
            sorted(distinct_levels),
            len(trade_elements),
        )

    return out


def _parse_cash(stmt: ET.Element) -> list[dict[str, Any]]:
    out = []
    for c in stmt.findall(".//CashTransactions/CashTransaction"):
        a = c.attrib
        out.append({
            "type": a.get("type"),
            "symbol": a.get("symbol") or None,
            "date_time": a.get("dateTime"),
            "amount": _to_float(a.get("amount"), "amount"),
            "currency": a.get("currency"),
            "description": a.get("description"),
            # account field needed by the canonical mapping (mapping.py)
            "account": a.get("accountId") or "",
        })
    return out


def _parse_corporate_actions(stmt: ET.Element) -> list[dict[str, Any]]:
    out = []
    for c in stmt.findall(".//CorporateActions/CorporateAction"):
        a = c.attrib
        out.append({
            "kind": "corporate_action",
            "symbol": a.get("symbol"),
            "type": a.get("type"),
            "date_time": a.get("dateTime"),
            "quantity": _to_float(a.get("quantity"), "quantity"),
            "description": a.get("description"),
        })
    for t in stmt.findall(".//Transfers/Transfer"):
        a = t.attrib
        # Priority: dateTime > date > reportDate > settleDate.
        # Treat empty string as missing so we don't pass "" to _parse_dt.
        date_time: str | None = (
            a.get("dateTime") or a.get("date") or a.get("reportDate") or a.get("settleDate") or None
        )
        out.append({
            "kind": "transfer",
            "symbol": a.get("symbol"),
            "type": a.get("type"),
            "date_time": date_time,
            "direction": a.get("direction"),
            "quantity": _to_float(a.get("quantity"), "quantity"),
            "description": a.get("description"),
            # Valuation fields for ACATS transfers
            "positionAmount": _to_float(a.get("positionAmount"), "positionAmount"),
            "positionAmountInBase": _to_float(a.get("positionAmountInBase"), "positionAmountInBase"),
            "transferPrice": _to_float(a.get("transferPrice"), "transferPrice"),
            "cashTransfer": _to_float(a.get("cashTransfer"), "cashTransfer"),
            "currency": a.get("currency"),
            "account": a.get("accountId"),
        })
    return out


def _parse_pnl(stmt: ET.Element) -> dict[str, Any]:
    nav: dict[str, Any] = {}
    nav_el = stmt.find(".//ChangeInNAV")
    if nav_el is not None:
        a = nav_el.attrib
        nav = {
            "from_date": a.get("fromDate"),
            "to_date": a.get("toDate"),
            "starting_value": _to_float(a.get("startingValue"), "startingValue"),
            "ending_value": _to_float(a.get("endingValue"), "endingValue"),
            "mtm": _to_float(a.get("mtm"), "mtm"),
            "realized": _to_float(a.get("realized"), "realized"),
            "dividends": _to_float(a.get("dividends"), "dividends"),
            "deposits_withdrawals": _to_float(a.get("depositsWithdrawals"), "depositsWithdrawals"),
        }
    fifo = []
    for u in stmt.findall(".//FIFOPerformanceSummaryUnderlying"):
        a = u.attrib
        fifo.append({
            "symbol": a.get("symbol"),
            "realized_total": _to_float(a.get("realizedTotal"), "realizedTotal"),
            "unrealized_total": _to_float(a.get("unrealizedTotal"), "unrealizedTotal"),
            "total_fifo_pnl": _to_float(a.get("totalFifoPnl"), "totalFifoPnl"),
        })
    return {"change_in_nav": nav, "fifo_by_symbol": fifo}


def _parse_nav(stmt: ET.Element) -> list[dict[str, Any]]:
    """Parse the daily NAV time series (EquitySummaryByReportDateInBase).

    Each row is one report date with the total account NAV in base currency and
    its cash / stock components. This series is the basis for historic return and
    risk-adjusted performance metrics (Sharpe, Sortino, drawdown). A query that
    does not request the equity summary simply yields an empty list — not an error.

    Flex reports the date in the ``reportDate`` attribute as either ``YYYY-MM-DD``
    or ``YYYYMMDD``; both are normalized to ``YYYY-MM-DD`` so the canonical mapping
    can treat them as date-only strings (ET midnight → UTC).
    """
    out = []
    for e in stmt.findall(".//EquitySummaryByReportDateInBase"):
        a = e.attrib
        out.append({
            "account": a.get("accountId") or "",
            "report_date": _norm_date(a.get("reportDate")),
            "total": _to_float(a.get("total"), "total"),
            "cash": _to_float(a.get("cash"), "cash"),
            "stock": _to_float(a.get("stock"), "stock"),
        })
    return out


def _norm_date(value: str | None) -> str | None:
    """Normalize a Flex date to ``YYYY-MM-DD``.

    Accepts ``YYYY-MM-DD`` (returned unchanged) or compact ``YYYYMMDD``
    (8 digits → dashed form). Anything else is returned as-is so the downstream
    date parser can decide how to handle it.
    """
    if not value:
        return None
    v = value.strip()
    if len(v) == 8 and v.isdigit():
        return f"{v[0:4]}-{v[4:6]}-{v[6:8]}"
    return v
