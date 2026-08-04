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

    Returns ``None`` for a missing, blank, or unparseable numeric field — this
    function never substitutes a default like 0.0. Logs a WARNING (including
    the field name and raw value) when *v* is a non-empty string that cannot
    be parsed as a float, distinguishing a genuine parse failure from a
    legitimately absent field (None / ""); a missing/blank field returns
    ``None`` silently. The caller (``mapping.py``) decides what a ``None``
    means per field: some numeric columns (e.g. Commission, RealizedPnl) pass
    it through as a nullable value, while others (e.g. Amount, Total, Price,
    Qty) treat it as reason to drop the row.

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
            "(mapping.py decides per-field whether None is kept as null or the row is dropped)",
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

    # A FlexQueryResponse can legitimately hold MORE THAN ONE <FlexStatement>
    # element — per-account, advisor, family, or non-consolidated Flex queries
    # report <FlexStatements count="N"> with N>1 sibling <FlexStatement>
    # blocks. root.find(...) here previously kept only the first block, so
    # every other statement's trades/cash/corporate-actions/pnl/nav rows were
    # silently dropped (no error, no warning) — which also defeated the
    # multi-account guard downstream in arrow.py (it derives the account set
    # from the nav rows it is handed, so it only ever saw statement #1's
    # account). Iterating and concatenating fixes this while staying
    # identical for the common count="1" case: findall returns a single
    # element there, so every _parse_* call runs exactly once, as before.
    stmts = root.findall(".//FlexStatement")
    if not stmts:
        return {"status": "error", "message": "no FlexStatement element found"}

    trades: list[dict[str, Any]] = []
    cash: list[dict[str, Any]] = []
    corporate_actions: list[dict[str, Any]] = []
    nav: list[dict[str, Any]] = []
    nav_changes: list[dict[str, Any]] = []
    fifo_by_symbol: list[dict[str, Any]] = []
    unparsed: list[str] = []
    for stmt in stmts:
        trades.extend(_parse_trades(stmt))
        cash.extend(_parse_cash(stmt))
        corporate_actions.extend(_parse_corporate_actions(stmt))
        nav.extend(_parse_nav(stmt))
        pnl = _parse_pnl(stmt)
        if pnl["change_in_nav"]:
            nav_changes.append(pnl["change_in_nav"])
        fifo_by_symbol.extend(pnl["fifo_by_symbol"])
        unparsed.extend(
            child.tag for child in stmt
            if child.tag not in _KNOWN_CONTAINERS and len(child) > 0
        )

    # change_in_nav is kept as a single dict (not a list) for backward
    # compatibility: it is returned verbatim to downstream consumers that
    # expect a dict shape and would break if this became a list. Each
    # ChangeInNAV record now also carries its own "account" (see the
    # nav_el.attrib block above), so the first-statement dict is no longer
    # anonymous. For the common single-statement case this is the exact
    # prior dict shape, just with an added "account" key.
    #
    # A genuinely multi-statement report has one ChangeInNAV record per
    # statement/account; merging them into one dict would silently overwrite
    # all but one. Rather than lose that data, change_in_nav_by_account below
    # carries the FULL list (every statement's record, each labeled with its
    # account) — the single dict stays first-statement-wins for callers that
    # haven't been updated, while the new list is lossless for callers that
    # need every account's NAV-change summary.
    pnl_section: dict[str, Any] = {
        "change_in_nav": nav_changes[0] if nav_changes else {},
        "change_in_nav_by_account": nav_changes,
        "fifo_by_symbol": fifo_by_symbol,
    }

    sections = {
        "trades": trades,
        "cash": cash,
        "corporate_actions": corporate_actions,
        "pnl": pnl_section,
        "nav": nav,
    }
    # De-dup unparsed tags across statements while preserving first-seen
    # order (a tag unparsed in statement 1 would otherwise also be reported
    # once per subsequent statement that has the same unhandled container).
    unparsed_sections = list(dict.fromkeys(unparsed))
    return {"status": "ok", "sections": sections, "unparsed_sections": unparsed_sections}


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
            # IBKR's local->base conversion rate for this row: base = amount * fxRateToBase.
            # Emitted only when the Flex query selects the field, so None is normal on a
            # single-currency query; the mapping keeps Amount/Currency as the LOCAL pair
            # and carries the rate alongside rather than converting here.
            "fxRateToBase": _to_float(a.get("fxRateToBase"), "fxRateToBase"),
            # IBKR's own identifier for the transaction, mirroring the exec_id line in
            # _parse_trades. Emitted only when the Flex query definition selects the
            # transactionID field, so "" is the common case and the canonical mapping
            # falls back to a deterministic content-derived id.
            "transaction_id": a.get("transactionID") or "",
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
            # IBKR's local->base conversion rate for this row: base = local * fxRateToBase.
            # Previously dropped entirely, which is why a base-currency magnitude could be
            # emitted under a local-currency label (see mapping._map_transfer).
            "fxRateToBase": _to_float(a.get("fxRateToBase"), "fxRateToBase"),
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
            "account": a.get("accountId") or "",
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
