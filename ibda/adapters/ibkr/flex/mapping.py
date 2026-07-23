"""Pure mapping: IBKR Flex parsed sections -> canonical execution/cash rows.

No engine imports. No network. Input is the ``sections`` dict returned by
``parse_statement``; output is a plain Python dict ready for the deephaven
builder (builders.py) or for direct consumption.

Timezone assumption
-------------------
Flex ``dateTime`` values are account-local with no explicit tz offset.  IBKR
Flex reports default to the account's configured timezone, which for US
accounts is **America/New_York** (US/Eastern, DST-aware).  We localize each
naive string to America/New_York and then convert to UTC so all Timestamp
columns are comparable to the live side (which is always UTC).

If your IBKR account is configured to a different Flex timezone, the single
knob to revisit is the ``_FLEX_TZ`` constant below.

This assumption was fixed when the Flex->canonical mapping was first built and has
not needed revisiting since (no account observed with a non-Eastern Flex timezone).

Qty sign convention
-------------------
Canonical Qty is unsigned (always >= 0). The ``Side`` column ("BUY" or "SELL")
carries direction. Flex quantity is negative for sells; we take ``abs(quantity)``.

ExecId source
-------------
Preferred: ``ibExecID`` attribute (native IB execution id).
Fallback:  ``tradeID`` attribute.
Synthetic: ``"{symbol}-{dateTime}-{quantity}-{tradePrice}"`` when both are absent.
The synthetic id is deterministic given the same Flex report.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# The account-local timezone assumed for all Flex dateTime strings.
# IBKR US accounts default to America/New_York (US/Eastern, DST-aware).
# Override this constant if the account is configured to a different timezone.
_FLEX_TZ: ZoneInfo = ZoneInfo("America/New_York")


def _parse_dt(date_time: str | None) -> datetime:
    """Parse a Flex dateTime string to a UTC-aware datetime.

    Accepted formats (tried in order):

    * ``"YYYY-MM-DDTHH:MM:SS.ffffff±HH:MM"`` — ISO-8601 with sub-seconds + offset
    * ``"YYYY-MM-DDTHH:MM:SS±HH:MM"`` — ISO-8601 with timezone offset
    * ``"YYYY-MM-DDTHH:MM:SS.ffffff"`` — ISO-8601 sub-second, no offset (FLEX_TZ)
    * ``"YYYY-MM-DDTHH:MM:SS"`` — ISO-8601 no offset (FLEX_TZ)
    * ``"YYYY-MM-DD HH:MM:SS"`` — dashed date + time (legacy/dashed reports)
    * ``"YYYY-MM-DD"`` — dashed date only
    * ``"YYYYMMDD;HHMMSS"`` — compact IBKR default (semicolon separator)
    * ``"YYYYMMDD HHMMSS"`` — compact with space separator (occasional IBKR variant)
    * ``"YYYYMMDD"`` — compact date only

    Formats that include a timezone offset (``%z``) are converted directly to UTC.
    Formats without an offset are localized to ``_FLEX_TZ`` (America/New_York)
    then converted to UTC so all Timestamp columns are comparable.

    Empty / None input returns the UTC epoch without logging (field absent is expected).
    Non-empty unparseable input returns the UTC epoch AND logs a WARNING, including
    the raw value, so silent substitutions surface in structured logs.

    The America/New_York assumption matches IBKR's Flex default for US accounts.
    See ``_FLEX_TZ`` to change it.
    """
    if not date_time:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    dt_str = date_time.strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f%z",  # ISO with sub-seconds + offset e.g. "2026-06-02T10:31:00.123456-04:00"
        "%Y-%m-%dT%H:%M:%S%z",      # ISO with offset              e.g. "2026-06-02T10:31:00-04:00"
        "%Y-%m-%dT%H:%M:%S.%f",     # ISO sub-second, no offset    e.g. "2026-06-02T10:31:00.123456"
        "%Y-%m-%dT%H:%M:%S",        # ISO no offset                e.g. "2026-06-02T10:31:00"
        "%Y-%m-%d %H:%M:%S",        # dashed date + time           e.g. "2026-06-02 10:31:00"
        "%Y-%m-%d",                  # dashed date only             e.g. "2026-06-02"
        "%Y%m%d;%H%M%S",            # compact + semicolon          e.g. "20260602;094850"
        "%Y%m%d %H%M%S",            # compact + space              e.g. "20260602 094850"
        "%Y%m%d",                   # compact date only            e.g. "20260608"
    ):
        try:
            dt_obj = datetime.strptime(dt_str, fmt)
            if dt_obj.tzinfo is not None:
                # Format included an explicit UTC offset — convert directly.
                return dt_obj.astimezone(timezone.utc)
            # No offset in the format — localize to _FLEX_TZ then convert to UTC.
            return dt_obj.replace(tzinfo=_FLEX_TZ).astimezone(timezone.utc)
        except ValueError:
            continue
    # Last-resort: return epoch and emit a WARNING so the caller can filter bad rows.
    logger.warning(
        "_parse_dt: could not parse dateTime %r — substituting 1970 epoch", date_time
    )
    return datetime(1970, 1, 1, tzinfo=timezone.utc)


def _map_execution(trade: dict[str, Any]) -> dict[str, Any] | None:
    """Map one parsed trade dict to a canonical execution row dict.

    Returns ``None`` (and logs a WARNING) when ``Sym``, ``Price``, or ``Qty``
    cannot be determined — the EXECUTION schema declares ``Sym``/``Price``/
    ``Qty`` ``nullable=False``, so silently substituting ``""``/``0.0`` would
    fabricate a "ghost fill" (a zero-price or zero-quantity trade that never
    happened) rather than surface the gap. The caller
    (:func:`flex_sections_to_canonical`) filters ``None`` results, mirroring
    the ``_map_transfer`` pattern.
    """
    symbol: str = trade.get("symbol") or ""
    date_time: str = trade.get("date_time") or ""
    quantity: float | None = trade.get("quantity")
    trade_price: float | None = trade.get("trade_price")

    if not symbol or trade_price is None or quantity is None:
        logger.warning(
            "_map_execution: dropping trade row with missing Sym/Price/Qty "
            "(Sym=%r Price=%r Qty=%r ExecId=%r) — a 0.0/empty-string "
            "substitution would fabricate a ghost fill",
            symbol or None,
            trade_price,
            quantity,
            trade.get("exec_id"),
        )
        return None

    # ExecId: prefer ibExecID > tradeID > synthetic
    exec_id: str = trade.get("exec_id") or ""
    if not exec_id:
        exec_id = f"{symbol}-{date_time}-{quantity}-{trade_price}"

    # Qty: canonical is unsigned; Side carries direction
    qty_abs: float = abs(quantity)

    # ConId: integer or None
    con_id_raw: str = trade.get("con_id") or ""
    con_id: int | None = int(con_id_raw) if con_id_raw.strip().isdigit() else None

    # OrderId: IB order id (ibOrderID/orderID); integer or None.
    order_id_raw: str = trade.get("order_id") or ""
    order_id: int | None = int(order_id_raw) if order_id_raw.strip().isdigit() else None

    # Venue: empty string -> None
    venue_raw: str = trade.get("venue") or ""
    venue: str | None = venue_raw if venue_raw else None

    # Account: empty string -> None
    account_raw: str = trade.get("account") or ""
    account: str | None = account_raw if account_raw else None

    # SecType: pass through asset_category as-is; None when absent.
    sec_type: str | None = trade.get("asset_category") or None

    # Multiplier: parse the string value when present and valid; default 1.0.
    # Flex omits the multiplier attribute for STK and CASH/FX rows; treat
    # absence as multiplier=1.0 (no contract multiplier).
    multiplier_raw: str | None = trade.get("multiplier")
    if multiplier_raw is not None:
        try:
            multiplier: float = float(multiplier_raw)
        except (ValueError, TypeError):
            logger.warning(
                "_map_execution: could not parse multiplier %r for %s — substituting 1.0",
                multiplier_raw,
                symbol,
            )
            multiplier = 1.0
    else:
        multiplier = 1.0
        # For STK and CASH/FX rows, absent multiplier is expected — Flex omits
        # the attribute and 1.0 is the correct default.  For other asset
        # categories (OPT, FUT, FOP, …) a missing multiplier could silently
        # mis-price contract notionals, so surface it as a WARNING.
        if sec_type not in (None, "STK", "CASH"):
            logger.warning(
                "_map_execution: multiplier attribute absent for %s (SecType=%r) — "
                "defaulting to 1.0; verify the Flex query includes the multiplier field",
                symbol,
                sec_type,
            )

    return {
        "ExecId": exec_id,
        "Timestamp": _parse_dt(date_time),
        "Account": account,
        "ConId": con_id,
        "OrderId": order_id,
        "Sym": symbol,
        "SecType": sec_type,
        "Side": trade.get("buy_sell") or "",
        "Qty": qty_abs,
        "Price": float(trade_price),
        "Multiplier": multiplier,
        "Venue": venue,
        "Liquidity": None,       # not available in Flex
        "Commission": trade.get("ib_commission"),
        "RealizedPnl": trade.get("fifo_pnl_realized"),
        "OrderRef": trade.get("order_reference"),
        "Currency": trade.get("currency"),
        "OpenClose": trade.get("open_close"),
    }


def _map_cash(txn: dict[str, Any]) -> dict[str, Any] | None:
    """Map one parsed cash-transaction dict to a canonical cash row dict.

    Returns ``None`` when ``amount`` is absent — the CASH schema declares
    ``Amount`` ``nullable=False``, so substituting 0.0 would fabricate a
    genuine (but bogus) zero-value cash movement rather than surface the gap.
    The caller (:func:`flex_sections_to_canonical`) filters ``None`` results
    and logs a warning, mirroring the ``_map_transfer`` pattern.
    """
    amount: float | None = txn.get("amount")
    if amount is None:
        return None

    # Account: empty string -> None
    account_raw: str = txn.get("account") or ""
    account: str | None = account_raw if account_raw else None

    symbol: str | None = txn.get("symbol") or None

    return {
        "Account": account,
        "Timestamp": _parse_dt(txn.get("date_time")),
        "Type": txn.get("type") or "",
        "Sym": symbol,
        "Amount": float(amount),
        "Currency": txn.get("currency"),
    }


def _map_nav(row: dict[str, Any]) -> dict[str, Any] | None:
    """Map one parsed daily-equity-summary dict to a canonical nav row dict.

    The ``report_date`` is a date-only string (``YYYY-MM-DD``); we localize it to
    ET midnight and convert to UTC via ``_parse_dt``, matching every other Flex
    Timestamp in this adapter.

    Returns ``None`` when ``total`` is absent — the NAV schema declares
    ``Total`` ``nullable=False``, and substituting 0.0 would fabricate a
    -100%-then-+inf daily return rather than surface the gap. The caller
    (:func:`flex_sections_to_canonical`) filters ``None`` results and logs a
    warning, mirroring the ``_map_transfer`` pattern.
    """
    total: float | None = row.get("total")
    if total is None:
        return None

    account_raw: str = row.get("account") or ""
    account: str | None = account_raw if account_raw else None

    cash: float | None = row.get("cash")
    stock: float | None = row.get("stock")

    return {
        "Account": account,
        "Timestamp": _parse_dt(row.get("report_date")),
        "Total": float(total),
        "Cash": float(cash) if cash is not None else None,
        "Stock": float(stock) if stock is not None else None,
    }


def _map_transfer(t: dict[str, Any]) -> dict[str, Any] | None:
    """Map one parsed transfer dict (kind=="transfer") to a canonical cash row dict.

    The cash-equivalent magnitude is taken from the first available valuation source
    in priority order:

    1. ``positionAmountInBase`` — base-currency value (preferred)
    2. ``positionAmount`` — position value in local currency
    3. ``quantity * transferPrice`` — computed when both are present
    4. ``cashTransfer`` — explicit cash transfer amount

    The sign follows ``direction``:
    * ``"IN"`` (or absent/unknown) → +magnitude (asset received)
    * ``"OUT"`` → -magnitude (asset sent)

    Returns ``None`` when no valuation source is available; the caller should
    skip the row and emit a warning.
    """
    position_amount_in_base: float | None = t.get("positionAmountInBase")
    position_amount: float | None = t.get("positionAmount")
    transfer_price: float | None = t.get("transferPrice")
    quantity: float | None = t.get("quantity")
    cash_transfer: float | None = t.get("cashTransfer")

    # Determine magnitude from first available source.
    magnitude: float | None = None
    if position_amount_in_base is not None:
        magnitude = abs(position_amount_in_base)
    elif position_amount is not None:
        magnitude = abs(position_amount)
    elif quantity is not None and transfer_price is not None:
        magnitude = abs(quantity * transfer_price)
    elif cash_transfer is not None:
        magnitude = abs(cash_transfer)

    if magnitude is None:
        return None

    direction: str = (t.get("direction") or "IN").strip().upper()
    signed_amount: float = magnitude if direction != "OUT" else -magnitude

    account_raw: str | None = t.get("account")
    account: str | None = account_raw if account_raw else None

    return {
        "Account": account,
        "Timestamp": _parse_dt(t.get("date_time")),
        "Type": "Transfer",
        "Sym": t.get("symbol") or None,
        "Amount": signed_amount,
        "Currency": t.get("currency") or None,
    }


def flex_sections_to_canonical(
    sections: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Map Flex parsed sections to canonical execution, cash, and nav row lists.

    Parameters
    ----------
    sections:
        The ``sections`` value from ``parse_statement``'s ``"ok"`` result —
        a dict with at least ``"trades"``, ``"cash"``, and ``"nav"`` keys.

    Returns
    -------
    dict with keys ``"execution"``, ``"cash"``, and ``"nav"``, each holding a
    list of row dicts keyed by canonical column names. ``"nav"`` is empty when
    the Flex query did not include the daily equity summary section.
    """
    trades: list[dict[str, Any]] = sections.get("trades") or []
    cash_txns: list[dict[str, Any]] = sections.get("cash") or []
    nav_rows: list[dict[str, Any]] = sections.get("nav") or []
    corporate_actions: list[dict[str, Any]] = sections.get("corporate_actions") or []

    execution_rows: list[dict[str, Any]] = []
    for t in trades:
        mapped_exec = _map_execution(t)
        if mapped_exec is not None:
            execution_rows.append(mapped_exec)
        # _map_execution already logs a WARNING internally when it drops a row.

    cash_rows: list[dict[str, Any]] = []
    for c in cash_txns:
        mapped_cash = _map_cash(c)
        if mapped_cash is not None:
            cash_rows.append(mapped_cash)
        else:
            logger.warning(
                "Skipping cash row with missing Amount (Type=%r Sym=%r DateTime=%r)",
                c.get("type"),
                c.get("symbol"),
                c.get("date_time"),
            )

    nav_rows_out: list[dict[str, Any]] = []
    for n in nav_rows:
        mapped_nav = _map_nav(n)
        if mapped_nav is not None:
            nav_rows_out.append(mapped_nav)
        else:
            logger.warning(
                "Skipping NAV row with missing Total (Account=%r ReportDate=%r)",
                n.get("account"),
                n.get("report_date"),
            )

    # Map ACATS/in-kind transfers from the Transfers section (parsed into
    # corporate_actions with kind=="transfer") to canonical cash rows so they
    # can be stripped from NAV-based returns by external_flows_from_cash.
    for corp in corporate_actions:
        if corp.get("kind") != "transfer":
            continue
        result = _map_transfer(corp)
        if result is not None:
            cash_rows.append(result)
        else:
            sym = corp.get("symbol") or "<unknown>"
            dt = corp.get("date_time") or "<unknown date>"
            logger.warning(
                "Skipping unvalued ACATS transfer for %s on %s — "
                "no positionAmount/positionAmountInBase/transferPrice/cashTransfer present. "
                "This transfer will NOT be stripped from returns.",
                sym,
                dt,
            )

    return {
        "execution": execution_rows,
        "cash": cash_rows,
        "nav": nav_rows_out,
    }
