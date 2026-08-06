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
Synthetic: a ``synx-`` prefixed content hash of Account/Sym/DateTime/Qty/Price when
both are absent. Deterministic, so the same fill gets the same id from any report
containing it — which is what lets :func:`_dedupe_execution_rows` collapse the
overlap between two ``<FlexStatement>`` blocks. It is a fingerprint, not an
identity; see :func:`_synthetic_exec_id`.

TxnId source (cash rows)
------------------------
Preferred: ``transactionID`` attribute, emitted only when the Flex query
definition selects that field.
Synthetic: a ``syn-`` prefixed content hash of the canonical row, used for every
transfer row and for any cash row IBKR did not identify. It is deterministic, so
the same movement gets the same id from any report containing it — but it is a
fingerprint, not an identity. See :func:`_synthetic_cash_id`.
"""
from __future__ import annotations

import hashlib
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

# Marks a cash TxnId that this adapter derived from row content rather than one
# IBKR supplied. Keeping the two visibly distinct matters: an IBKR transactionID
# is an identity, a derived id is only a content fingerprint.
_SYNTHETIC_CASH_ID_PREFIX: str = "syn-"

#: Same idea for executions. A ``<Trade>`` normally carries ``ibExecID`` (a real IBKR
#: execution identity), but a Flex query that does not select it leaves this adapter
#: to fabricate one — and a fabricated id must be visibly distinct from a real one,
#: because :func:`_dedupe_execution_rows` reports the two confidences separately.
_SYNTHETIC_EXEC_ID_PREFIX: str = "synx-"


def _synthetic_exec_id(
    *,
    account: str | None,
    symbol: str | None,
    date_time: str | None,
    quantity: float,
    trade_price: float,
) -> str:
    """Content fingerprint for a ``<Trade>`` with no ``ibExecID`` / ``tradeID``.

    Mirrors :func:`_synthetic_cash_id`: hash the fields that identify the fill into a
    stable, prefixed string. ``account`` is included — its omission from the previous
    string form is what made two accounts' identical fills collide — and the digest
    form means the id can never end in ``.NN``, which ``reconcile.normalize_exec_id``
    would strip.

    Same guarantee and same limit as the cash equivalent: the same fill mapped from
    any report yields the same id (which is what makes an overlapping re-ingest
    de-duplicable), but two genuinely separate fills agreeing on every field hash to
    one id and cannot be told apart. Select ``ibExecID`` in the Flex query to get a
    real identity.
    """
    parts: list[str] = [
        str(account or ""),
        str(symbol or ""),
        str(date_time or ""),
        repr(float(quantity)),
        repr(float(trade_price)),
    ]
    digest = hashlib.sha1(
        "|".join(parts).encode("utf-8"), usedforsecurity=False
    ).hexdigest()
    return f"{_SYNTHETIC_EXEC_ID_PREFIX}{digest[:16]}"


def _parse_dt_or_none(date_time: str | None) -> datetime | None:
    """Parse a Flex dateTime string to a UTC-aware datetime, or ``None``.

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

    Returns ``None`` — silently, with no substitute value — for input that is
    absent, blank, or in none of the accepted formats. This is the form callers
    use when an unusable timestamp means the row must be dropped rather than
    anchored somewhere arbitrary; :func:`_parse_dt` is the substituting form.

    The America/New_York assumption matches IBKR's Flex default for US accounts.
    See ``_FLEX_TZ`` to change it.
    """
    if not date_time:
        return None
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
        except ValueError:
            continue
        if dt_obj.tzinfo is not None:
            # Format included an explicit UTC offset — convert directly.
            return dt_obj.astimezone(timezone.utc)
        # No offset in the format — localize to _FLEX_TZ then convert to UTC.
        return dt_obj.replace(tzinfo=_FLEX_TZ).astimezone(timezone.utc)
    return None


def _parse_dt(date_time: str | None) -> datetime:
    """Parse a Flex dateTime string to a UTC-aware datetime, substituting the epoch.

    The substituting form of :func:`_parse_dt_or_none` — same accepted formats and
    same timezone handling, but an unusable value yields the UTC epoch instead of
    ``None`` so the caller always receives a datetime.

    Empty / None input returns the UTC epoch without logging (field absent is expected).
    Non-empty unparseable input returns the UTC epoch AND logs a WARNING, including
    the raw value, so silent substitutions surface in structured logs.

    Substituting the epoch attributes the row to 1970-01-01, a day no NAV series
    covers. Callers for whom that is a fabricated value rather than a harmless
    placeholder — anything whose Timestamp decides which period a money movement
    belongs to — should call :func:`_parse_dt_or_none` and drop the row instead.
    """
    parsed = _parse_dt_or_none(date_time)
    if parsed is not None:
        return parsed
    if date_time:
        # Distinguish a genuine parse failure from a legitimately absent field:
        # only the former is worth a warning.
        logger.warning(
            "_parse_dt: could not parse dateTime %r — substituting 1970 epoch", date_time
        )
    return datetime(1970, 1, 1, tzinfo=timezone.utc)


def _synthetic_cash_id(row: Mapping[str, Any]) -> str:
    """Derive a deterministic id for a canonical cash row from its own content.

    IBKR does not always give a cash movement an identifier: ``transactionID`` is
    emitted only when the Flex query definition selects that field, and nothing
    else on a ``<CashTransaction>`` or ``<Transfer>`` element is guaranteed unique.
    So when there is no real id this hashes the six canonical fields that describe
    the movement — ``Account``, ``Timestamp``, ``Type``, ``Sym``, ``Amount``,
    ``Currency`` — into a stable string, prefixed :data:`_SYNTHETIC_CASH_ID_PREFIX`.

    What this does guarantee: the same movement, mapped from any report that
    contains it, yields the same id. That is what lets an overlapping re-ingest be
    de-duplicated instead of double-counted, and it is reproducible from the
    canonical row alone — no parser state is folded in.

    What it does NOT guarantee: uniqueness. Two genuinely separate movements that
    agree on all six fields (say the same fee charged twice on a date-only stamp)
    hash to one id and cannot be told apart. A content fingerprint is the strongest
    identity the data supports when IBKR supplies none; select ``transactionID`` in
    the Flex query to get a real one.
    """
    ts: Any = row.get("Timestamp")
    amount: Any = row.get("Amount")
    parts: list[str] = [
        str(row.get("Account") or ""),
        ts.isoformat() if isinstance(ts, datetime) else "",
        str(row.get("Type") or ""),
        str(row.get("Sym") or ""),
        # repr() of a float round-trips exactly and is stable across runs, so the
        # same amount never hashes two ways.
        repr(float(amount)) if amount is not None else "",
        str(row.get("Currency") or ""),
    ]
    digest = hashlib.sha1(
        "|".join(parts).encode("utf-8"), usedforsecurity=False
    ).hexdigest()
    return f"{_SYNTHETIC_CASH_ID_PREFIX}{digest[:16]}"


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
    # `parse.py` extracts BOTH dateTime and tradeDate, and this mapper used only the first.
    # A Flex query whose Trades section selects Trade Date but not Date/Time therefore
    # produced an empty `date_time`, and `_parse_dt` epoch-anchors an empty value WITHOUT a
    # warning (it only warns when the input is truthy) -- so every fill silently landed on
    # 1970-01-01. That is worse than a lost timestamp: `date_time` is also a hash input to
    # the synthetic ExecId below, so with it empty the fingerprint lost all time resolution
    # and two genuine same-day fills in the same name at the same price collapsed to one id
    # -- the second was then DELETED by _dedupe_execution_rows. Shares and commission gone,
    # and round-trip reconstruction (sorted by Timestamp) saw the survivors as a flip
    # through zero, fabricating a short round-trip that never happened.
    #
    # Falling back to tradeDate restores day resolution for both the Timestamp and the
    # fingerprint. Rows that carry dateTime are byte-unchanged, so existing synthetic ids
    # are stable. Every sibling mapper (_map_cash, _map_nav, _map_transfer) was already
    # migrated off the substituting parser for this reason; this was the one left behind,
    # and an execution's Timestamp decides round-trip ordering, hold time, markout window
    # and TCA scope.
    date_time: str = trade.get("date_time") or ""
    if not date_time:
        trade_date_only: str = trade.get("trade_date") or ""
        if trade_date_only:
            logger.warning(
                "_map_execution: Trades row for %r has no dateTime; falling back to "
                "tradeDate %r (day resolution only). Add Date/Time to the Flex query's "
                "Trades section for intraday ordering, markouts and hold times.",
                symbol,
                trade_date_only,
            )
            date_time = trade_date_only

    # ...and when NEITHER is present, drop the row rather than anchor it. The fallback
    # above only fires when `tradeDate` exists; with both absent this still reached the
    # substituting `_parse_dt` and produced exactly the 1970 fingerprint collapse the
    # comment describes, silently -- `_parse_dt` warns only on truthy input, and the
    # tradeDate warning needs a tradeDate. `_parse_dt`'s own docstring states the rule
    # this now follows: a caller whose Timestamp decides which period a row belongs to
    # must use the non-substituting form and drop instead. That is the whole sibling set
    # (_map_cash, _map_nav, _map_transfer); this mapper is now genuinely among them.
    timestamp: datetime | None = _parse_dt_or_none(date_time)
    if timestamp is None:
        logger.warning(
            "_map_execution: dropping Trades row for %r with an unusable Timestamp "
            "(dateTime=%r tradeDate=%r). Anchoring it to the epoch would collapse the "
            "synthetic ExecId's time resolution and let _dedupe_execution_rows delete "
            "genuine distinct fills.",
            symbol,
            trade.get("date_time"),
            trade.get("trade_date"),
        )
        return None

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

    buy_sell: str = trade.get("buy_sell") or ""
    if not buy_sell:
        logger.warning(
            "_map_execution: buySell absent/blank for %s (ExecId=%r) — emitting "
            "Side='' rather than fabricating BUY/SELL; downstream consumers that "
            "assume BUY/SELL should treat an empty Side as unknown direction",
            symbol,
            trade.get("exec_id"),
        )

    # ExecId: prefer ibExecID > tradeID > synthetic.
    #
    # The synthetic form is a `syn-` prefixed content hash, matching
    # _synthetic_cash_id, and it is what makes _dedupe_execution_rows safe. The
    # previous form -- f"{symbol}-{date_time}-{quantity}-{trade_price}" -- was wrong
    # in two ways that only matter once executions are de-duplicated:
    #
    #   * it omitted `account`, so the SAME fill in two accounts produced byte-
    #     identical ids and a dedupe pass would silently drop one account's fill;
    #   * it ended in a price, and `reconcile.normalize_exec_id` strips a trailing
    #     `\.\d{1,3}$`, so "AAPL-...-100-200.25" and "AAPL-...-100-200" normalise
    #     together.
    #
    # It also carried no marker, so a consumer could not tell a fabricated id from a
    # real ibExecID -- which is exactly the distinction the cash side uses to report
    # its two dedupe confidences separately.
    exec_id: str = trade.get("exec_id") or ""
    if not exec_id:
        exec_id = _synthetic_exec_id(
            account=trade.get("account"),
            symbol=symbol,
            date_time=date_time,
            quantity=quantity,
            trade_price=trade_price,
        )

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
        "Timestamp": timestamp,
        "Account": account,
        "ConId": con_id,
        "OrderId": order_id,
        "Sym": symbol,
        "SecType": sec_type,
        "Side": buy_sell,
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

    Returns ``None`` (and logs a WARNING) in two cases, both of which would
    otherwise put a fabricated value into a money column:

    * ``amount`` is absent — the CASH schema declares ``Amount``
      ``nullable=False``, so substituting 0.0 would fabricate a genuine (but
      bogus) zero-value cash movement rather than surface the gap.
    * ``date_time`` is absent or unparseable — anchoring the row to the 1970
      epoch invents a cash movement on a day no NAV series covers, so the
      movement is silently dropped from flow-adjusted returns anyway. Dropping
      the row makes that loss countable and logged instead of invisible.

    A date that merely looks unusual is not affected: any value the accepted
    Flex formats can parse is kept, including a legitimate 1970-01-01 stamp.

    The caller (:func:`flex_sections_to_canonical`) filters ``None`` results and
    counts the two causes separately, mirroring the ``_map_transfer`` pattern.
    """
    # Account: empty string -> None
    account_raw: str = txn.get("account") or ""
    account: str | None = account_raw if account_raw else None

    amount: float | None = txn.get("amount")
    if amount is None:
        logger.warning(
            "_map_cash: dropping cash row with no Amount "
            "(Account=%r Type=%r Sym=%r DateTime=%r) — a 0.0 substitution would "
            "fabricate a zero-value cash movement",
            account,
            txn.get("type"),
            txn.get("symbol"),
            txn.get("date_time"),
        )
        return None

    timestamp: datetime | None = _parse_dt_or_none(txn.get("date_time"))
    if timestamp is None:
        logger.warning(
            "_map_cash: dropping cash row with an unusable Timestamp "
            "(Account=%r DateTime=%r Type=%r Amount=%r) — anchoring it to the 1970 "
            "epoch would attribute a real cash movement to a day no NAV series "
            "covers, losing it from flow-adjusted returns without a trace",
            account,
            txn.get("date_time"),
            txn.get("type"),
            amount,
        )
        return None

    symbol: str | None = txn.get("symbol") or None

    row: dict[str, Any] = {
        "TxnId": "",  # filled in below, once the identifying fields are settled
        "Account": account,
        "Timestamp": timestamp,
        "Type": txn.get("type") or "",
        "Sym": symbol,
        # Amount and Currency stay the LOCAL pair. Converting to base here would
        # emit a base-currency magnitude under a local-currency label, which is
        # the internally-inconsistent row shape this column exists to avoid;
        # FxRateToBase carries the conversion instead, for consumers that need it.
        "Amount": float(amount),
        "Currency": txn.get("currency"),
        "FxRateToBase": txn.get("fxRateToBase"),
    }
    row["TxnId"] = txn.get("transaction_id") or _synthetic_cash_id(row)
    return row


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

    Returns ``None`` **also** when ``report_date`` is unusable, for the same reason
    and by the same rule :func:`_parse_dt` states in its own docstring: a caller
    "whose Timestamp decides which period a money movement belongs to should call
    :func:`_parse_dt_or_none` and drop the row instead". NAV is not merely such a
    caller — it is the *denominator* of every return this adapter produces, so a
    fabricated epoch row is the worst possible instance of that rule being broken.
    ``_norm_date`` passes a non-``YYYYMMDD`` value through unchanged and an absent
    ``reportDate`` does not even warn, so the substitution was silent: the row sorts
    first in ``_nav_series``, becomes ``starting_nav``, and the 56-year gap counts as
    one ordinary period. Measured on one otherwise-clean series, a single epoch row
    moved annualised return from +87.47% to +52.04% with no error and no warning.
    """
    total: float | None = row.get("total")
    if total is None:
        return None

    timestamp: datetime | None = _parse_dt_or_none(row.get("report_date"))
    if timestamp is None:
        logger.warning(
            "_map_nav: dropping NAV row with an unusable Timestamp "
            "(Account=%r ReportDate=%r Total=%r) — anchoring the DENOMINATOR of "
            "every return to the 1970 epoch silently rewrites the whole performance "
            "series rather than losing one row",
            row.get("account"),
            row.get("report_date"),
            total,
        )
        return None

    account_raw: str = row.get("account") or ""
    account: str | None = account_raw if account_raw else None

    cash: float | None = row.get("cash")
    stock: float | None = row.get("stock")

    return {
        "Account": account,
        "Timestamp": timestamp,
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

    **Currency consistency.** The emitted row carries ``Amount`` and ``Currency`` together,
    so they must describe the same thing. Source 1 is in the *base* currency while
    ``Currency`` is the row's *local* currency — on a multi-currency statement that
    combination is simply wrong (a 1,000 EUR transfer emitted as ``Amount=1080,
    Currency="EUR"`` when base is USD). Where IBKR supplies ``fxRateToBase`` we convert the
    base magnitude back to local (``local = base / fxRateToBase``) so the pair agrees.

    Where it is absent we cannot tell "single-currency account, rate is 1" from "missing
    rate on a cross-currency row" — but we do not have to, when ``positionAmount`` is
    present: that IS the local magnitude, and local is what ``Currency`` labels. So with no
    usable rate the order is ``positionAmount`` first, base only as a last resort (with the
    warning). That inverts the base-first priority stated above, deliberately and only in
    the no-rate case: preferring base there means preferring a number in the wrong unit
    over one in the right unit.

    This branch previously fell through to the base value and suppressed the warning
    *because* ``positionAmount`` was present — so the one case where the mismatch could be
    corroborated was the one case that went unreported. On a single-currency book
    base == local and nothing changes either way.

    The rate itself is carried on the emitted row as ``FxRateToBase`` (``None`` when
    IBKR did not supply one), so a consumer that must aggregate in base currency can
    convert rather than re-deriving the rate or assuming it is 1.

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
    fx_rate_to_base: float | None = t.get("fxRateToBase")

    # Determine magnitude from first available source.
    magnitude: float | None = None
    if position_amount_in_base is not None:
        # Source is base-currency; the row is labelled with the LOCAL currency. Convert so
        # Amount and Currency agree. base = local * fxRateToBase, hence local = base / rate.
        if fx_rate_to_base is not None and fx_rate_to_base > 0.0:
            magnitude = abs(position_amount_in_base) / fx_rate_to_base
        elif position_amount is not None:
            # No usable rate, but the LOCAL magnitude is right here — and local is exactly
            # what `Currency` labels, so there is nothing to infer.
            #
            # This branch used to fall through to the base value, and suppressed the warning
            # below *precisely because* `positionAmount` was present: the one case where the
            # mismatch could be corroborated was the one case that went unreported. Verified:
            # positionAmountInBase=1080, positionAmount=1000, currency='EUR' emitted
            # Amount=1080 Currency='EUR' silently. Transfers feed `external_flows_from_cash`,
            # so an overstated flow is stripped from NAV-based returns and the settlement
            # period's return is wrong by the FX spread.
            #
            # This inverts the documented source priority (base first), deliberately. That
            # priority is right when a rate lets us convert; with no rate, preferring base
            # means preferring a number in the wrong unit over one in the right unit. On a
            # single-currency book the two are equal and nothing changes.
            magnitude = abs(position_amount)
        else:
            magnitude = abs(position_amount_in_base)
            logger.warning(
                "_map_transfer: using positionAmountInBase (base currency) for %s but "
                "no fxRateToBase and no positionAmount are present — emitting it under "
                "the local currency label %r. Correct only if this account's base "
                "currency IS %s.",
                t.get("symbol") or "<unknown>",
                t.get("currency"),
                t.get("currency"),
            )
    elif position_amount is not None:
        magnitude = abs(position_amount)
    elif quantity is not None and transfer_price is not None:
        magnitude = abs(quantity * transfer_price)
    elif cash_transfer is not None:
        magnitude = abs(cash_transfer)

    if magnitude is None:
        return None

    direction_raw: str = t.get("direction") or ""
    if not direction_raw.strip():
        logger.warning(
            "_map_transfer: direction absent/blank for %s — defaulting to IN "
            "(+magnitude); a genuine OUT transfer missing this attribute would "
            "silently be treated as an inflow",
            t.get("symbol") or "<unknown>",
        )
    direction: str = (direction_raw or "IN").strip().upper()
    signed_amount: float = magnitude if direction != "OUT" else -magnitude

    account_raw: str | None = t.get("account")
    account: str | None = account_raw if account_raw else None

    # Same rule as _map_cash, and it matters for the same reason: a transfer lands in
    # the SAME canonical cash table, read by the SAME external_flows_from_cash. A
    # transfer anchored to 1970 matches no NAV point, so it is never stripped from
    # returns and the settlement day books the whole in-kind value as a trading gain.
    # parse.py already anticipates all four date attributes being absent, so this is
    # reachable, and _map_cash three functions up drops-and-logs on exactly this input.
    timestamp: datetime | None = _parse_dt_or_none(t.get("date_time"))
    if timestamp is None:
        logger.warning(
            "_map_transfer: dropping transfer with an unusable Timestamp "
            "(Account=%r Sym=%r DateTime=%r Amount=%r) — a 1970-anchored transfer is "
            "never stripped from returns, so the settlement day would report the "
            "in-kind value as a trading gain",
            account,
            t.get("symbol"),
            t.get("date_time"),
            signed_amount,
        )
        return None

    row: dict[str, Any] = {
        "TxnId": "",  # filled in below, once the identifying fields are settled
        "Account": account,
        "Timestamp": timestamp,
        "Type": "Transfer",
        "Sym": t.get("symbol") or None,
        "Amount": signed_amount,
        "Currency": t.get("currency") or None,
        # The rate that converts this row's (local) Amount back to base. Every
        # valuation source above yields a local magnitude — source 1 is divided by
        # the rate precisely so it does — so the same rate applies to all of them.
        "FxRateToBase": fx_rate_to_base,
    }
    # A <Transfer> element carries no transactionID, so this is always synthetic —
    # but it must still be populated: a cash table where transfers share a null id
    # would collapse every transfer into one under a de-duplication pass.
    row["TxnId"] = _synthetic_cash_id(row)
    return row


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
    the Flex query did not include the daily equity summary section. Cash rows
    (including transfers) are de-duplicated by ``TxnId`` — see
    :func:`_dedupe_cash_rows` for what that does and does not guarantee.
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
    dropped_missing_amount = 0
    dropped_bad_timestamp = 0
    for c in cash_txns:
        mapped_cash = _map_cash(c)
        if mapped_cash is not None:
            cash_rows.append(mapped_cash)
            continue
        # _map_cash logs the per-row detail; classify here so the two causes stay
        # separable in aggregate. A missing Amount is checked first there, so it
        # is the reason whenever the amount is absent.
        if c.get("amount") is None:
            dropped_missing_amount += 1
        else:
            dropped_bad_timestamp += 1
    if dropped_missing_amount or dropped_bad_timestamp:
        logger.warning(
            "Skipped %d cash row(s): %d with a missing Amount, %d with an unusable "
            "Timestamp — these movements are absent from the canonical cash table "
            "and therefore from flow-adjusted returns",
            dropped_missing_amount + dropped_bad_timestamp,
            dropped_missing_amount,
            dropped_bad_timestamp,
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
        "execution": _dedupe_execution_rows(execution_rows),
        "cash": _dedupe_cash_rows(cash_rows),
        "nav": _dedupe_nav_rows(nav_rows_out),
    }


def _dedupe_nav_rows(nav_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop repeat NAV rows by ``(Account, Timestamp)``, keeping the first occurrence.

    NAV was the only one of the three canonical Flex tables left un-de-duplicated, and it is
    the one whose duplicates do the most damage. ``parse.py`` deliberately concatenates every
    ``<FlexStatement>`` block in a response, so a report holding two blocks with overlapping
    date ranges yields the same report date twice — the exact scenario
    :func:`_dedupe_execution_rows` already names as reachable and reproduced for fills.

    Nothing downstream compensated. ``analytics.performance._nav_series`` sorts but preserves
    duplicates, so ``_dated_returns`` emitted an extra zero-length period per duplicated date
    **and re-subtracted that date's external flow a second time** — a deposit stripped twice.
    Measured on a two-statement report overlapping two days with one $5,000 deposit:
    cumulative return −2.10% → −6.95%, max drawdown −4.00% → −8.75%, Sharpe −3.81 → −7.83.
    Silent, in the direction of a worse-looking track record, with no error anywhere.

    Two further reasons this is a defect and not a tolerated shape: ``ibda/schema/nav.py``
    declares the contract "One row per report date", and ``analytics.benchmark._returns_by_date``
    treats the identical input as *fatal* ("duplicate return date … ambiguous input"). So the
    same table raised in ``relative_summary`` while silently mis-reporting in
    ``performance_summary`` — the two disagreed about whether it was even valid.

    Keying on ``(Account, Timestamp)`` rather than ``Timestamp`` alone keeps a genuine
    multi-account consolidated report intact; ``performance_from_sections`` guards that case
    separately.
    """
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    repeats = 0
    for row in nav_rows:
        key = (row.get("Account"), row.get("Timestamp"))
        if key in seen:
            repeats += 1
            continue
        seen.add(key)
        deduped.append(row)

    if repeats:
        logger.warning(
            "Dropped %d duplicate NAV row(s) sharing an (Account, report date) — most often "
            "two overlapping <FlexStatement> blocks in one response. Keeping one avoids both "
            "a fabricated zero-length return period and double-subtracting that date's "
            "external flow.",
            repeats,
        )
    return deduped


def _dedupe_execution_rows(execution_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop repeat execution rows by ``ExecId``, keeping the first occurrence.

    The cash side has been de-duplicated since overlapping ``<FlexStatement>`` blocks
    became reachable; executions were not, though the argument is identical and the
    consequence is larger. ``parse_statement`` deliberately concatenates **all**
    statement blocks, so a report holding ranges 06-01→06-03 and 06-02→06-04 carries
    every fill in the overlap twice. Nothing downstream de-duplicated it, so Qty,
    Commission and RealizedPnl all doubled — reproduced: one fill of 100 AAPL @ 200
    (comm −1.00, realised 50.00) summed to 200 / −2.00 / 100.00.

    Safe because IBKR gives partial fills of one order **distinct** ``ibExecID``s
    (``…01.01``, ``…02.01``), and ``parse.py`` already filters to
    ``levelOfDetail="EXECUTION"`` so ORDER/CLOSED_LOT aggregate siblings never reach
    here. A repeated ExecId is therefore the same fill reported twice, not two fills.

    Note this is the opposite policy from the **live** execution stream, where
    ``ibda/tests_jvm/test_views.py`` asserts the spec must never carry
    ``dedupe_keys`` because that table is an event log. The difference is the source,
    not the schema: a live stream delivers each execution once by construction,
    whereas a Flex report is a re-queryable document whose blocks can overlap by
    design. De-duplicating the event log would hide real re-fires; not
    de-duplicating the document double-counts money.
    """
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    repeat_ibkr_ids = 0
    repeat_synthetic_ids = 0
    for row in execution_rows:
        exec_id: str = str(row.get("ExecId") or "")
        if exec_id and exec_id in seen:
            if exec_id.startswith(_SYNTHETIC_EXEC_ID_PREFIX):
                repeat_synthetic_ids += 1
            else:
                repeat_ibkr_ids += 1
            continue
        if exec_id:
            seen.add(exec_id)
        deduped.append(row)

    if repeat_ibkr_ids:
        logger.warning(
            "Dropped %d duplicate execution row(s) sharing an IBKR ibExecID — the "
            "same fill was reported more than once (overlapping <FlexStatement> "
            "ranges) and would otherwise double Qty, Commission and RealizedPnl",
            repeat_ibkr_ids,
        )
    if repeat_synthetic_ids:
        logger.warning(
            "Dropped %d execution row(s) identical in Account/Sym/DateTime/Qty/Price "
            "to an earlier row. This report carries no ibExecID, so two genuinely "
            "separate identical fills are indistinguishable from a duplicate here — "
            "select ibExecID in the Flex query to resolve it",
            repeat_synthetic_ids,
        )
    return deduped


def _dedupe_cash_rows(cash_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop repeat cash rows by ``TxnId``, keeping the first occurrence.

    A single Flex response can carry the same movement more than once — most
    obviously when it holds several ``<FlexStatement>`` blocks whose date ranges
    overlap. Summing that table counts the movement twice, which moves money in
    every downstream figure that reads it.

    Two kinds of repeat are collapsed, and they carry different confidence:

    * a repeated IBKR ``transactionID`` is the same transaction, full stop;
    * a repeated ``syn-`` id means two rows agree on every field this adapter can
      see. That is almost always the same movement reported twice — but it cannot
      be distinguished from two real movements that happen to be identical, so the
      warning says so and names the count.

    This is where every consumer funnels through, so it is the one place the
    de-duplication has to happen. It cannot span separate calls: loading two
    overlapping reports one after another still yields the row twice, once per
    call. What makes that case fixable is the id itself — it is derived only from
    row content, so the two loads agree on it and a store can key on it.
    """
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    repeat_ibkr_ids = 0
    repeat_synthetic_ids = 0
    for row in cash_rows:
        txn_id: str = row["TxnId"]
        if txn_id in seen:
            if txn_id.startswith(_SYNTHETIC_CASH_ID_PREFIX):
                repeat_synthetic_ids += 1
            else:
                repeat_ibkr_ids += 1
            continue
        seen.add(txn_id)
        deduped.append(row)

    if repeat_ibkr_ids:
        logger.warning(
            "Dropped %d duplicate cash row(s) sharing an IBKR transactionID — the "
            "same transaction was reported more than once and would otherwise be "
            "counted more than once",
            repeat_ibkr_ids,
        )
    if repeat_synthetic_ids:
        logger.warning(
            "Dropped %d cash row(s) identical in Account/Timestamp/Type/Sym/Amount/"
            "Currency to an earlier row. Keeping one avoids double-counting a "
            "re-reported movement, but this report carries no transactionID, so "
            "genuinely separate identical movements are indistinguishable from "
            "duplicates here — select transactionID in the Flex query to resolve it",
            repeat_synthetic_ids,
        )
    return deduped
