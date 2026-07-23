"""ibda.adapters.ibkr.specs — pure canonical↔raw mapping data for IBKR tables.

PURE: no engine, no vendor imports. All data is plain dataclass instances and
constants. This module may be imported from anywhere in ibda without
triggering the boundary guard.

Usage::

    from ibda.adapters.ibkr.specs import CANONICAL_SPECS, null_columns
    from ibda.schema import POSITION

    spec = CANONICAL_SPECS["position"]
    missing = null_columns(spec, POSITION)
"""
from __future__ import annotations

from dataclasses import dataclass

from ibda.schema import ALL as ALL_SCHEMAS
from ibda.schema._core import Schema


@dataclass(frozen=True)
class IbkrTableSpec:
    """Mapping descriptor from one deephaven-ib raw table to a canonical schema.

    Attributes
    ----------
    raw_table:
        The deephaven-ib table name (key in ``session.tables``).
    renames:
        Mapping of raw column names to canonical column names.  Only columns
        that exist in the raw table appear here.  Canonical columns absent from
        ``renames`` must be null-filled by the engine helper.
    schema_name:
        The canonical schema name (key in ``ibda.schema.ALL``).
    join_table:
        Optional deephaven-ib raw table name to natural_join before renaming.
        When set, ``apply_canonical_view`` performs the join BEFORE applying
        renames, so that ``join_cols`` are available for subsequent renaming.
        The caller must pass ``join_raw`` (the live deephaven Table for this
        table name) to ``apply_canonical_view``.
    join_on:
        Column expression for the ``natural_join on=`` parameter.  Supports
        both ``"ColName"`` (same name on both tables) and ``"Left=Right"``
        (when the join key has different names on each side).  Only used when
        ``join_table`` is set.  The left table is ``raw_table``; the right
        table is ``join_table``.
    join_cols:
        Raw column names to pull from ``join_table`` into the result (the
        ``joins=`` list in ``natural_join``).  Only used when ``join_table``
        is set.
    join_cast_left_col:
        When the join-key column on the left table has a different type than
        the matching column on the right table, set this to the left table's
        raw column name that needs casting to long (``int64``) before joining.
        Currently supports String→long only (the ``accounts_pnl_single.ConId``
        case where ConId is a String parsed from the IB request Note).
        ``None`` (default) means no pre-join cast is performed.
    join_dedupe_raw_keys:
        Optional raw column names (on ``join_table``, BEFORE renaming) to
        ``last_by`` the right-hand table on before joining.  Required whenever
        ``join_table`` is itself a current-state-per-key stream that can emit
        more than one row for the same join key (e.g. a corrected commission
        report re-sent for the same ExecId) — ``natural_join`` errors if more
        than one right row matches a given left row.  ``None`` (default) means
        the right table is joined as-is (correct only when the caller has
        already guaranteed at most one row per join key, e.g. a small
        auto-populated reference table like ``news_providers``).
    dedupe_keys:
        Optional list of canonical column names (after renames) to deduplicate
        on.  When set, ``apply_canonical_view`` applies ``.last_by(dedupe_keys)``
        as the final step, reducing the append-only table to one row per unique
        key combination (the most-recent row).

        Use for current-state tables (e.g. ``accounts_positions``) where
        deephaven-ib writes one row per update and the canonical view must
        reflect the latest state only.  Do NOT set this on event-log tables
        (e.g. ``execution``) — those must retain all rows.
    """

    raw_table: str
    renames: dict[str, str]
    schema_name: str
    join_table: str | None = None
    join_on: str | None = None
    join_cols: tuple[str, ...] = ()
    join_cast_left_col: str | None = None
    join_dedupe_raw_keys: tuple[str, ...] | None = None
    dedupe_keys: list[str] | None = None


# ---------------------------------------------------------------------------
# Canonical specs (all five live-view tables)
# NOTE: ``account`` is intentionally absent — it is a KEY-VALUE → wide PIVOT,
# handled separately in supervisor.canonical_account_snapshot().
# ---------------------------------------------------------------------------

CANONICAL_SPECS: dict[str, IbkrTableSpec] = {
    "position": IbkrTableSpec(
        raw_table="accounts_positions",
        renames={
            "Account": "Account",
            "ContractId": "ConId",
            "Symbol": "Sym",
            "SecType": "SecType",        # accounts_positions col #8 — intrinsic position attribute
            "Position": "Qty",
            "AvgCost": "AvgCost",
            # Option/future identity columns — deephaven-ib's position/portfolio
            # callbacks log the full Contract object (logger_contract in the
            # vendored fork), so these are always populated per-position, unlike
            # contracts_details (source of `definition`), which is populated only
            # per explicit reqContractDetails request and is sparse for a full
            # book. Multiplier arrives as a double here (logger_contract's
            # map_multiplier defaults empty -> 1.0 before writing) — no string
            # cast needed, unlike execution/definition's Multiplier.
            "Right": "Right",
            "Strike": "Strike",
            "LastTradeDateOrContractMonth": "LastTradeDateOrContractMonth",
            "LocalSymbol": "LocalSymbol",
            "Multiplier": "Multiplier",
            # Currency/Exchange — also from logger_contract; needed by
            # subscribe.py's Contract reconstruction for reqMktData/reqPnLSingle.
            "Currency": "Currency",
            "Exchange": "Exchange",
        },
        # MarketPrice / MarketValue / UnrealizedPnl are not in accounts_positions
        # raw table — they will be null-filled by apply_canonical_view.
        schema_name="position",
        # accounts_positions is an append-only table. deephaven-ib issues two
        # overlapping reqPositionsMulti subscriptions on connect (an "All" sweep
        # plus a per-account repeat from the managedAccounts callback), writing
        # each position twice at startup. Intraday updates also append rows.
        # last_by reduces to the most-recent row per (Account, ConId), making
        # the view a correct current-state snapshot regardless of duplicates.
        dedupe_keys=["Account", "ConId"],
    ),
    "execution": IbkrTableSpec(
        raw_table="orders_exec_details",
        renames={
            "ExecId": "ExecId",
            "Timestamp": "Timestamp",
            "Account": "Account",
            "ContractId": "ConId",
            "OrderId": "OrderId",
            "OrderRef": "OrderRef",
            "Symbol": "Sym",
            "SecType": "SecType",
            "Side": "Side",
            "Shares": "Qty",
            "Price": "Price",
            "Multiplier": "Multiplier",
            "ExecutionExchange": "Venue",
            "Currency": "Currency",
            # From orders_exec_commission_report (right table, via join):
            "Commission": "Commission",
            "RealizedPNL": "RealizedPnl",   # raw column is all-caps PNL
        },
        # Liquidity has no source anywhere in deephaven-ib (neither
        # orders_exec_details nor orders_exec_commission_report carry a
        # maker/taker flag) — remains null-filled by apply_canonical_view.
        # OpenClose also has no source anywhere in deephaven-ib — the Flex
        # equivalent (openCloseIndicator) has no orders_exec_details analogue —
        # remains null-filled on the live path; only the Flex mapper populates it.
        # Multiplier arrives as string in orders_exec_details → cast to double
        # via _STRING_TO_DOUBLE_COLS in views.py.
        # Side vocabulary: orders_exec_details.Side arrives as IB's raw
        # execDetails.side value ("BOT"/"SLD"), not the canonical "BUY"/"SELL"
        # the Flex mapper emits — apply_canonical_view normalizes it (see
        # _SIDE_NORMALIZE_COLS in views.py).
        # Commission sign: commissionReport.commission is a positive magnitude
        # on the live path (unlike Flex's already-negative ibCommission) —
        # apply_canonical_view negates it to match the negative=cost convention
        # (see _COMMISSION_SIGN_FLIP_COLS in views.py). The standalone
        # `commission` spec below reads the SAME raw table independently and
        # is negated the same way — both are single negations of the same raw
        # positive value, not a chained double-flip of each other.
        schema_name="execution",
        # Commission/RealizedPnl arrive on a SEPARATE async callback
        # (commissionReport, not execDetails) — see the `commission` spec
        # above. Join execution ⋈ commission ON ExecId so a matched fill gets
        # its real Commission/RealizedPnl and an unmatched fill (report not
        # yet arrived, or this session never received it — e.g. a read-only
        # connection, or a fill that predates this process) gets NULL, never
        # 0.0 (natural_join is left-outer: every execution row survives).
        join_table="orders_exec_commission_report",
        join_on="ExecId",
        join_cols=("Commission", "RealizedPNL"),
        # orders_exec_commission_report can re-emit a correction for the same
        # ExecId; natural_join errors on >1 right-side match per left row, so
        # the right table must be reduced to one (latest) row per ExecId first.
        join_dedupe_raw_keys=("ExecId",),
    ),
    "order": IbkrTableSpec(
        raw_table="orders_submitted",
        renames={
            "OrderId": "OrderId",
            "PermId": "PermId",
            "Account": "Account",
            "Symbol": "Sym",
            "Action": "Side",              # orders_submitted names BUY/SELL "Action"
            "TotalQuantity": "Qty",
            "FilledQuantity": "FilledQty",
            "LmtPrice": "LimitPrice",
            "Status": "Status",
            "ReceiveTime": "Timestamp",    # last status-update time (UTC)
        },
        # orders_submitted (162 cols) carries every canonical order column,
        # including Status and FilledQuantity — no orders_status join needed.
        # NOTE: the table is empty in READ-ONLY sessions; connect read_only=False
        # (and uncheck TWS "Read-Only API") for it to populate.
        schema_name="order",
    ),
    "definition": IbkrTableSpec(
        raw_table="contracts_details",
        renames={
            "ContractId": "ConId",
            "Symbol": "Sym",
            "SecType": "SecType",
            "PrimaryExchange": "Exchange",
            "Currency": "Currency",
            "Multiplier": "Multiplier",
            "Right": "Right",
            "Strike": "Strike",
            "LastTradeDateOrContractMonth": "LastTradeDateOrContractMonth",
            "LocalSymbol": "LocalSymbol",
        },
        # Multiplier may be string in the raw table → cast to FLOAT64 done by
        # apply_canonical_view (needs engine; handled there). Right/Strike/
        # LastTradeDateOrContractMonth/LocalSymbol map 1:1 from contracts_details
        # (Strike is already Float64 on the raw table; the others are String, no
        # cast needed) — null for non-option/non-future contracts.
        schema_name="definition",
    ),
    "quote": IbkrTableSpec(
        raw_table="ticks_bid_ask",
        renames={
            "Symbol": "Sym",
            "Timestamp": "Timestamp",
            "BidPrice": "Bid",
            "AskPrice": "Ask",
            "BidSize": "BidSize",
            "AskSize": "AskSize",
        },
        # Last is not present in ticks_bid_ask (would need ticks_price LAST tick)
        # → null-filled.  Table is empty until reqMktData() subscriptions are made.
        schema_name="quote",
    ),
    "trade": IbkrTableSpec(
        raw_table="ticks_trade",
        renames={
            "Symbol": "Sym",
            "Timestamp": "Timestamp",
            "Price": "Price",
            "Size": "Size",
        },
        # All canonical trade columns map 1-to-1; no null-fill required.
        # Empty until a tick-by-tick LAST subscription is made (see the
        # trades() tool / request_tick_data_historical).
        schema_name="trade",
    ),
    "bar": IbkrTableSpec(
        raw_table="bars_realtime",
        renames={
            "Symbol": "Sym",
            "Timestamp": "Timestamp",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume",
        },
        # All bar columns map 1-to-1; no null-fill required.
        # Table is empty until request_bars_realtime() subscriptions are made.
        schema_name="bar",
    ),
    "position_pnl": IbkrTableSpec(
        raw_table="accounts_pnl_single",
        renames={
            # From accounts_pnl_single (left table after ConId String→long cast):
            "Account": "Account",
            "ConId": "ConId",      # String in raw; cast to long by apply_canonical_view
                                   # before join; canonical type is INT64.
            "DailyPnL": "DailyPnl",       # raw uses capital L; canonical uses lowercase l
            "UnrealizedPnL": "UnrealizedPnl",
            "RealizedPnL": "RealizedPnl",
            "Value": "MarketValue",
            # From accounts_positions (right table, via join):
            "Symbol": "Sym",       # accounts_positions raw column name for symbol
        },
        schema_name="position_pnl",
        # Join accounts_positions to resolve Symbol from ContractId.
        # Left join key: ConId (cast to long from String before join).
        # Right join key: ContractId (int64 in accounts_positions).
        join_table="accounts_positions",
        join_on="ConId=ContractId",    # left=right key expression
        join_cols=("Symbol",),         # pull Symbol from accounts_positions
        join_cast_left_col="ConId",    # cast accounts_pnl_single.ConId String→long
        # accounts_positions is append-only and deephaven-ib issues overlapping
        # reqPositionsMulti subscriptions on connect (see the "position" spec's
        # dedupe_keys comment above) — the SAME raw table joined here as the
        # RIGHT side can carry more than one row for a given ContractId.
        # natural_join errors on more than one right-side match per left row,
        # so the right table must be reduced to one (latest) row per
        # ContractId first — mirrors the "execution" spec's commission join.
        join_dedupe_raw_keys=("ContractId",),
    ),
    "commission": IbkrTableSpec(
        raw_table="orders_exec_commission_report",
        renames={
            "ExecId": "ExecId",
            "ReceiveTime": "Timestamp",
            "Commission": "Commission",
            "Currency": "Currency",
            "RealizedPNL": "RealizedPnl",   # raw column is all-caps PNL
            "Yield": "Yield",
            "YieldRedemptionDate": "YieldRedemptionDate",
        },
        schema_name="commission",
        # orders_exec_commission_report arrives on a separate async callback from
        # orders_exec_details (execDetails vs commissionReport); a correction can
        # resend a report for the same ExecId. last_by keeps the most-recent report
        # per execution — commission is a current-state-per-key table, not an
        # event log (contrast with execution, which retains every fill).
        # Commission sign: negated by apply_canonical_view to the negative=cost
        # convention (see _COMMISSION_SIGN_FLIP_COLS in views.py), matching the
        # canonical `execution.Commission` column above — both read this same
        # raw table independently, so this is a single negation of the raw
        # positive value, not a double-flip of execution's already-negated one.
        dedupe_keys=["ExecId"],
    ),
    "news": IbkrTableSpec(
        raw_table="news_historical",
        renames={
            "Timestamp": "Timestamp",
            "Symbol": "Sym",
            "ContractId": "ConId",
            "ProviderCode": "ProviderCode",
            "ArticleId": "ArticleId",
            "Headline": "Headline",
            # From news_providers (right table, via join):
            "Name": "ProviderName",
        },
        schema_name="news",
        # news_historical is empty until subscribe_news()/request_news_historical()
        # issues a request per contract (ibda.adapters.ibkr.marketdata). Joined
        # against news_providers (small, auto-populated on connect) to resolve
        # ProviderName from ProviderCode — same shape as the position_pnl join.
        join_table="news_providers",
        join_on="ProviderCode=Code",    # left=right key expression
        join_cols=("Name",),
        # news_providers can emit a re-sent row for the same Code (see the
        # "news_provider" spec's dedupe_keys comment / ibda/schema/news_provider.py
        # docstring) — the SAME raw table joined here as the RIGHT side needs the
        # same guard: natural_join errors on more than one right-side match per
        # left row, so reduce to one (latest) row per Code first.
        join_dedupe_raw_keys=("Code",),
        # Append-only event log — no dedupe_keys (this is the news spec's OWN
        # final-reduction step, unrelated to join_dedupe_raw_keys above).
    ),
    "errors": IbkrTableSpec(
        raw_table="errors",
        renames={
            "ReceiveTime": "Timestamp",
            "RequestId": "ReqId",
            "ErrorCode": "Code",
            "ErrorDescription": "Message",
            "Error": "Detail",
            "Note": "Note",
            "Tier": "Severity",   # vendored deephaven-ib fork's tier classifier column
            "Symbol": "Sym",
            "ContractId": "ConId",
        },
        schema_name="errors",
        # Auto-populated from connect (every error()/notification callback).
        # Append-only event log — no dedupe_keys.
    ),
    "account_pnl": IbkrTableSpec(
        raw_table="accounts_pnl",
        renames={
            "Account": "Account",
            "ReceiveTime": "Timestamp",
            "DailyPnl": "DailyPnl",
            "UnrealizedPnl": "UnrealizedPnl",
            "RealizedPnl": "RealizedPnl",
        },
        schema_name="account_pnl",
        # accounts_pnl is append-only (deephaven-ib writes a new row per PnL
        # update); last_by(Account) reduces it to current-state-per-account,
        # matching ACCOUNT_PNL's documented "one row per account, refreshes
        # intraday" semantics. Auto-populated: deephaven-ib's managedAccounts
        # callback calls request_account_pnl(account) for every managed
        # account — no explicit subscription needed from ibda.
        dedupe_keys=["Account"],
    ),
    "price_tick": IbkrTableSpec(
        raw_table="ticks_price",
        renames={
            "ReceiveTime": "Timestamp",
            "ContractId": "ConId",
            "Symbol": "Sym",
            "TickType": "TickType",
            "Price": "Price",
        },
        schema_name="price_tick",
        # Sym comes straight from ticks_price.Symbol — no join needed (unlike
        # position_pnl/news, which resolve Sym/ProviderName via a join).
        # Streaming event log, empty until request_market_data() is called
        # per contract — no dedupe_keys (every TickType/price update kept).
    ),
    "news_provider": IbkrTableSpec(
        raw_table="news_providers",
        renames={
            "Code": "Code",
            "Name": "Name",
        },
        schema_name="news_provider",
        # Small auto-populated reference table; last_by guards against a
        # re-emitted row for the same Code (mirrors the "commission" spec's
        # rationale for a current-state-per-key table).
        dedupe_keys=["Code"],
    ),
}


# Fallback position spec sourced from accounts_portfolio (deephaven-ib's
# reqAccountUpdates/updatePortfolio stream).  Used by connect_live when
# accounts_positions is empty (read-only TWS login withholds reqPositions).
# NOT part of CANONICAL_SPECS — the position view is built from whichever raw
# table has rows.  accounts_portfolio natively carries MarketPrice/MarketValue/
# UnrealizedPnL, so those canonical columns get populated on this path.
POSITION_PORTFOLIO_SPEC: IbkrTableSpec = IbkrTableSpec(
    raw_table="accounts_portfolio",
    renames={
        "Account": "Account",
        "ContractId": "ConId",
        "Symbol": "Sym",
        "SecType": "SecType",
        "Position": "Qty",
        "AvgCost": "AvgCost",
        "MarketPrice": "MarketPrice",
        "MarketValue": "MarketValue",
        "UnrealizedPnL": "UnrealizedPnl",
        # Same logger_contract-sourced identity columns as accounts_positions
        # (updatePortfolio logs the full Contract object too) — see the
        # CANONICAL_SPECS["position"] comment above.
        "Right": "Right",
        "Strike": "Strike",
        "LastTradeDateOrContractMonth": "LastTradeDateOrContractMonth",
        "LocalSymbol": "LocalSymbol",
        "Multiplier": "Multiplier",
        "Currency": "Currency",
        "Exchange": "Exchange",
    },
    schema_name="position",
    dedupe_keys=["Account", "ConId"],
)


def null_columns(spec: IbkrTableSpec, schema: Schema) -> list[str]:
    """Return canonical columns that are not covered by ``spec.renames``.

    These columns must be null-filled when building the canonical view.  The
    union of ``set(spec.renames.values())`` and the returned list equals the
    full set of canonical column names.

    Parameters
    ----------
    spec:
        The mapping descriptor for one IBKR raw table.
    schema:
        The canonical Schema for that table.

    Returns
    -------
    List of canonical column names absent from ``spec.renames.values()``.
    """
    covered: set[str] = set(spec.renames.values())
    return [col.name for col in schema.columns if col.name not in covered]


def get_schema(spec: IbkrTableSpec) -> Schema:
    """Return the canonical Schema for *spec* from the global registry."""
    return ALL_SCHEMAS[spec.schema_name]
