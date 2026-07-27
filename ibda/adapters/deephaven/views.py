"""ibda.adapters.deephaven.views — canonical view and snapshot helpers.

ENGINE-CONFINED: this module imports deephaven; only adapters/deephaven/ may
do so.

Two public functions:

``apply_canonical_view(raw, spec, schema) -> Any``
    Build a live deephaven Table that conforms to *schema* from a raw
    deephaven-ib table according to *spec*.  The result MUST pass
    ``schema.validate(to_arrow(result))``.

``snapshot_rows(raw_table) -> list[dict[str, Any]]``
    Read a raw deephaven Table to a list of plain Python dicts (one per row).
    Engine-only helper; the supervisor calls this to avoid importing deephaven
    directly.

Deephaven API surface used here (all verified against the JVM test):
- ``table.rename_columns(["NewName=OldName", ...])``          — rename in place
- ``table.view(["Col", "NewName=OldName", "NewCol=(type)expr"])`` — project+expr
- ``table.meta_table``                                        — schema metadata
- ``deephaven.pandas.to_pandas(table)``                       — read to DataFrame
- Groovy-style null literals: ``NULL_DOUBLE``, ``NULL_LONG``   — in view exprs
- String→double cast in Groovy: ``Double.parseDouble(col)``   — for Multiplier

Engine note on Multiplier (contracts_details): the raw column type varies by
deephaven-ib version and connection mode.  In the paper/live account the column
arrives as ``double``; in some environments it is a ``String``.  We inspect the
actual column DataType via ``tbl.meta_table`` and choose the correct expression:
- String column → ``isNull(M) || M.isEmpty() ? NULL_DOUBLE : Double.parseDouble(M)``
- double column → no cast needed (already FLOAT64)
- other numeric  → ``(double)M``
"""
from __future__ import annotations

import logging
from typing import Any, cast

from ibda.adapters.ibkr.specs import IbkrTableSpec, null_columns
from ibda.schema._core import DType, Schema

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Null literal expressions for each canonical dtype
# ---------------------------------------------------------------------------

_NULL_LITERAL: dict[DType, str] = {
    DType.STRING: "null",
    DType.INT64: "NULL_LONG",
    DType.FLOAT64: "NULL_DOUBLE",
    DType.BOOL: "(Boolean)null",
    DType.TIMESTAMP_NS: "(Instant)null",
}

# Columns that conceptually hold a double value but whose raw type varies by
# deephaven-ib version / connection mode and needs a type-aware cast.
# Detected by (schema_name, canonical_col) pair.  The actual cast expression
# is chosen at runtime based on the real DataType in the renamed table.
_STRING_TO_DOUBLE_COLS: set[tuple[str, str]] = {
    ("definition", "Multiplier"),
    ("execution", "Multiplier"),
}

# Columns whose live-path RAW vocabulary/sign differs from the canonical
# convention and needs a value-level (not type-level) transform.  Detected by
# (schema_name, canonical_col) pair, mirroring _STRING_TO_DOUBLE_COLS above.
#
# ("execution", "Side"): IB's execDetails.side callback (source of
#   orders_exec_details.Side) emits "BOT"/"SLD", not the canonical "BUY"/"SELL"
#   the Flex mapper already emits (ibda.adapters.ibkr.flex.mapping._map_execution
#   passes through Flex's buySell attribute, which IS "BUY"/"SELL"). Normalized
#   here so both producers of the canonical `execution` table agree.
#   CONFIRMED against the live book: raw execDetails.side arrives as
#   "BOT"/"SLD" (see the comment on the expression below).
#
# ("execution", "Commission") / ("commission", "Commission"): commissionReport.
#   commission is a positive magnitude on the live path, whereas Flex's
#   ibCommission is already negative. Project convention is negative = cost;
#   negated here so both the `execution` and standalone `commission` canonical
#   tables agree with Flex and with each other. Both specs source
#   orders_exec_commission_report INDEPENDENTLY — execution.Commission via a
#   natural_join of the RAW commission table (spec.join_table in specs.py),
#   commission.Commission via the RAW commission table directly as its own
#   raw_table — so this is two single negations of the same raw positive
#   value, not a chained double-flip of one canonical output onto the other.
#   CONFIRMED against the live book: raw commissionReport.commission
#   arrives as a positive magnitude (e.g. 0.048029, 0.022261, 0.003 USD).
_SIDE_NORMALIZE_COLS: set[tuple[str, str]] = {("execution", "Side")}
_COMMISSION_SIGN_FLIP_COLS: set[tuple[str, str]] = {
    ("execution", "Commission"),
    ("commission", "Commission"),
}

# Live PnL columns that carry IBKR's Double.MAX_VALUE (1.7976931348623157e308)
# sentinel for "not yet computed" instead of a null. Mirrors the scrub already
# applied on the Arrow snapshot path (``ibda.adapters.ibkr.pnl._f``: "IBKR uses
# Double.MAX_VALUE as a sentinel for 'not yet available'") and the vendored
# deephaven-ib fork's commissionReport ``map_null_value`` scrub of the same
# sentinel for RealizedPNL. Detected by (schema_name, canonical_col) pair,
# mirroring _STRING_TO_DOUBLE_COLS / _SIDE_NORMALIZE_COLS above.
#
# Unlike the Arrow snapshot path, the live deephaven-view path
# (apply_canonical_view) previously did no sentinel handling at all: a
# ``sum(RealizedPnl)`` (or any other component) over the live ~2,100-position
# book would add ~1.8e308 per not-yet-computed row — an astronomically wrong
# aggregate rather than a merely-incomplete one. Scrubbed here so both the
# Arrow snapshot and live-view paths agree that "not yet available" is
# represented as null, never as the raw IBKR sentinel value.
_PNL_SENTINEL_SCRUB_COLS: set[tuple[str, str]] = {
    ("position_pnl", "DailyPnl"),
    ("position_pnl", "UnrealizedPnl"),
    ("position_pnl", "RealizedPnl"),
    ("position_pnl", "MarketValue"),
    ("account_pnl", "DailyPnl"),
    ("account_pnl", "UnrealizedPnl"),
    ("account_pnl", "RealizedPnl"),
}


def _col_types(tbl: Any) -> dict[str, str]:
    """Return {column_name: DataType_string} by reading *tbl*'s meta_table.

    Uses ``deephaven.pandas.to_pandas`` which is already imported at call
    sites.  The meta_table has columns ``Name`` and ``DataType``; DataType is
    a Java class object whose string representation is used for matching:
    - String column  → contains ``"String"``  (e.g. ``java.lang.String``)
    - double column  → ``"double"``
    - long column    → ``"long"``
    - float column   → ``"float"``
    - int column     → ``"int"``
    """
    from deephaven.pandas import to_pandas  # noqa: PLC0415 — engine-confined

    meta_df = to_pandas(tbl.meta_table)
    return {
        str(row["Name"]): str(row["DataType"])
        for _, row in meta_df.iterrows()
    }


def _double_cast_expr(col_name: str, datatype_str: str) -> str | None:
    """Return a Groovy update_view expression that ensures *col_name* is double.

    Returns ``None`` when no cast is needed (column is already double).

    Parameters
    ----------
    col_name:
        The canonical column name as it appears after rename.
    datatype_str:
        The DataType string from meta_table (e.g. ``"double"``,
        ``"java.lang.String"``).
    """
    if "String" in datatype_str:
        # String column: guard against null/empty before parseDouble.
        return (
            f"{col_name} = isNull({col_name}) || {col_name}.isEmpty() "
            f"? NULL_DOUBLE : Double.parseDouble({col_name})"
        )
    if datatype_str == "double":
        # Already the right type — no cast needed.
        return None
    # Other numeric types (long, int, float, …): widen to double.
    return f"{col_name} = (double){col_name}"


def apply_canonical_view(
    raw: Any,
    spec: IbkrTableSpec,
    schema: Schema,
    join_raw: Any = None,
) -> Any:
    """Build a live deephaven Table conforming to *schema* from a raw table.

    Steps:
    0. (Optional join) If *spec.join_table* is set, perform a pre-join step
       (the caller MUST pass the corresponding *join_raw* table in this case —
       see the *join_raw* parameter note below):
       a. If *spec.join_cast_left_col* is set, cast that left-table column from
          String to long via ``update_view`` (required for ``accounts_pnl_single.ConId``
          which is a String but must match ``accounts_positions.ContractId`` (long)).
       b. If *spec.join_dedupe_raw_keys* is set, reduce *join_raw* to one row
          per key via ``last_by`` (required when the right table can emit more
          than one row for the same join key — e.g. a corrected commission
          report resent for the same ExecId — since ``natural_join`` errors on
          more than one right-side match per left row).
       c. ``natural_join(join_raw, on=spec.join_on, joins=list(spec.join_cols))``
          — LEFT-outer semantics: every left row survives; a left row with no
          matching right row gets NULL in the joined columns (verified under a
          real JVM — see ``ibda/tests_jvm/test_views.py``
          ``test_execution_view_unmatched_exec_id_yields_null_commission``).
    1. Rename raw columns to canonical names per *spec.renames*.
    1.5. Value-level normalizations for columns whose live-path vocabulary or
         sign differs from the canonical convention (e.g. execution.Side's
         "BOT"/"SLD" → "BUY"/"SELL"; execution.Commission's positive magnitude
         → negative=cost), plus a sentinel scrub for the PnL columns that
         carry IBKR's Double.MAX_VALUE "not yet computed" sentinel instead of
         null (position_pnl/account_pnl's DailyPnl/UnrealizedPnl/RealizedPnl/
         MarketValue). See ``_SIDE_NORMALIZE_COLS`` /
         ``_COMMISSION_SIGN_FLIP_COLS`` / ``_PNL_SENTINEL_SCRUB_COLS``.
    2. If any renamed column needs a numeric cast (e.g. Multiplier → double),
       inspect the actual DataType via the table's meta_table and choose the
       correct Groovy expression: String→parseDouble, already-double→no-op,
       other numeric→(double) widening cast.
    3. Project to exactly the canonical column set: use the renamed columns
       directly and inject null literals for columns absent in the raw table.

    Parameters
    ----------
    raw:
        A live deephaven Table (the raw deephaven-ib table).
    spec:
        The mapping descriptor (raw_table, renames, schema_name, and optional
        join_table / join_on / join_cols / join_cast_left_col /
        join_dedupe_raw_keys fields).
    schema:
        The canonical Schema the result must conform to.
    join_raw:
        The live deephaven Table for *spec.join_table*.  REQUIRED when
        *spec.join_table* is set — omitting it while the spec declares a join
        raises when the join-sourced columns are referenced downstream (rename/
        view), since those columns do not exist on *raw* alone.  Ignored when
        *spec.join_table* is ``None``.

    Returns
    -------
    A live deephaven Table with exactly the canonical columns in schema order.
    """
    tbl = raw

    # Step 0: optional pre-join.
    if spec.join_table is not None and join_raw is not None:
        # Step 0a: cast join-key column from String to long when required.
        if spec.join_cast_left_col is not None:
            cast_col = spec.join_cast_left_col
            # Guard against null/empty String before parseLong.
            tbl = tbl.update_view([
                f"{cast_col} = isNull({cast_col}) || {cast_col}.isEmpty() "
                f"? NULL_LONG : Long.parseLong({cast_col})"
            ])
        # Step 0b: reduce the right table to one row per join key when the
        # spec says it can emit duplicates for the same key (e.g. a corrected
        # commission report resent for the same ExecId) — natural_join errors
        # on more than one right-side match per left row.
        if spec.join_dedupe_raw_keys is not None:
            join_raw = join_raw.last_by(list(spec.join_dedupe_raw_keys))
        # Step 0c: natural_join to bring in columns from the right table.
        # LEFT-outer semantics: every left (raw) row survives; a left row with
        # no matching right row gets NULL in the joined columns rather than
        # being dropped or defaulted to 0.0.
        tbl = tbl.natural_join(join_raw, on=spec.join_on, joins=list(spec.join_cols))

    # Step 1: rename raw → canonical column names.
    # deephaven rename_columns takes ["NewName=OldName"] pairs; only include
    # renames where the names actually differ to avoid no-op rename errors.
    rename_exprs = [
        f"{canonical}={raw_col}"
        for raw_col, canonical in spec.renames.items()
        if raw_col != canonical
    ]
    tbl = tbl.rename_columns(rename_exprs) if rename_exprs else tbl

    renamed_cols: set[str] = set(spec.renames.values())

    # Step 1.5: value-level normalizations — vocabulary/sign conventions that
    # differ between the live IB API and the canonical convention (distinct
    # from the type-level casts in Step 2 below). See _SIDE_NORMALIZE_COLS /
    # _COMMISSION_SIGN_FLIP_COLS module-level comments for the "why".
    value_exprs: list[str] = []
    if (spec.schema_name, "Side") in _SIDE_NORMALIZE_COLS and "Side" in renamed_cols:
        # IB execDetails.side is BOT/SLD — CONFIRMED against the live book.
        value_exprs.append(
            'Side = isNull(Side) ? Side : '
            '(Side.equals("BOT") ? "BUY" : (Side.equals("SLD") ? "SELL" : Side))'
        )
    if (
        (spec.schema_name, "Commission") in _COMMISSION_SIGN_FLIP_COLS
        and "Commission" in renamed_cols
    ):
        # live commissionReport.commission is a positive magnitude —
        # CONFIRMED against the live book (e.g. 0.048029, 0.022261, 0.003 USD).
        value_exprs.append("Commission = isNull(Commission) ? Commission : -Commission")
    # PnL sentinel scrub: IBKR's Double.MAX_VALUE "not yet computed" sentinel
    # must become NULL_DOUBLE, never survive into a summed aggregate. Deephaven's
    # NULL_DOUBLE is -Double.MAX_VALUE (a large NEGATIVE value), so it is never
    # >= +Double.MAX_VALUE — a pre-existing null passes through this comparison
    # untouched, with no isNull guard needed (see _PNL_SENTINEL_SCRUB_COLS above).
    for col in schema.columns:
        if (spec.schema_name, col.name) in _PNL_SENTINEL_SCRUB_COLS and col.name in renamed_cols:
            value_exprs.append(
                f"{col.name} = {col.name} >= 1.7976931348623157e308 ? NULL_DOUBLE : {col.name}"
            )
    if value_exprs:
        tbl = tbl.update_view(value_exprs)

    # Step 2: type casts for columns that need it.
    # For each column in _STRING_TO_DOUBLE_COLS, inspect the actual DataType
    # from the renamed table's meta_table and choose the correct Groovy
    # expression.  This handles the live contracts_details case where
    # Multiplier arrives as a double (not a String as the fixture assumed).
    cast_exprs: list[str] = []
    needs_meta = any(
        (spec.schema_name, col.name) in _STRING_TO_DOUBLE_COLS
        and col.name in renamed_cols
        for col in schema.columns
    )
    col_type_map: dict[str, str] = _col_types(tbl) if needs_meta else {}
    for col in schema.columns:
        if (spec.schema_name, col.name) in _STRING_TO_DOUBLE_COLS:
            if col.name in renamed_cols:
                datatype_str = col_type_map.get(col.name, "")
                expr = _double_cast_expr(col.name, datatype_str)
                if expr is not None:
                    cast_exprs.append(expr)
    if cast_exprs:
        tbl = tbl.update_view(cast_exprs)

    # Step 3: project to canonical columns in schema order.
    # For columns present after rename/cast: include by name.
    # For null-fill columns: inject a typed null literal.
    null_col_names: set[str] = set(null_columns(spec, schema))
    view_exprs: list[str] = []
    for col in schema.columns:
        if col.name in null_col_names:
            null_lit = _NULL_LITERAL[col.dtype]
            view_exprs.append(f"{col.name} = ({null_lit})")
        else:
            view_exprs.append(col.name)

    tbl = tbl.view(view_exprs)

    # Step 4 (optional): deduplicate to current-state-per-key.
    # Applied to append-only tables (e.g. accounts_positions) where each update
    # appends a new row; last_by keeps the most-recent row per key combination.
    # Not applied to event-log tables (executions) — those must retain all rows.
    if spec.dedupe_keys is not None:
        tbl = tbl.last_by(spec.dedupe_keys)

    return tbl


def enrich_position_with_marks(position_view: Any, portfolio_raw: Any) -> Any:
    """Join live marks from accounts_portfolio onto a position view from accounts_positions.

    Called after ``apply_canonical_view`` has built the position view from
    ``accounts_positions``.  That view has MarketPrice/MarketValue/UnrealizedPnl
    null-filled (accounts_positions does not carry live valuation columns).  This
    function replaces those nulls with live marks from ``accounts_portfolio``,
    matched by ConId.

    Steps
    -----
    1. Deduplicate ``accounts_portfolio`` to one row per ContractId (latest mark).
       accounts_portfolio is append-only; ``last_by`` ensures the most-recent mark wins.
    2. Select and rename mark columns to canonical names:
       - ContractId  → ConId
       - UnrealizedPnL (capital L in raw)  → UnrealizedPnl (canonical casing)
    3. Drop the null-filled placeholder columns from the position view.
       Without this drop, ``natural_join`` would fail with a duplicate-column error.
    4. ``natural_join`` (left semantics): all position rows survive; mark columns are
       null when a ConId has no matching row in ``accounts_portfolio``.

    Column order is preserved: the three mark columns appear last in the POSITION
    schema (after AvgCost) and the join appends them in that same order.

    Parameters
    ----------
    position_view:
        The canonical position deephaven Table produced by ``apply_canonical_view``
        over ``accounts_positions``.  Must have columns Account, ConId, Sym, SecType,
        Qty, AvgCost and the null-filled placeholders MarketPrice, MarketValue,
        UnrealizedPnl.
    portfolio_raw:
        The live deephaven ``accounts_portfolio`` raw table.  Must be non-empty
        (the caller — ``_enrich_position_marks`` in ``live.py`` — guards this).

    Returns
    -------
    A live deephaven Table with the same structure as ``position_view`` but with
    MarketPrice, MarketValue, and UnrealizedPnl populated from ``accounts_portfolio``
    for matching ConIds (null for non-matching ConIds).
    """
    # Step 1: deduplicate to the latest mark per contract.
    marks = portfolio_raw.last_by(["ContractId"])

    # Step 2: select only the three mark columns with canonical naming/casing.
    # "UnrealizedPnl=UnrealizedPnL" renames raw UnrealizedPnL (capital L) to
    # canonical UnrealizedPnl so the natural_join populates the correct column.
    marks = marks.view([
        "ConId=ContractId",
        "MarketPrice",
        "MarketValue",
        "UnrealizedPnl=UnrealizedPnL",
    ])

    # Step 3: drop null-filled placeholder columns so the join can add them back.
    # This avoids the duplicate-column error that natural_join raises when both
    # the left and right tables carry a column with the same name.
    enriched = position_view.drop_columns(["MarketPrice", "MarketValue", "UnrealizedPnl"])

    # Step 4: left join — all position rows kept; right columns null on no match.
    enriched = enriched.natural_join(
        marks, on="ConId", joins=["MarketPrice", "MarketValue", "UnrealizedPnl"]
    )

    return enriched


# ---------------------------------------------------------------------------
# Live account / cash_balance pivots (accounts_overview key-value -> wide)
#
# accounts_overview is a key-value table (Account, Currency, Key, Value,
# DoubleValue, ReceiveTime, RequestId, ...) — it cannot be expressed as an
# IbkrTableSpec (rename + null-fill + optional join), which assumes a
# columnar raw source. These two builders pivot specific Key values into
# named columns via last_by-per-key sub-views joined together — a pivot
# pattern already proven against a live IBKR account before being
# generalized into this reusable form.
# ---------------------------------------------------------------------------

# accounts_overview Key values -> canonical `account` column names. IBKR
# emits "MaintMarginReq" but the canonical column is "MaintMargin" (mirrors
# IbkrSupervisor.canonical_account_snapshot's mapping for the same pivot).
# TotalCashValue is IBKR's account-level cash summary — already
# base-currency-converted and summed across ALL currencies held (distinct
# from cash_balance's per-currency $LEDGER-CashBalance legs below); this
# mirrors a total-cash pivot helper this package's views were generalized
# from.
_ACCOUNT_METRIC_KEYS: dict[str, str] = {
    "NetLiquidation": "NetLiquidation",
    "BuyingPower": "BuyingPower",
    "MaintMarginReq": "MaintMargin",
    "GrossPositionValue": "GrossPositionValue",
    "TotalCashValue": "TotalCashValue",
}

# Per-currency ledger tags on accounts_overview. IBKR reports these under
# "$LEDGER-"-prefixed Key values (verified live against a real IBKR paper
# session); the unprefixed keys above are the account-level,
# single-currency summary and never carry more than one row / a Currency
# dimension.
_CASH_BALANCE_LEDGER_KEYS: dict[str, str] = {
    "$LEDGER-CashBalance": "CashBalance",
    "$LEDGER-NetLiquidationByCurrency": "NetLiquidation",
    "$LEDGER-ExchangeRate": "ExchangeRate",
}


def build_account_view(overview_raw: Any, account: str | None = None) -> Any:
    """Build a LIVE canonical `account` pivot view from accounts_overview.

    ``accounts_overview`` is a key-value table that ticks as TWS pushes
    account updates. Unlike ``IbkrSupervisor.canonical_account_snapshot()``
    (a v1 helper that reads the table ONCE and returns a static row dict,
    frozen at the moment it is called), this function returns a live
    deephaven Table: each of the four metrics is projected as its own
    single-row ``last_by([])`` sub-view (the latest value for that Key), and
    the four are joined together on a constant key. The result re-ticks
    whenever TWS pushes a new value for any of the four metrics — it is the
    replacement for the ``table_from_rows(ACCOUNT, ...)`` snapshot
    previously built in ``connect_live``.

    Parameters
    ----------
    overview_raw:
        The live deephaven-ib ``accounts_overview`` raw table.
    account:
        If given, restrict to rows for this account id (multi-account
        sessions); ``None`` uses all rows (the common single-account case).

    Returns
    -------
    A live deephaven Table with exactly the ``ACCOUNT`` canonical columns
    (Account, Timestamp, NetLiquidation, BuyingPower, MaintMargin,
    GrossPositionValue, Currency, TotalCashValue), one row (zero until the
    first matching update arrives), ticking thereafter.
    """
    tbl = overview_raw
    if account:
        # Falsy (None or "") skips the filter entirely — mirrors the old
        # canonical_account_snapshot()'s degradation when account discovery
        # failed (an empty self._account short-circuited its own filter, so
        # ALL rows were used rather than none).
        tbl = tbl.where(f"Account = `{account}`")

    def _metric(key: str, col: str) -> Any:
        return (
            tbl.where(f"Key = `{key}`")
            .last_by([])
            .view([
                "Account",
                "Currency",
                "Timestamp = ReceiveTime",
                f"{col} = DoubleValue",
            ])
            .update_view("_k = (int)1")
        )

    net_liq = _metric("NetLiquidation", _ACCOUNT_METRIC_KEYS["NetLiquidation"])
    buying_power = _metric("BuyingPower", _ACCOUNT_METRIC_KEYS["BuyingPower"]).view(
        ["_k", "BuyingPower"]
    )
    maint_margin = _metric("MaintMarginReq", _ACCOUNT_METRIC_KEYS["MaintMarginReq"]).view(
        ["_k", "MaintMargin"]
    )
    gross_pos = _metric(
        "GrossPositionValue", _ACCOUNT_METRIC_KEYS["GrossPositionValue"]
    ).view(["_k", "GrossPositionValue"])
    total_cash = _metric(
        "TotalCashValue", _ACCOUNT_METRIC_KEYS["TotalCashValue"]
    ).view(["_k", "TotalCashValue"])

    return (
        net_liq
        .natural_join(buying_power, on="_k", joins=["BuyingPower"])
        .natural_join(maint_margin, on="_k", joins=["MaintMargin"])
        .natural_join(gross_pos, on="_k", joins=["GrossPositionValue"])
        .natural_join(total_cash, on="_k", joins=["TotalCashValue"])
        .view([
            "Account", "Timestamp", "NetLiquidation", "BuyingPower",
            "MaintMargin", "GrossPositionValue", "Currency", "TotalCashValue",
        ])
    )


def build_cash_balance_view(overview_raw: Any, account: str | None = None) -> Any:
    """Build a LIVE canonical `cash_balance` per-currency ledger view.

    ``accounts_overview`` carries the per-currency ledger under
    ``"$LEDGER-"``-prefixed Key values (e.g. ``"$LEDGER-CashBalance"``,
    ``"$LEDGER-NetLiquidationByCurrency"``), one row per (Account, Currency)
    pair — distinct from the account-level summary keys ``build_account_view``
    pivots. This view keeps EVERY currency row reported, including the
    home-currency ``"BASE"`` mirror row; callers that want an FX-exposure
    view (excluding BASE, filtering near-zero balances) should apply that
    filtering downstream — this canonical table is the raw per-currency
    ledger, undiluted.

    Parameters
    ----------
    overview_raw:
        The live deephaven-ib ``accounts_overview`` raw table.
    account:
        If given, restrict to rows for this account id.

    Returns
    -------
    A live deephaven Table with columns Account, Currency, CashBalance,
    NetLiquidation, ExchangeRate (per currency) — ticking as TWS pushes
    ledger updates.
    """
    tbl = overview_raw
    if account:
        # Falsy (None or "") skips the filter — see build_account_view's note.
        tbl = tbl.where(f"Account = `{account}`")
    ov = tbl.where("!isNull(Currency)")

    def _per_key(key: str, col: str) -> Any:
        return (
            ov.where(f"Key = `{key}`")
            .last_by(["Account", "Currency"])
            .view(["Account", "Currency", f"{col} = DoubleValue"])
        )

    cash = _per_key("$LEDGER-CashBalance", _CASH_BALANCE_LEDGER_KEYS["$LEDGER-CashBalance"])
    net_liq_ccy = _per_key(
        "$LEDGER-NetLiquidationByCurrency",
        _CASH_BALANCE_LEDGER_KEYS["$LEDGER-NetLiquidationByCurrency"],
    )
    exchange_rate = _per_key(
        "$LEDGER-ExchangeRate",
        _CASH_BALANCE_LEDGER_KEYS["$LEDGER-ExchangeRate"],
    )

    return (
        cash.natural_join(net_liq_ccy, on=["Account", "Currency"], joins=["NetLiquidation"])
        .natural_join(exchange_rate, on=["Account", "Currency"], joins=["ExchangeRate"])
        .view(["Account", "Currency", "CashBalance", "NetLiquidation", "ExchangeRate"])
    )


def snapshot_rows(raw_table: Any) -> list[dict[str, Any]]:
    """Read a deephaven Table to a list of row dicts.

    Used by the IBKR supervisor to read ``accounts_overview`` key-value rows
    without importing deephaven directly.

    Parameters
    ----------
    raw_table:
        A deephaven Table (any schema).

    Returns
    -------
    A list of dicts, one per row, keyed by column name.  All values are plain
    Python scalars (strings, ints, floats, None, datetime objects).
    """
    from deephaven.pandas import to_pandas

    df = to_pandas(raw_table)
    return cast(list[dict[str, Any]], df.to_dict(orient="records"))


def snapshot_rows_where(raw_table: Any, predicate: str) -> list[dict[str, Any]]:
    """Filter *raw_table* in the Deephaven engine then materialize to row dicts.

    Calls ``raw_table.where(predicate)`` before ``to_pandas``, so only matching
    rows cross into Python.  This is O(matching) per call vs. O(total) for a
    full snapshot — critical for append-only tick tables that grow unbounded
    over a hot session.

    Failed-source guard: a shared streaming source (notably ``bars_historical``)
    can transition to a permanent Deephaven engine "failed" state mid-session
    (observed against a live session). Once failed, every
    ``.where()`` against it raises ``deephaven.dherror.DHError`` for the rest of
    the process. Callers such as intraday P&L recompute this on a ~10s cycle, so
    left unguarded the error repeats forever. We pre-check ``raw_table.is_failed``
    and return ``[]`` rather than propagating the DHError -- callers must treat
    ``[]`` as "source unavailable / no rows": the same degrade-not-crash contract
    used by any downstream chart-rendering path built on this table. Any OTHER
    exception (e.g. a table that fails in the race window between this check and
    the ``where()`` call, or an unrelated construction error) is NOT swallowed and
    propagates as before.

    Parameters
    ----------
    raw_table:
        A live Deephaven Table.
    predicate:
        A Deephaven filter expression (e.g. ``"ContractId == 12345"``).

    Returns
    -------
    A list of row dicts for matching rows only, or ``[]`` if *raw_table* is
    already in a failed state.
    """
    if getattr(raw_table, "is_failed", False):
        logger.warning(
            "snapshot_rows_where: source table is in a failed state; "
            "returning no rows for predicate=%r", predicate,
        )
        return []

    from deephaven.pandas import to_pandas  # noqa: PLC0415 — engine-confined

    filtered = raw_table.where(predicate)
    df = to_pandas(filtered)
    return cast(list[dict[str, Any]], df.to_dict(orient="records"))
