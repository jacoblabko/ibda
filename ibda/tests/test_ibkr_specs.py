"""Tests for ibda.adapters.ibkr.specs — pure mapping descriptor correctness.

Invariants:
- Every spec's renames.values() is a subset of its schema's column_names.
- Every spec references a known schema in ibda.schema.ALL.
- null_columns returns the complement of renames.values() w.r.t. schema columns.
- The union of (renames.values() + null_columns) == full schema column set
  (guarantees a complete, conformant canonical table can be produced).
"""
from __future__ import annotations

import pytest

from ibda.adapters.ibkr.specs import CANONICAL_SPECS, IbkrTableSpec, null_columns
from ibda.schema import ALL as ALL_SCHEMAS


# ---------------------------------------------------------------------------
# Parametrized over all five specs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("canonical_name", list(CANONICAL_SPECS))
def test_spec_references_known_schema(canonical_name: str) -> None:
    spec = CANONICAL_SPECS[canonical_name]
    assert spec.schema_name in ALL_SCHEMAS, (
        f"spec {canonical_name!r}: schema_name {spec.schema_name!r} not in ALL_SCHEMAS"
    )


@pytest.mark.parametrize("canonical_name", list(CANONICAL_SPECS))
def test_spec_renames_target_canonical_columns(canonical_name: str) -> None:
    spec = CANONICAL_SPECS[canonical_name]
    schema = ALL_SCHEMAS[spec.schema_name]
    col_names: set[str] = set(schema.column_names)
    bad = set(spec.renames.values()) - col_names
    assert not bad, (
        f"spec {canonical_name!r}: renames point to non-existent columns: {bad}"
    )


@pytest.mark.parametrize("canonical_name", list(CANONICAL_SPECS))
def test_null_columns_is_complement(canonical_name: str) -> None:
    spec = CANONICAL_SPECS[canonical_name]
    schema = ALL_SCHEMAS[spec.schema_name]
    covered = set(spec.renames.values())
    expected_nulls = {c.name for c in schema.columns} - covered
    assert set(null_columns(spec, schema)) == expected_nulls


@pytest.mark.parametrize("canonical_name", list(CANONICAL_SPECS))
def test_renames_plus_nulls_covers_full_schema(canonical_name: str) -> None:
    spec = CANONICAL_SPECS[canonical_name]
    schema = ALL_SCHEMAS[spec.schema_name]
    covered = set(spec.renames.values())
    nulls = set(null_columns(spec, schema))
    full = set(schema.column_names)
    assert covered | nulls == full, (
        f"spec {canonical_name!r}: covered={covered}, nulls={nulls}, full={full}"
    )


@pytest.mark.parametrize("canonical_name", list(CANONICAL_SPECS))
def test_renames_and_nulls_are_disjoint(canonical_name: str) -> None:
    spec = CANONICAL_SPECS[canonical_name]
    schema = ALL_SCHEMAS[spec.schema_name]
    covered = set(spec.renames.values())
    nulls = set(null_columns(spec, schema))
    overlap = covered & nulls
    assert not overlap, (
        f"spec {canonical_name!r}: column(s) appear in both renames and nulls: {overlap}"
    )


# ---------------------------------------------------------------------------
# Specific schema_name cross-checks (regression against the spec table)
# ---------------------------------------------------------------------------

def test_account_not_in_canonical_specs() -> None:
    """account is a pivot, not a rename; it must not appear in CANONICAL_SPECS."""
    assert "account" not in CANONICAL_SPECS


def test_position_spec_maps_to_position_schema() -> None:
    spec = CANONICAL_SPECS["position"]
    assert spec.schema_name == "position"
    assert spec.raw_table == "accounts_positions"
    # Core renames must be present
    assert spec.renames.get("Symbol") == "Sym"
    assert spec.renames.get("ContractId") == "ConId"
    assert spec.renames.get("Position") == "Qty"


def test_position_spec_maps_option_identity_columns_from_accounts_positions() -> None:
    """Right/Strike/LastTradeDateOrContractMonth/LocalSymbol/Multiplier map 1:1
    from accounts_positions (NOT via a join to `definition`/contracts_details,
    which is populated only per explicit request and is sparse for a full
    book — see ibda/schema/position.py's module docstring)."""
    spec = CANONICAL_SPECS["position"]
    assert spec.renames.get("Right") == "Right"
    assert spec.renames.get("Strike") == "Strike"
    assert (
        spec.renames.get("LastTradeDateOrContractMonth")
        == "LastTradeDateOrContractMonth"
    )
    assert spec.renames.get("LocalSymbol") == "LocalSymbol"
    assert spec.renames.get("Multiplier") == "Multiplier"


def test_position_spec_maps_currency_and_exchange() -> None:
    """Currency/Exchange also map 1:1 from accounts_positions — needed by
    subscribe.py's Contract reconstruction (reqMktData/reqPnLSingle)."""
    spec = CANONICAL_SPECS["position"]
    assert spec.renames.get("Currency") == "Currency"
    assert spec.renames.get("Exchange") == "Exchange"


def test_execution_spec_maps_to_execution_schema() -> None:
    spec = CANONICAL_SPECS["execution"]
    assert spec.schema_name == "execution"
    assert spec.raw_table == "orders_exec_details"
    assert spec.renames.get("Shares") == "Qty"
    assert spec.renames.get("ExecutionExchange") == "Venue"


def test_execution_spec_joins_commission() -> None:
    """execution must join orders_exec_commission_report to resolve Commission/RealizedPnl."""
    spec = CANONICAL_SPECS["execution"]
    assert spec.join_table == "orders_exec_commission_report"
    assert spec.join_on == "ExecId"
    assert "Commission" in spec.join_cols
    assert "RealizedPNL" in spec.join_cols
    assert spec.renames.get("Commission") == "Commission"
    assert spec.renames.get("RealizedPNL") == "RealizedPnl"


def test_execution_spec_dedupes_commission_join_source_on_exec_id() -> None:
    """A corrected commission report (same ExecId, resent) must not break the join —
    the right side is reduced to one (latest) row per ExecId before natural_join."""
    spec = CANONICAL_SPECS["execution"]
    assert spec.join_dedupe_raw_keys == ("ExecId",)


def test_execution_spec_covers_commission_and_realized_pnl_via_join() -> None:
    """Commission/RealizedPnl must NOT be in null_columns — they come from the join.

    Liquidity has no source anywhere in deephaven-ib and stays null-filled.
    """
    from ibda.schema import EXECUTION

    spec = CANONICAL_SPECS["execution"]
    nulls = set(null_columns(spec, EXECUTION))
    assert "Commission" not in nulls
    assert "RealizedPnl" not in nulls
    assert "Liquidity" in nulls


def test_execution_spec_maps_currency_from_orders_exec_details() -> None:
    """orders_exec_details.Currency maps 1:1 to canonical Currency (no join needed)."""
    spec = CANONICAL_SPECS["execution"]
    assert spec.renames.get("Currency") == "Currency"


def test_execution_spec_open_close_has_no_live_source() -> None:
    """OpenClose has no orders_exec_details/commission-report equivalent — it
    stays null-filled on the live path (only the Flex mapper populates it)."""
    from ibda.schema import EXECUTION

    spec = CANONICAL_SPECS["execution"]
    assert "OpenClose" not in spec.renames
    assert "OpenClose" in set(null_columns(spec, EXECUTION))


def test_order_spec_maps_to_order_schema() -> None:
    spec = CANONICAL_SPECS["order"]
    assert spec.schema_name == "order"
    assert spec.raw_table == "orders_submitted"
    assert spec.renames.get("Action") == "Side"
    assert spec.renames.get("TotalQuantity") == "Qty"
    assert spec.renames.get("FilledQuantity") == "FilledQty"
    assert spec.renames.get("LmtPrice") == "LimitPrice"
    assert spec.renames.get("ReceiveTime") == "Timestamp"


def test_order_has_no_null_columns() -> None:
    """orders_submitted covers every canonical order column (no null-fill)."""
    from ibda.schema import ORDER

    spec = CANONICAL_SPECS["order"]
    assert null_columns(spec, ORDER) == []


def test_definition_spec_maps_to_definition_schema() -> None:
    spec = CANONICAL_SPECS["definition"]
    assert spec.schema_name == "definition"
    assert spec.raw_table == "contracts_details"
    assert spec.renames.get("PrimaryExchange") == "Exchange"


def test_definition_spec_maps_option_identity_columns() -> None:
    """Right/Strike/LastTradeDateOrContractMonth/LocalSymbol map 1:1 from
    contracts_details — added so a downstream positions view can drop its residual
    raw-table dependency and read these columns from the canonical `definition`
    table instead."""
    from ibda.schema import DEFINITION

    spec = CANONICAL_SPECS["definition"]
    assert spec.renames.get("Right") == "Right"
    assert spec.renames.get("Strike") == "Strike"
    assert (
        spec.renames.get("LastTradeDateOrContractMonth")
        == "LastTradeDateOrContractMonth"
    )
    assert spec.renames.get("LocalSymbol") == "LocalSymbol"
    assert null_columns(spec, DEFINITION) == []


def test_quote_null_columns_include_last() -> None:
    """Last comes from ticks_price, not ticks_bid_ask; must be null-filled."""
    from ibda.schema import QUOTE

    spec = CANONICAL_SPECS["quote"]
    nulls = null_columns(spec, QUOTE)
    assert "Last" in nulls


def test_trade_spec_maps_to_trade_schema() -> None:
    from ibda.schema import TRADE

    spec = CANONICAL_SPECS["trade"]
    assert spec.schema_name == "trade"
    assert spec.raw_table == "ticks_trade"
    assert spec.renames.get("Symbol") == "Sym"
    assert spec.renames.get("Price") == "Price"
    assert spec.renames.get("Size") == "Size"
    assert null_columns(spec, TRADE) == []


def test_bar_has_no_null_columns() -> None:
    """All bar columns map 1-to-1; no null-fill expected."""
    from ibda.schema import BAR

    spec = CANONICAL_SPECS["bar"]
    nulls = null_columns(spec, BAR)
    assert nulls == []


# ---------------------------------------------------------------------------
# IbkrTableSpec dataclass behaviour
# ---------------------------------------------------------------------------

def test_spec_is_frozen() -> None:
    spec = IbkrTableSpec(raw_table="t", renames={"A": "B"}, schema_name="bar")
    try:
        spec.raw_table = "x"  # type: ignore[misc]
    except AttributeError:
        pass
    else:
        pytest.fail("IbkrTableSpec should be frozen (immutable)")


# ---------------------------------------------------------------------------
# position_pnl spec
# ---------------------------------------------------------------------------

def test_position_pnl_spec_in_canonical_specs() -> None:
    assert "position_pnl" in CANONICAL_SPECS


def test_position_pnl_spec_raw_table() -> None:
    spec = CANONICAL_SPECS["position_pnl"]
    assert spec.raw_table == "accounts_pnl_single"


def test_position_pnl_spec_schema_name() -> None:
    spec = CANONICAL_SPECS["position_pnl"]
    assert spec.schema_name == "position_pnl"


def test_position_pnl_spec_join_fields() -> None:
    """position_pnl must have join fields for accounts_positions."""
    spec = CANONICAL_SPECS["position_pnl"]
    assert spec.join_table == "accounts_positions"
    assert spec.join_on is not None
    assert "Symbol" in spec.join_cols
    assert spec.join_cast_left_col == "ConId"


def test_position_pnl_spec_dedupes_position_join_source_on_contract_id() -> None:
    """A prior regression: accounts_positions is append-only with overlapping startup
    sweeps that can carry more than one row per ContractId (see the "position" spec's
    dedupe_keys comment) — the SAME raw table joined here as the RIGHT side must
    be reduced to one (latest) row per ContractId before natural_join, or the
    join errors on a duplicate right-side match per left row."""
    spec = CANONICAL_SPECS["position_pnl"]
    assert spec.join_dedupe_raw_keys == ("ContractId",)


def test_position_pnl_spec_renames_cover_canonical_columns() -> None:
    """All renames must target valid canonical columns."""
    from ibda.schema import POSITION_PNL as POSITION_PNL_SCHEMA

    spec = CANONICAL_SPECS["position_pnl"]
    col_names: set[str] = set(POSITION_PNL_SCHEMA.column_names)
    bad = set(spec.renames.values()) - col_names
    assert not bad, f"position_pnl renames point to non-existent columns: {bad}"


def test_position_pnl_spec_covers_sym_via_join() -> None:
    """Sym must come via the join (Symbol in renames, not null-filled)."""
    from ibda.schema import POSITION_PNL as POSITION_PNL_SCHEMA

    spec = CANONICAL_SPECS["position_pnl"]
    schema = POSITION_PNL_SCHEMA
    assert "Symbol" in spec.renames, "Symbol (raw join col) must appear in renames"
    assert spec.renames["Symbol"] == "Sym"
    # Sym must NOT be in null_columns (it comes from the join)
    from ibda.adapters.ibkr.specs import null_columns
    nulls = null_columns(spec, schema)
    assert "Sym" not in nulls, "Sym should be covered by the join+rename, not null-filled"


# ---------------------------------------------------------------------------
# IbkrTableSpec join field defaults
# ---------------------------------------------------------------------------

def test_spec_join_fields_default_to_none() -> None:
    """Non-join specs must have join fields defaulting to None/empty."""
    spec = IbkrTableSpec(raw_table="t", renames={"A": "B"}, schema_name="bar")
    assert spec.join_table is None
    assert spec.join_on is None
    assert spec.join_cols == ()
    assert spec.join_cast_left_col is None
    assert spec.join_dedupe_raw_keys is None


# ---------------------------------------------------------------------------
# commission spec
# ---------------------------------------------------------------------------

def test_commission_spec_maps_to_commission_schema() -> None:
    spec = CANONICAL_SPECS["commission"]
    assert spec.schema_name == "commission"
    assert spec.raw_table == "orders_exec_commission_report"
    assert spec.renames.get("RealizedPNL") == "RealizedPnl"
    assert spec.renames.get("ReceiveTime") == "Timestamp"


def test_commission_spec_dedupes_on_exec_id() -> None:
    """commission is current-state-per-ExecId (correction resends), not an event log."""
    spec = CANONICAL_SPECS["commission"]
    assert spec.dedupe_keys == ["ExecId"]


def test_commission_has_no_null_columns() -> None:
    """orders_exec_commission_report covers every canonical commission column."""
    from ibda.schema import COMMISSION

    spec = CANONICAL_SPECS["commission"]
    assert null_columns(spec, COMMISSION) == []


# ---------------------------------------------------------------------------
# news spec
# ---------------------------------------------------------------------------

def test_news_spec_maps_to_news_schema() -> None:
    spec = CANONICAL_SPECS["news"]
    assert spec.schema_name == "news"
    assert spec.raw_table == "news_historical"
    assert spec.renames.get("Symbol") == "Sym"
    assert spec.renames.get("ContractId") == "ConId"


def test_news_spec_join_fields() -> None:
    """news must join news_providers to resolve ProviderName from ProviderCode."""
    spec = CANONICAL_SPECS["news"]
    assert spec.join_table == "news_providers"
    assert spec.join_on == "ProviderCode=Code"
    assert "Name" in spec.join_cols
    assert spec.renames.get("Name") == "ProviderName"


def test_news_spec_dedupes_provider_join_source_on_code() -> None:
    """A sibling of the prior accounts_positions regression above: news_providers
    can re-emit a row for the same Code (see ibda/schema/news_provider.py's
    docstring / the news_provider spec's dedupe_keys) — the SAME raw table joined
    here as news's RIGHT side must be reduced to one (latest) row per Code before
    natural_join, or the join errors on a duplicate right-side match per left row."""
    spec = CANONICAL_SPECS["news"]
    assert spec.join_dedupe_raw_keys == ("Code",)


def test_news_spec_has_no_dedupe_keys() -> None:
    """news is an append-only event log — no last_by dedupe."""
    spec = CANONICAL_SPECS["news"]
    assert spec.dedupe_keys is None


def test_news_has_no_null_columns() -> None:
    """news_historical + news_providers join covers every canonical news column."""
    from ibda.schema import NEWS

    spec = CANONICAL_SPECS["news"]
    assert null_columns(spec, NEWS) == []


# ---------------------------------------------------------------------------
# errors spec
# ---------------------------------------------------------------------------

def test_errors_spec_maps_to_errors_schema() -> None:
    spec = CANONICAL_SPECS["errors"]
    assert spec.schema_name == "errors"
    assert spec.raw_table == "errors"
    assert spec.renames.get("ErrorCode") == "Code"
    assert spec.renames.get("ErrorDescription") == "Message"
    assert spec.renames.get("Tier") == "Severity"


def test_errors_spec_has_no_dedupe_keys() -> None:
    """errors is an append-only event log — no last_by dedupe."""
    spec = CANONICAL_SPECS["errors"]
    assert spec.dedupe_keys is None


def test_errors_has_no_null_columns() -> None:
    """The raw errors table (joined with requests upstream) covers every canonical column."""
    from ibda.schema import ERRORS

    spec = CANONICAL_SPECS["errors"]
    assert null_columns(spec, ERRORS) == []


# ---------------------------------------------------------------------------
# account_pnl spec (Pass B #1 gap-closer: now a plain CANONICAL_SPECS entry
# instead of a table_from_rows snapshot built by hand in connect_live)
# ---------------------------------------------------------------------------

def test_account_pnl_spec_in_canonical_specs() -> None:
    assert "account_pnl" in CANONICAL_SPECS


def test_account_pnl_spec_maps_to_account_pnl_schema() -> None:
    spec = CANONICAL_SPECS["account_pnl"]
    assert spec.schema_name == "account_pnl"
    assert spec.raw_table == "accounts_pnl"
    assert spec.renames.get("ReceiveTime") == "Timestamp"
    assert spec.renames.get("DailyPnl") == "DailyPnl"
    assert spec.renames.get("UnrealizedPnl") == "UnrealizedPnl"
    assert spec.renames.get("RealizedPnl") == "RealizedPnl"


def test_account_pnl_spec_dedupes_by_account() -> None:
    """accounts_pnl is append-only; current state per account requires last_by."""
    spec = CANONICAL_SPECS["account_pnl"]
    assert spec.dedupe_keys == ["Account"]


def test_account_pnl_has_no_null_columns() -> None:
    from ibda.schema import ACCOUNT_PNL

    spec = CANONICAL_SPECS["account_pnl"]
    assert null_columns(spec, ACCOUNT_PNL) == []


# ---------------------------------------------------------------------------
# price_tick spec (Pass B #2 gap-closer)
# ---------------------------------------------------------------------------

def test_price_tick_spec_in_canonical_specs() -> None:
    assert "price_tick" in CANONICAL_SPECS


def test_price_tick_spec_maps_to_price_tick_schema() -> None:
    spec = CANONICAL_SPECS["price_tick"]
    assert spec.schema_name == "price_tick"
    assert spec.raw_table == "ticks_price"
    assert spec.renames.get("ContractId") == "ConId"
    assert spec.renames.get("Symbol") == "Sym"
    assert spec.renames.get("TickType") == "TickType"
    assert spec.renames.get("Price") == "Price"
    assert spec.renames.get("ReceiveTime") == "Timestamp"


def test_price_tick_spec_has_no_dedupe_keys() -> None:
    """price_tick is a streaming event log — every TickType/price update is kept."""
    spec = CANONICAL_SPECS["price_tick"]
    assert spec.dedupe_keys is None


def test_price_tick_has_no_null_columns() -> None:
    """ticks_price covers every canonical price_tick column directly (no join needed)."""
    from ibda.schema import PRICE_TICK

    spec = CANONICAL_SPECS["price_tick"]
    assert null_columns(spec, PRICE_TICK) == []


# ---------------------------------------------------------------------------
# news_provider spec (Pass B #4 gap-closer)
# ---------------------------------------------------------------------------

def test_news_provider_spec_in_canonical_specs() -> None:
    assert "news_provider" in CANONICAL_SPECS


def test_news_provider_spec_maps_to_news_provider_schema() -> None:
    spec = CANONICAL_SPECS["news_provider"]
    assert spec.schema_name == "news_provider"
    assert spec.raw_table == "news_providers"
    assert spec.renames.get("Code") == "Code"
    assert spec.renames.get("Name") == "Name"


def test_news_provider_spec_dedupes_by_code() -> None:
    spec = CANONICAL_SPECS["news_provider"]
    assert spec.dedupe_keys == ["Code"]


def test_news_provider_has_no_null_columns() -> None:
    from ibda.schema import NEWS_PROVIDER

    spec = CANONICAL_SPECS["news_provider"]
    assert null_columns(spec, NEWS_PROVIDER) == []


# ---------------------------------------------------------------------------
# cash_balance: NOT in CANONICAL_SPECS (same rationale as `account` — it is a
# key-value pivot over accounts_overview, built by
# ibda.adapters.deephaven.views.build_cash_balance_view, not a rename spec).
# ---------------------------------------------------------------------------

def test_cash_balance_not_in_canonical_specs() -> None:
    assert "cash_balance" not in CANONICAL_SPECS
