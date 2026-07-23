"""Pure tests for the Pass B canonical schemas: cash_balance, price_tick, news_provider.

No JVM, no TWS: validates column names, types, nullability, and registration
in ibda.schema.ALL. Mirrors test_commission_news_errors_schemas.py.
"""
from __future__ import annotations

import pyarrow as pa

from ibda.schema import ALL as ALL_SCHEMAS
from ibda.schema import CASH_BALANCE, NEWS_PROVIDER, PRICE_TICK


# ---------------------------------------------------------------------------
# CASH_BALANCE schema
# ---------------------------------------------------------------------------


def test_cash_balance_name() -> None:
    assert CASH_BALANCE.name == "cash_balance"


def test_cash_balance_column_names() -> None:
    assert CASH_BALANCE.column_names == (
        "Account",
        "Currency",
        "CashBalance",
        "NetLiquidation",
        "ExchangeRate",
    )


def test_cash_balance_account_and_currency_are_string_not_null() -> None:
    for col_name in ("Account", "Currency"):
        col = next(c for c in CASH_BALANCE.columns if c.name == col_name)
        assert col.dtype.name == "STRING"
        assert col.nullable is False


def test_cash_balance_numeric_cols_are_float64() -> None:
    for col_name in ("CashBalance", "NetLiquidation", "ExchangeRate"):
        col = next(c for c in CASH_BALANCE.columns if c.name == col_name)
        assert col.dtype.name == "FLOAT64"


def test_cash_balance_arrow_schema_roundtrip() -> None:
    arrow = CASH_BALANCE.to_arrow_schema()
    assert arrow.field("Currency").type == pa.string()
    assert arrow.field("Currency").nullable is False
    assert arrow.field("CashBalance").type == pa.float64()


def test_cash_balance_registered_in_all() -> None:
    assert "cash_balance" in ALL_SCHEMAS
    assert ALL_SCHEMAS["cash_balance"] is CASH_BALANCE


def test_cash_balance_in_public_surface() -> None:
    import ibda.schema as schema_mod

    assert "CASH_BALANCE" in schema_mod.__all__


# ---------------------------------------------------------------------------
# PRICE_TICK schema
# ---------------------------------------------------------------------------


def test_price_tick_name() -> None:
    assert PRICE_TICK.name == "price_tick"


def test_price_tick_column_names() -> None:
    assert PRICE_TICK.column_names == (
        "ConId",
        "Sym",
        "TickType",
        "Price",
        "Timestamp",
    )


def test_price_tick_conid_is_int64_not_null() -> None:
    col = next(c for c in PRICE_TICK.columns if c.name == "ConId")
    assert col.dtype.name == "INT64"
    assert col.nullable is False


def test_price_tick_tick_type_is_string_not_null() -> None:
    col = next(c for c in PRICE_TICK.columns if c.name == "TickType")
    assert col.dtype.name == "STRING"
    assert col.nullable is False


def test_price_tick_timestamp_is_timestamp_not_null() -> None:
    col = next(c for c in PRICE_TICK.columns if c.name == "Timestamp")
    assert col.dtype.name == "TIMESTAMP_NS"
    assert col.nullable is False


def test_price_tick_price_is_float64() -> None:
    col = next(c for c in PRICE_TICK.columns if c.name == "Price")
    assert col.dtype.name == "FLOAT64"


def test_price_tick_arrow_schema_roundtrip() -> None:
    arrow = PRICE_TICK.to_arrow_schema()
    assert arrow.field("ConId").type == pa.int64()
    assert arrow.field("ConId").nullable is False
    assert arrow.field("TickType").type == pa.string()


def test_price_tick_registered_in_all() -> None:
    assert "price_tick" in ALL_SCHEMAS
    assert ALL_SCHEMAS["price_tick"] is PRICE_TICK


def test_price_tick_in_public_surface() -> None:
    import ibda.schema as schema_mod

    assert "PRICE_TICK" in schema_mod.__all__


# ---------------------------------------------------------------------------
# NEWS_PROVIDER schema
# ---------------------------------------------------------------------------


def test_news_provider_name() -> None:
    assert NEWS_PROVIDER.name == "news_provider"


def test_news_provider_column_names() -> None:
    assert NEWS_PROVIDER.column_names == ("Code", "Name")


def test_news_provider_columns_are_string_not_null() -> None:
    for col_name in ("Code", "Name"):
        col = next(c for c in NEWS_PROVIDER.columns if c.name == col_name)
        assert col.dtype.name == "STRING"
        assert col.nullable is False


def test_news_provider_arrow_schema_roundtrip() -> None:
    arrow = NEWS_PROVIDER.to_arrow_schema()
    assert arrow.field("Code").type == pa.string()
    assert arrow.field("Code").nullable is False


def test_news_provider_registered_in_all() -> None:
    assert "news_provider" in ALL_SCHEMAS
    assert ALL_SCHEMAS["news_provider"] is NEWS_PROVIDER


def test_news_provider_in_public_surface() -> None:
    import ibda.schema as schema_mod

    assert "NEWS_PROVIDER" in schema_mod.__all__
