"""Tests for the account_pnl and position_pnl canonical schemas.

Pure (no JVM, no TWS): validates column names, types, nullability, and registration
in ibda.schema.ALL.
"""
from __future__ import annotations

import pyarrow as pa
import pytest

from ibda.schema import ALL as ALL_SCHEMAS
from ibda.schema import ACCOUNT_PNL, POSITION_PNL


# ---------------------------------------------------------------------------
# ACCOUNT_PNL schema
# ---------------------------------------------------------------------------


def test_account_pnl_name() -> None:
    assert ACCOUNT_PNL.name == "account_pnl"


def test_account_pnl_column_names() -> None:
    assert ACCOUNT_PNL.column_names == (
        "Account",
        "Timestamp",
        "DailyPnl",
        "UnrealizedPnl",
        "RealizedPnl",
    )


def test_account_pnl_account_is_string_not_null() -> None:
    col = next(c for c in ACCOUNT_PNL.columns if c.name == "Account")
    assert col.dtype.name == "STRING"
    assert col.nullable is False


def test_account_pnl_timestamp_is_timestamp_nullable() -> None:
    col = next(c for c in ACCOUNT_PNL.columns if c.name == "Timestamp")
    assert col.dtype.name == "TIMESTAMP_NS"
    assert col.nullable is True


@pytest.mark.parametrize("col_name", ["DailyPnl", "UnrealizedPnl", "RealizedPnl"])
def test_account_pnl_pnl_cols_are_float64(col_name: str) -> None:
    col = next(c for c in ACCOUNT_PNL.columns if c.name == col_name)
    assert col.dtype.name == "FLOAT64"


def test_account_pnl_arrow_schema_roundtrip() -> None:
    arrow = ACCOUNT_PNL.to_arrow_schema()
    assert arrow.field("Account").type == pa.string()
    assert arrow.field("Account").nullable is False
    assert arrow.field("Timestamp").type == pa.timestamp("ns", tz="UTC")
    assert arrow.field("DailyPnl").type == pa.float64()


def test_account_pnl_registered_in_all() -> None:
    assert "account_pnl" in ALL_SCHEMAS
    assert ALL_SCHEMAS["account_pnl"] is ACCOUNT_PNL


# ---------------------------------------------------------------------------
# POSITION_PNL schema
# ---------------------------------------------------------------------------


def test_position_pnl_name() -> None:
    assert POSITION_PNL.name == "position_pnl"


def test_position_pnl_column_names() -> None:
    assert POSITION_PNL.column_names == (
        "Account",
        "ConId",
        "Sym",
        "DailyPnl",
        "UnrealizedPnl",
        "RealizedPnl",
        "MarketValue",
    )


def test_position_pnl_account_is_string_not_null() -> None:
    col = next(c for c in POSITION_PNL.columns if c.name == "Account")
    assert col.dtype.name == "STRING"
    assert col.nullable is False


def test_position_pnl_conid_is_int64_not_null() -> None:
    col = next(c for c in POSITION_PNL.columns if c.name == "ConId")
    assert col.dtype.name == "INT64"
    assert col.nullable is False


def test_position_pnl_sym_is_string_nullable() -> None:
    col = next(c for c in POSITION_PNL.columns if c.name == "Sym")
    assert col.dtype.name == "STRING"
    assert col.nullable is True


@pytest.mark.parametrize("col_name", ["DailyPnl", "UnrealizedPnl", "RealizedPnl", "MarketValue"])
def test_position_pnl_numeric_cols_are_float64(col_name: str) -> None:
    col = next(c for c in POSITION_PNL.columns if c.name == col_name)
    assert col.dtype.name == "FLOAT64"


def test_position_pnl_arrow_schema_roundtrip() -> None:
    arrow = POSITION_PNL.to_arrow_schema()
    assert arrow.field("Account").type == pa.string()
    assert arrow.field("ConId").type == pa.int64()
    assert arrow.field("Sym").type == pa.string()
    assert arrow.field("DailyPnl").type == pa.float64()
    assert arrow.field("MarketValue").type == pa.float64()


def test_position_pnl_registered_in_all() -> None:
    assert "position_pnl" in ALL_SCHEMAS
    assert ALL_SCHEMAS["position_pnl"] is POSITION_PNL


# ---------------------------------------------------------------------------
# Public surface — exported in __all__
# ---------------------------------------------------------------------------


def test_account_pnl_in_public_surface() -> None:
    import ibda.schema as schema_mod

    assert "ACCOUNT_PNL" in schema_mod.__all__


def test_position_pnl_in_public_surface() -> None:
    import ibda.schema as schema_mod

    assert "POSITION_PNL" in schema_mod.__all__
