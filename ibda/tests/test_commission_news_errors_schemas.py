"""Tests for the commission, news, and errors canonical schemas.

Pure (no JVM, no TWS): validates column names, types, nullability, and
registration in ibda.schema.ALL.
"""
from __future__ import annotations

import pyarrow as pa
import pytest

from ibda.schema import ALL as ALL_SCHEMAS
from ibda.schema import COMMISSION, ERRORS, NEWS


# ---------------------------------------------------------------------------
# COMMISSION schema
# ---------------------------------------------------------------------------


def test_commission_name() -> None:
    assert COMMISSION.name == "commission"


def test_commission_column_names() -> None:
    assert COMMISSION.column_names == (
        "ExecId",
        "Timestamp",
        "Commission",
        "Currency",
        "RealizedPnl",
        "Yield",
        "YieldRedemptionDate",
    )


def test_commission_exec_id_is_string_not_null() -> None:
    col = next(c for c in COMMISSION.columns if c.name == "ExecId")
    assert col.dtype.name == "STRING"
    assert col.nullable is False


def test_commission_timestamp_is_timestamp_not_null() -> None:
    col = next(c for c in COMMISSION.columns if c.name == "Timestamp")
    assert col.dtype.name == "TIMESTAMP_NS"
    assert col.nullable is False


@pytest.mark.parametrize("col_name", ["Commission", "RealizedPnl", "Yield"])
def test_commission_numeric_cols_are_float64(col_name: str) -> None:
    col = next(c for c in COMMISSION.columns if c.name == col_name)
    assert col.dtype.name == "FLOAT64"


def test_commission_arrow_schema_roundtrip() -> None:
    arrow = COMMISSION.to_arrow_schema()
    assert arrow.field("ExecId").type == pa.string()
    assert arrow.field("ExecId").nullable is False
    assert arrow.field("Timestamp").type == pa.timestamp("ns", tz="UTC")
    assert arrow.field("Commission").type == pa.float64()


def test_commission_registered_in_all() -> None:
    assert "commission" in ALL_SCHEMAS
    assert ALL_SCHEMAS["commission"] is COMMISSION


def test_commission_in_public_surface() -> None:
    import ibda.schema as schema_mod

    assert "COMMISSION" in schema_mod.__all__


# ---------------------------------------------------------------------------
# NEWS schema
# ---------------------------------------------------------------------------


def test_news_name() -> None:
    assert NEWS.name == "news"


def test_news_column_names() -> None:
    assert NEWS.column_names == (
        "Timestamp",
        "Sym",
        "ConId",
        "ProviderCode",
        "ProviderName",
        "ArticleId",
        "Headline",
    )


def test_news_timestamp_is_timestamp_not_null() -> None:
    col = next(c for c in NEWS.columns if c.name == "Timestamp")
    assert col.dtype.name == "TIMESTAMP_NS"
    assert col.nullable is False


def test_news_headline_is_string_not_null() -> None:
    col = next(c for c in NEWS.columns if c.name == "Headline")
    assert col.dtype.name == "STRING"
    assert col.nullable is False


def test_news_conid_is_int64_nullable() -> None:
    col = next(c for c in NEWS.columns if c.name == "ConId")
    assert col.dtype.name == "INT64"
    assert col.nullable is True


def test_news_arrow_schema_roundtrip() -> None:
    arrow = NEWS.to_arrow_schema()
    assert arrow.field("Headline").type == pa.string()
    assert arrow.field("Headline").nullable is False
    assert arrow.field("ConId").type == pa.int64()


def test_news_registered_in_all() -> None:
    assert "news" in ALL_SCHEMAS
    assert ALL_SCHEMAS["news"] is NEWS


def test_news_in_public_surface() -> None:
    import ibda.schema as schema_mod

    assert "NEWS" in schema_mod.__all__


# ---------------------------------------------------------------------------
# ERRORS schema
# ---------------------------------------------------------------------------


def test_errors_name() -> None:
    assert ERRORS.name == "errors"


def test_errors_column_names() -> None:
    assert ERRORS.column_names == (
        "Timestamp",
        "ReqId",
        "Code",
        "Message",
        "Detail",
        "Note",
        "Severity",
        "Sym",
        "ConId",
    )


def test_errors_timestamp_is_timestamp_not_null() -> None:
    col = next(c for c in ERRORS.columns if c.name == "Timestamp")
    assert col.dtype.name == "TIMESTAMP_NS"
    assert col.nullable is False


def test_errors_code_is_int64_not_null() -> None:
    col = next(c for c in ERRORS.columns if c.name == "Code")
    assert col.dtype.name == "INT64"
    assert col.nullable is False


def test_errors_severity_is_string_not_null() -> None:
    col = next(c for c in ERRORS.columns if c.name == "Severity")
    assert col.dtype.name == "STRING"
    assert col.nullable is False


def test_errors_reqid_is_int64_nullable() -> None:
    col = next(c for c in ERRORS.columns if c.name == "ReqId")
    assert col.dtype.name == "INT64"
    assert col.nullable is True


def test_errors_arrow_schema_roundtrip() -> None:
    arrow = ERRORS.to_arrow_schema()
    assert arrow.field("Code").type == pa.int64()
    assert arrow.field("Code").nullable is False
    assert arrow.field("Severity").type == pa.string()


def test_errors_registered_in_all() -> None:
    assert "errors" in ALL_SCHEMAS
    assert ALL_SCHEMAS["errors"] is ERRORS


def test_errors_in_public_surface() -> None:
    import ibda.schema as schema_mod

    assert "ERRORS" in schema_mod.__all__
