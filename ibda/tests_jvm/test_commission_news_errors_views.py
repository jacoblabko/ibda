"""JVM test: apply_canonical_view for commission, news, and errors specs.

Mirrors the pattern in ``test_views.py``: constructs raw-shaped in-process
tables, applies ``apply_canonical_view``, snapshots through the port, and
asserts ``schema.validate`` passes and renamed columns carry the right values.

``news`` additionally exercises the join_table code path (join to
news_providers to resolve ProviderName) — the second real usage of that
mechanism after position_pnl (which is currently untested at the JVM level).

Run with:
    uv run pytest ibda/tests_jvm/test_commission_news_errors_views.py -q
"""
from __future__ import annotations

from typing import Any

import pytest

import ibda
from ibda.adapters.deephaven.views import apply_canonical_view
from ibda.adapters.ibkr.specs import CANONICAL_SPECS
from ibda.schema import COMMISSION, ERRORS, NEWS


# ---------------------------------------------------------------------------
# commission spec — orders_exec_commission_report → canonical commission
# ---------------------------------------------------------------------------

def _raw_commission_table() -> Any:
    from deephaven import new_table
    from deephaven.column import datetime_col, double_col, string_col
    from deephaven.time import to_j_instant

    t0 = to_j_instant("2026-06-10T15:00:05 UTC")
    return new_table([
        string_col("ExecId", ["exec001"]),
        datetime_col("ReceiveTime", [t0]),
        double_col("Commission", [1.25]),
        string_col("Currency", ["USD"]),
        double_col("RealizedPNL", [12.5]),
        double_col("Yield", [0.0]),
        string_col("YieldRedemptionDate", [""]),
    ])


def test_commission_view_conforms_to_schema() -> None:
    spec = CANONICAL_SPECS["commission"]
    viewed = apply_canonical_view(_raw_commission_table(), spec, COMMISSION)
    port = ibda.connect({"commission": viewed})
    arrow = port.table("commission").snapshot()
    COMMISSION.validate(arrow)
    assert arrow.num_rows == 1


def test_commission_view_renames_columns() -> None:
    """Commission sign: raw commissionReport.commission is a positive magnitude
    (1.25); the canonical `commission` table negates it to the unified
    negative=cost convention (matching execution.Commission) — see the
    ("commission", "Commission") entry in views._COMMISSION_SIGN_FLIP_COLS.
    """
    spec = CANONICAL_SPECS["commission"]
    viewed = apply_canonical_view(_raw_commission_table(), spec, COMMISSION)
    port = ibda.connect({"commission": viewed})
    row = port.table("commission").snapshot().to_pylist()[0]
    assert row["ExecId"] == "exec001"
    assert row["Commission"] == pytest.approx(-1.25)
    assert row["RealizedPnl"] == pytest.approx(12.5)
    assert row["Currency"] == "USD"


def test_commission_spec_dedupes_by_exec_id() -> None:
    """A correction resend for the same ExecId must collapse to one row (last wins)."""
    from deephaven import new_table
    from deephaven.column import datetime_col, double_col, string_col
    from deephaven.pandas import to_pandas
    from deephaven.time import to_j_instant

    t0 = to_j_instant("2026-06-10T15:00:05 UTC")
    t1 = to_j_instant("2026-06-10T15:05:00 UTC")
    raw = new_table([
        string_col("ExecId", ["exec001", "exec001"]),
        datetime_col("ReceiveTime", [t0, t1]),
        double_col("Commission", [1.25, 1.10]),
        string_col("Currency", ["USD", "USD"]),
        double_col("RealizedPNL", [12.5, 13.0]),
        double_col("Yield", [0.0, 0.0]),
        string_col("YieldRedemptionDate", ["", ""]),
    ])
    spec = CANONICAL_SPECS["commission"]
    viewed = apply_canonical_view(raw, spec, COMMISSION)
    df = to_pandas(viewed)
    assert len(df) == 1
    # Commission sign: negated to negative=cost by apply_canonical_view (see
    # test_commission_view_renames_columns above).
    assert df.iloc[0]["Commission"] == pytest.approx(-1.10)
    assert df.iloc[0]["RealizedPnl"] == pytest.approx(13.0)


# ---------------------------------------------------------------------------
# news spec — news_historical + news_providers join → canonical news
# ---------------------------------------------------------------------------

def _raw_news_historical_table() -> Any:
    from deephaven import new_table
    from deephaven.column import datetime_col, long_col, string_col
    from deephaven.time import to_j_instant

    t0 = to_j_instant("2026-06-10T13:00:00 UTC")
    return new_table([
        datetime_col("Timestamp", [t0]),
        string_col("Symbol", ["AAPL"]),
        long_col("ContractId", [12345]),
        string_col("ProviderCode", ["BRFG"]),
        string_col("ArticleId", ["BRFG$123abc"]),
        string_col("Headline", ["Apple announces new product"]),
    ])


def _raw_news_providers_table() -> Any:
    from deephaven import new_table
    from deephaven.column import string_col

    return new_table([
        string_col("Code", ["BRFG"]),
        string_col("Name", ["Briefing.com"]),
    ])


def test_news_view_conforms_to_schema() -> None:
    spec = CANONICAL_SPECS["news"]
    viewed = apply_canonical_view(
        _raw_news_historical_table(), spec, NEWS, join_raw=_raw_news_providers_table()
    )
    port = ibda.connect({"news": viewed})
    arrow = port.table("news").snapshot()
    NEWS.validate(arrow)
    assert arrow.num_rows == 1


def test_news_view_renames_and_resolves_provider_name() -> None:
    spec = CANONICAL_SPECS["news"]
    viewed = apply_canonical_view(
        _raw_news_historical_table(), spec, NEWS, join_raw=_raw_news_providers_table()
    )
    port = ibda.connect({"news": viewed})
    row = port.table("news").snapshot().to_pylist()[0]
    assert row["Sym"] == "AAPL"
    assert row["ConId"] == 12345
    assert row["ProviderCode"] == "BRFG"
    assert row["ProviderName"] == "Briefing.com"
    assert row["Headline"] == "Apple announces new product"


def test_news_view_unmatched_provider_code_yields_null_name() -> None:
    """A ProviderCode absent from news_providers must yield a null ProviderName (left join)."""
    from deephaven import new_table
    from deephaven.column import datetime_col, long_col, string_col
    from deephaven.time import to_j_instant

    t0 = to_j_instant("2026-06-10T13:00:00 UTC")
    raw = new_table([
        datetime_col("Timestamp", [t0]),
        string_col("Symbol", ["MSFT"]),
        long_col("ContractId", [67890]),
        string_col("ProviderCode", ["UNKNOWN"]),
        string_col("ArticleId", ["UNK$1"]),
        string_col("Headline", ["Some headline"]),
    ])
    spec = CANONICAL_SPECS["news"]
    viewed = apply_canonical_view(raw, spec, NEWS, join_raw=_raw_news_providers_table())
    port = ibda.connect({"news": viewed})
    row = port.table("news").snapshot().to_pylist()[0]
    assert row["ProviderName"] is None


def test_news_spec_dedupes_right_side_duplicate_provider_code() -> None:
    """A sibling of the accounts_positions dedupe regression: news_providers can
    re-emit a row for the same Code (see ibda/schema/news_provider.py's docstring
    and the news_provider spec's dedupe_keys) — the SAME raw table joined here as
    news's RIGHT side needs the identical guard, or natural_join errors ('more
    than one right row') whenever news_providers holds two rows for the same Code.
    """
    from deephaven import new_table
    from deephaven.column import string_col
    from deephaven.pandas import to_pandas

    duplicate_providers = new_table([
        string_col("Code", ["BRFG", "BRFG"]),
        string_col("Name", ["Briefing.com (old)", "Briefing.com"]),
    ])
    spec = CANONICAL_SPECS["news"]
    assert spec.join_dedupe_raw_keys == ("Code",)  # sanity: fix is in place

    viewed = apply_canonical_view(
        _raw_news_historical_table(), spec, NEWS, join_raw=duplicate_providers
    )
    df = to_pandas(viewed)
    assert len(df) == 1
    assert df.iloc[0]["ProviderName"] == "Briefing.com"


def test_news_join_without_dedupe_crashes_on_duplicate_right_row() -> None:
    """Negative-proof: strip join_dedupe_raw_keys and confirm natural_join errors."""
    import dataclasses

    from deephaven import new_table
    from deephaven.column import string_col

    duplicate_providers = new_table([
        string_col("Code", ["BRFG", "BRFG"]),
        string_col("Name", ["Briefing.com (old)", "Briefing.com"]),
    ])
    spec = CANONICAL_SPECS["news"]
    pre_fix_spec = dataclasses.replace(spec, join_dedupe_raw_keys=None)

    with pytest.raises(Exception, match="(?i)duplicate right key"):
        apply_canonical_view(
            _raw_news_historical_table(), pre_fix_spec, NEWS, join_raw=duplicate_providers
        )


def test_news_no_dedupe_retains_all_headlines() -> None:
    """news has no dedupe_keys — every historical headline row must be retained."""
    from deephaven import new_table
    from deephaven.column import datetime_col, long_col, string_col
    from deephaven.pandas import to_pandas
    from deephaven.time import to_j_instant

    t0 = to_j_instant("2026-06-10T13:00:00 UTC")
    t1 = to_j_instant("2026-06-10T14:00:00 UTC")
    raw = new_table([
        datetime_col("Timestamp", [t0, t1]),
        string_col("Symbol", ["AAPL", "AAPL"]),
        long_col("ContractId", [12345, 12345]),
        string_col("ProviderCode", ["BRFG", "BRFG"]),
        string_col("ArticleId", ["BRFG$1", "BRFG$2"]),
        string_col("Headline", ["Headline one", "Headline two"]),
    ])
    spec = CANONICAL_SPECS["news"]
    viewed = apply_canonical_view(raw, spec, NEWS, join_raw=_raw_news_providers_table())
    df = to_pandas(viewed)
    assert len(df) == 2


# ---------------------------------------------------------------------------
# errors spec — errors → canonical errors (Tier → Severity passthrough)
# ---------------------------------------------------------------------------

def _raw_errors_table() -> Any:
    from deephaven import new_table
    from deephaven.column import datetime_col, long_col, string_col
    from deephaven.time import to_j_instant

    t0 = to_j_instant("2026-06-10T18:23:41 UTC")
    return new_table([
        datetime_col("ReceiveTime", [t0]),
        long_col("RequestId", [-1]),
        long_col("ErrorCode", [2104]),
        string_col("ErrorDescription", ["Market data farm connection is OK"]),
        string_col("Error", ["Market data farm connection is OK:usfarm"]),
        string_col("Note", ["A notification that connection to the market data server is ok."]),
        string_col("Tier", ["INFORMATIONAL"]),
        string_col("Symbol", [""]),
        long_col("ContractId", [0]),
    ])


def test_errors_view_conforms_to_schema() -> None:
    spec = CANONICAL_SPECS["errors"]
    viewed = apply_canonical_view(_raw_errors_table(), spec, ERRORS)
    port = ibda.connect({"errors": viewed})
    arrow = port.table("errors").snapshot()
    ERRORS.validate(arrow)
    assert arrow.num_rows == 1


def test_errors_view_renames_tier_to_severity() -> None:
    spec = CANONICAL_SPECS["errors"]
    viewed = apply_canonical_view(_raw_errors_table(), spec, ERRORS)
    port = ibda.connect({"errors": viewed})
    row = port.table("errors").snapshot().to_pylist()[0]
    assert row["Code"] == 2104
    assert row["Severity"] == "INFORMATIONAL"
    assert row["Message"] == "Market data farm connection is OK"
    assert row["ReqId"] == -1


def test_errors_no_dedupe_retains_all_rows() -> None:
    """errors has no dedupe_keys — every callback row must be retained."""
    from deephaven import new_table
    from deephaven.column import datetime_col, long_col, string_col
    from deephaven.pandas import to_pandas
    from deephaven.time import to_j_instant

    t0 = to_j_instant("2026-06-10T18:23:41 UTC")
    t1 = to_j_instant("2026-06-10T18:23:42 UTC")
    raw = new_table([
        datetime_col("ReceiveTime", [t0, t1]),
        long_col("RequestId", [-1, -1]),
        long_col("ErrorCode", [2104, 502]),
        string_col("ErrorDescription", ["Market data farm connection is OK", "Couldn't connect to TWS"]),
        string_col("Error", ["...:usfarm", "Couldn't connect to TWS"]),
        string_col("Note", ["ok notice", ""]),
        string_col("Tier", ["INFORMATIONAL", "GENUINE_ERROR"]),
        string_col("Symbol", ["", ""]),
        long_col("ContractId", [0, 0]),
    ])
    spec = CANONICAL_SPECS["errors"]
    viewed = apply_canonical_view(raw, spec, ERRORS)
    df = to_pandas(viewed)
    assert len(df) == 2
    assert set(df["Severity"]) == {"INFORMATIONAL", "GENUINE_ERROR"}
