"""Non-JVM tests for load_flex_xml and load_flex_file.

Tests that do NOT require a Deephaven engine live here.  They verify the
error-handling contract (malformed XML, in_progress status, fail status).

JVM-requiring tests (happy-path row counts via port.snapshot()) live in
ibda/tests_jvm/test_flex_loader.py.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ibda.adapters.ibkr.flex.parse import parse_statement
from ibda.errors import FlexParseError

_FIXTURE = Path(__file__).parent / "fixtures" / "flex" / "report_full.xml"
_MULTI_ACCOUNT_FIXTURE = (
    Path(__file__).parent / "fixtures" / "flex" / "report_multi_account.xml"
)

# ---------------------------------------------------------------------------
# Minimal XML snippets
# ---------------------------------------------------------------------------

_MALFORMED_XML = "<<this is not XML>>"

_IN_PROGRESS_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <FlexStatementResponse timestamp="20260602;104500 EST">
    <Status>Warn</Status>
    <ErrorCode>1019</ErrorCode>
    <ErrorMessage>Statement generation in progress. Please try again shortly.</ErrorMessage>
    </FlexStatementResponse>
""")

_FAIL_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <FlexStatementResponse timestamp="20260602;104500 EST">
    <Status>Fail</Status>
    <ErrorCode>1003</ErrorCode>
    <ErrorMessage>IP restriction</ErrorMessage>
    </FlexStatementResponse>
""")


# ---------------------------------------------------------------------------
# load_flex_xml error-path tests (no engine needed)
# ---------------------------------------------------------------------------

class TestLoadFlexXmlErrors:
    """load_flex_xml raises FlexParseError on non-ok statuses."""

    def test_malformed_xml_raises_flex_parse_error(self) -> None:
        import ibda

        with pytest.raises(FlexParseError, match="malformed"):
            ibda.load_flex_xml(_MALFORMED_XML)

    def test_in_progress_raises_with_distinct_message(self) -> None:
        import ibda

        with pytest.raises(FlexParseError, match="in_progress"):
            ibda.load_flex_xml(_IN_PROGRESS_XML)

    def test_in_progress_message_includes_retry_hint(self) -> None:
        import ibda

        with pytest.raises(FlexParseError, match="[Rr]etry"):
            ibda.load_flex_xml(_IN_PROGRESS_XML)

    def test_fail_status_raises_flex_parse_error(self) -> None:
        import ibda

        with pytest.raises(FlexParseError, match="fail"):
            ibda.load_flex_xml(_FAIL_XML)

    def test_fail_status_message_included_in_error(self) -> None:
        import ibda

        with pytest.raises(FlexParseError, match="IP restriction"):
            ibda.load_flex_xml(_FAIL_XML)


# ---------------------------------------------------------------------------
# load_flex_file error-path tests (no engine needed)
# ---------------------------------------------------------------------------

class TestLoadFlexFileErrors:
    """load_flex_file delegates to load_flex_xml; same error contract."""

    def test_missing_file_raises_oserror(self, tmp_path: Path) -> None:
        import ibda

        missing = tmp_path / "does_not_exist.xml"
        with pytest.raises(OSError):
            ibda.load_flex_file(str(missing))

    def test_malformed_file_raises_flex_parse_error(self, tmp_path: Path) -> None:
        import ibda

        bad_file = tmp_path / "bad.xml"
        bad_file.write_text(_MALFORMED_XML, encoding="utf-8")
        with pytest.raises(FlexParseError):
            ibda.load_flex_file(str(bad_file))

    def test_in_progress_file_raises_flex_parse_error(self, tmp_path: Path) -> None:
        import ibda

        xml_file = tmp_path / "inprogress.xml"
        xml_file.write_text(_IN_PROGRESS_XML, encoding="utf-8")
        with pytest.raises(FlexParseError, match="in_progress"):
            ibda.load_flex_file(str(xml_file))


# ---------------------------------------------------------------------------
# Public-surface tests (no engine)
# ---------------------------------------------------------------------------

class TestParseStatementMultiStatement:
    """parse_statement must concatenate ALL <FlexStatement> blocks, not just #1.

    A Flex response is <FlexStatements count="N"> and can legitimately hold
    multiple <FlexStatement> blocks (per-account / advisor / family /
    non-consolidated queries). Statements 2..N used to be silently dropped
    because parse_statement located only the first block with root.find(...).
    """

    def test_trades_sum_across_both_statements(self) -> None:
        result = parse_statement(_MULTI_ACCOUNT_FIXTURE.read_text())

        assert result["status"] == "ok"
        # Statement 1 has 2 trades (AAPL x2), statement 2 has 1 trade (MSFT).
        assert len(result["sections"]["trades"]) == 3

    def test_both_account_ids_present_in_trades(self) -> None:
        result = parse_statement(_MULTI_ACCOUNT_FIXTURE.read_text())

        accounts = {t["account"] for t in result["sections"]["trades"]}
        assert accounts == {"U1111111", "U2222222"}

    def test_nav_rows_from_both_statements_present(self) -> None:
        result = parse_statement(_MULTI_ACCOUNT_FIXTURE.read_text())

        nav_accounts = {row["account"] for row in result["sections"]["nav"]}
        assert nav_accounts == {"U1111111", "U2222222"}
        # 2 report dates per statement x 2 statements = 4 NAV rows total.
        assert len(result["sections"]["nav"]) == 4

    def test_cash_rows_from_both_statements_present(self) -> None:
        result = parse_statement(_MULTI_ACCOUNT_FIXTURE.read_text())

        cash_accounts = {row["account"] for row in result["sections"]["cash"]}
        assert cash_accounts == {"U1111111", "U2222222"}
        assert len(result["sections"]["cash"]) == 2

    def test_fifo_by_symbol_concatenated_across_statements(self) -> None:
        result = parse_statement(_MULTI_ACCOUNT_FIXTURE.read_text())

        symbols = {row["symbol"] for row in result["sections"]["pnl"]["fifo_by_symbol"]}
        assert symbols == {"AAPL", "MSFT"}

    def test_change_in_nav_is_first_statement_for_back_compat(self) -> None:
        """change_in_nav stays a single dict (first statement wins) — non-breaking."""
        result = parse_statement(_MULTI_ACCOUNT_FIXTURE.read_text())

        change_in_nav = result["sections"]["pnl"]["change_in_nav"]
        assert isinstance(change_in_nav, dict)
        assert change_in_nav["account"] == "U1111111"
        assert change_in_nav["ending_value"] == 1004580.0

    def test_change_in_nav_by_account_has_one_entry_per_statement(self) -> None:
        """change_in_nav_by_account is the lossless list — every statement's record."""
        result = parse_statement(_MULTI_ACCOUNT_FIXTURE.read_text())

        by_account = result["sections"]["pnl"]["change_in_nav_by_account"]
        assert len(by_account) == 2
        accounts = {row["account"] for row in by_account}
        assert accounts == {"U1111111", "U2222222"}
        by_id = {row["account"]: row for row in by_account}
        assert by_id["U1111111"]["ending_value"] == 1004580.0
        assert by_id["U2222222"]["ending_value"] == 500250.0


class TestParseStatementSingleStatementRegression:
    """A single-statement (count="1") report must parse identically to before."""

    def test_full_fixture_row_counts_unchanged(self) -> None:
        result = parse_statement(_FIXTURE.read_text())

        assert result["status"] == "ok"
        sections = result["sections"]
        assert len(sections["trades"]) == 2
        assert len(sections["cash"]) == 3
        # 1 CorporateAction (Split) + 1 Transfer (kind="transfer") = 2 rows.
        assert len(sections["corporate_actions"]) == 2
        assert len(sections["nav"]) == 6
        assert sections["pnl"]["change_in_nav"]["ending_value"] == 1004580.0
        assert sections["pnl"]["fifo_by_symbol"][0]["symbol"] == "AAPL"

    def test_change_in_nav_carries_account_single_statement(self) -> None:
        """A single-statement report's change_in_nav dict now also carries account."""
        result = parse_statement(_FIXTURE.read_text())

        change_in_nav = result["sections"]["pnl"]["change_in_nav"]
        assert isinstance(change_in_nav, dict)
        assert change_in_nav["account"] == "U0000000"

    def test_change_in_nav_by_account_single_entry_for_single_statement(self) -> None:
        """For the common single-statement case, change_in_nav_by_account == [that dict]."""
        result = parse_statement(_FIXTURE.read_text())

        change_in_nav = result["sections"]["pnl"]["change_in_nav"]
        by_account = result["sections"]["pnl"]["change_in_nav_by_account"]
        assert by_account == [change_in_nav]


class TestPublicSurface:
    """load_flex_xml and load_flex_file appear in ibda.__all__."""

    def test_load_flex_xml_in_all(self) -> None:
        import ibda

        assert "load_flex_xml" in ibda.__all__

    def test_load_flex_file_in_all(self) -> None:
        import ibda

        assert "load_flex_file" in ibda.__all__

    def test_import_ibda_does_not_import_deephaven(self) -> None:
        """Importing ibda must not trigger a deephaven_server import.

        Runs in a FRESH subprocess (Fix I) rather than checking
        ``sys.modules`` in the current pytest process: this test's own
        process may already have run other tests that boot an in-process
        Deephaven Server (any suite whose conftest constructs one), which
        would put ``deephaven_server`` in ``sys.modules`` for reasons
        unrelated to ``import ibda`` and make this test fail on collection
        order alone. A clean subprocess is immune to that pollution and
        keeps the intent unchanged: importing ``ibda`` alone must not
        eagerly import ``deephaven_server``.
        """
        check_src = (
            "import ibda\n"
            "import sys\n"
            "assert 'deephaven_server' not in sys.modules, ("
            "'deephaven_server was imported at ibda module level; '"
            "'the lazy-import contract is broken.')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", check_src],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"subprocess check failed (rc={result.returncode}):\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
