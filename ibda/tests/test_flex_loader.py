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

from ibda.errors import FlexParseError

_FIXTURE = Path(__file__).parent / "fixtures" / "flex" / "report_full.xml"

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
