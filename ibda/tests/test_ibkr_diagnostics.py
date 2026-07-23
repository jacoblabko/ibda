"""Unit tests for ibda.adapters.ibkr.diagnostics.

Covers:
- Every ErrorTier with representative codes from the rule doc.
- In-band (2100-2169) "broken"/"lost" message override.
- Farm parsing: OK / broken / inactive, with/without farm suffix, non-farm → None.
- is_fatal / is_connectivity_event / is_connection_lost / is_connection_restored.
- IbDiagnostic.from_callback factory.

No live TWS; no JVM; pure stdlib + pytest.
"""

from __future__ import annotations

import pytest

from ibda.adapters.ibkr.diagnostics import (
    ErrorTier,
    FarmStatus,
    IbDiagnostic,
    classify_error,
    classify_farm,
    is_connection_lost,
    is_connection_restored,
    is_connectivity_event,
    is_fatal,
)


# ---------------------------------------------------------------------------
# classify_error — tier by code
# ---------------------------------------------------------------------------


class TestClassifyErrorInformational:
    """2100–2169 band (excluding degraded codes) + restore codes are INFORMATIONAL."""

    @pytest.mark.parametrize("code", [
        2100,  # band start
        2104,  # Market data farm connection is OK
        2106,  # HMDS data farm connection is OK
        2107,  # HMDS data farm connection is inactive (demand-only)
        2108,  # Market data farm connection is inactive
        2113,  # Bond order size advisory
        2119,  # Market data farm is connected
        2158,  # Sec-def data farm connection is OK
        2169,  # band end (inclusive)
        1101,  # Restore — subscriptions lost
        1102,  # Restore — subscriptions preserved
    ])
    def test_informational_code(self, code: int) -> None:
        assert classify_error(code) is ErrorTier.INFORMATIONAL

    def test_message_ok_still_informational(self) -> None:
        assert classify_error(2104, "Market data farm connection is OK:usfarm") is ErrorTier.INFORMATIONAL

    def test_no_message_still_informational_for_band(self) -> None:
        for code in range(2100, 2170):
            if code in {2103, 2105, 2110, 2157}:
                continue  # these are CONNECTIVITY_DEGRADED by code
            assert classify_error(code) is ErrorTier.INFORMATIONAL, (
                f"code {code} should be INFORMATIONAL (no message)"
            )


class TestClassifyErrorConnectivityDegraded:
    """Hard-coded degraded codes + in-band message override."""

    @pytest.mark.parametrize("code", [
        2103,   # Market data farm connection is broken
        2105,   # Historical data farm connection is broken
        2110,   # TWS↔server broken
        2157,   # Sec-def data farm connection is broken
        1100,   # Connectivity between IB and TWS lost
    ])
    def test_degraded_by_code(self, code: int) -> None:
        assert classify_error(code) is ErrorTier.CONNECTIVITY_DEGRADED

    @pytest.mark.parametrize("code,message", [
        (2108, "Market data farm connection is broken:usfarm"),
        (2108, "Market data farm connection is BROKEN:usfarm"),
        (2100, "Something lost connection"),
        (2150, "Connection lost to farm:ushmds"),
    ])
    def test_in_band_message_override_broken_or_lost(self, code: int, message: str) -> None:
        """Any in-band code whose message contains 'broken' or 'lost' → CONNECTIVITY_DEGRADED."""
        assert classify_error(code, message) is ErrorTier.CONNECTIVITY_DEGRADED

    def test_non_in_band_code_with_broken_message_stays_genuine(self) -> None:
        """The message override only applies to the 2100–2169 band."""
        # Code 502 with the word "broken" in the message is still GENUINE_ERROR.
        assert classify_error(502, "connection broken") is ErrorTier.GENUINE_ERROR


class TestClassifyErrorMarketDataWarning:
    """10000-band market-data subscription codes."""

    @pytest.mark.parametrize("code", [10167, 10168, 10197, 10089, 10091])
    def test_market_data_warning(self, code: int) -> None:
        assert classify_error(code) is ErrorTier.MARKET_DATA_WARNING


class TestClassifyErrorInformationalOutOfBand:
    """Individually-classified informational codes outside the 2100-2169 band --
    these would otherwise fall through to the GENUINE_ERROR default. Classify by
    errorCode, never by ibapi's raw ERROR: log severity."""

    def test_2176_fractional_share_warning_is_informational(self) -> None:
        assert classify_error(2176) is ErrorTier.INFORMATIONAL


class TestClassifyErrorGenuineError:
    """Low codes and anything unrecognised default to GENUINE_ERROR."""

    @pytest.mark.parametrize("code", [
        502,    # can't connect to TWS
        504,    # not connected
        326,    # duplicate client id
        200,    # no security definition
        162,    # historical data error
        1,      # unknown low code
        9999,   # random mid-range
        99999,  # high but not in 10000-band warning set
    ])
    def test_genuine_error(self, code: int) -> None:
        assert classify_error(code) is ErrorTier.GENUINE_ERROR


# ---------------------------------------------------------------------------
# is_fatal
# ---------------------------------------------------------------------------


class TestIsFatal:
    def test_genuine_error_is_fatal(self) -> None:
        assert is_fatal(502) is True

    def test_informational_not_fatal(self) -> None:
        assert is_fatal(2104) is False

    def test_connectivity_degraded_not_fatal(self) -> None:
        assert is_fatal(1100) is False

    def test_market_data_warning_not_fatal(self) -> None:
        assert is_fatal(10167) is False

    def test_message_forwarded_to_classify(self) -> None:
        # In-band code with no message → INFORMATIONAL → not fatal.
        assert is_fatal(2108) is False
        # Same code with degradation text → CONNECTIVITY_DEGRADED → not fatal.
        assert is_fatal(2108, "Market data farm connection is broken:usfarm") is False


# ---------------------------------------------------------------------------
# Connectivity predicates
# ---------------------------------------------------------------------------


class TestConnectivityPredicates:
    def test_is_connectivity_event_includes_lost(self) -> None:
        assert is_connectivity_event(1100) is True

    def test_is_connectivity_event_includes_restore(self) -> None:
        assert is_connectivity_event(1101) is True
        assert is_connectivity_event(1102) is True

    def test_is_connectivity_event_includes_degraded_codes(self) -> None:
        for code in (2103, 2105, 2110, 2157):
            assert is_connectivity_event(code) is True, f"code {code} should be connectivity event"

    def test_is_connectivity_event_false_for_informational(self) -> None:
        assert is_connectivity_event(2104) is False

    def test_is_connectivity_event_false_for_genuine(self) -> None:
        assert is_connectivity_event(502) is False

    def test_is_connection_lost_only_1100(self) -> None:
        assert is_connection_lost(1100) is True
        assert is_connection_lost(1101) is False
        assert is_connection_lost(1102) is False
        assert is_connection_lost(2103) is False

    def test_is_connection_restored_1101_and_1102(self) -> None:
        assert is_connection_restored(1101) is True
        assert is_connection_restored(1102) is True
        assert is_connection_restored(1100) is False
        assert is_connection_restored(2104) is False


# ---------------------------------------------------------------------------
# classify_farm
# ---------------------------------------------------------------------------


class TestClassifyFarm:
    def test_market_data_ok_with_farm(self) -> None:
        result = classify_farm("Market data farm connection is OK:usfarm")
        assert result == FarmStatus(farm="usfarm", kind="market_data", is_ok=True)

    def test_market_data_ok_with_hfarm(self) -> None:
        # A real farm-status message carries a farm-name suffix, e.g. "hfarm".
        result = classify_farm("Market data farm connection is OK:hfarm")
        assert result == FarmStatus(farm="hfarm", kind="market_data", is_ok=True)

    def test_historical_data_broken_with_farm(self) -> None:
        result = classify_farm("Historical data farm connection is broken:ushmds")
        assert result == FarmStatus(farm="ushmds", kind="historical_data", is_ok=False)

    def test_hmds_inactive_dot_suffix(self) -> None:
        # IB uses a dot separator in some messages: "...inactive but should be available upon demand.ushmds"
        result = classify_farm(
            "HMDS data farm connection is inactive but should be available upon demand.ushmds"
        )
        assert result is not None
        assert result.kind == "historical_data"
        assert result.is_ok is False
        assert result.farm == "ushmds"

    def test_market_data_broken_is_not_ok(self) -> None:
        result = classify_farm("Market data farm connection is broken:usfarm")
        assert result is not None
        assert result.is_ok is False
        assert result.farm == "usfarm"

    def test_market_data_ok_no_farm_suffix(self) -> None:
        result = classify_farm("Market data farm connection is OK")
        assert result is not None
        assert result.is_ok is True
        assert result.farm is None

    def test_sec_def_farm_classified_as_market_data(self) -> None:
        result = classify_farm("Sec-def data farm connection is OK:secdefnj")
        assert result is not None
        assert result.kind == "market_data"
        assert result.is_ok is True

    def test_non_farm_message_returns_none(self) -> None:
        assert classify_farm("[502] Could not connect to TWS") is None
        assert classify_farm("Order submitted successfully") is None
        assert classify_farm("") is None
        assert classify_farm("2104 connection is OK") is None

    def test_leading_whitespace_tolerated(self) -> None:
        result = classify_farm("  Market data farm connection is OK:usfarm")
        assert result is not None
        assert result.is_ok is True


# ---------------------------------------------------------------------------
# IbDiagnostic.from_callback
# ---------------------------------------------------------------------------


class TestIbDiagnosticFromCallback:
    def test_informational_farm_ok(self) -> None:
        diag = IbDiagnostic.from_callback(-1, 2104, "Market data farm connection is OK:usfarm")
        assert diag.code == 2104
        assert diag.tier is ErrorTier.INFORMATIONAL
        assert diag.message == "Market data farm connection is OK:usfarm"
        assert diag.req_id == -1
        assert diag.farm is not None
        assert diag.farm.is_ok is True
        assert diag.farm.farm == "usfarm"

    def test_connectivity_degraded_no_farm(self) -> None:
        diag = IbDiagnostic.from_callback(-1, 1100, "Connectivity between IB and TWS has been lost")
        assert diag.tier is ErrorTier.CONNECTIVITY_DEGRADED
        assert diag.farm is None

    def test_genuine_error_with_req_id(self) -> None:
        diag = IbDiagnostic.from_callback(7, 200, "No security definition has been found for the request")
        assert diag.tier is ErrorTier.GENUINE_ERROR
        assert diag.req_id == 7
        assert diag.farm is None

    def test_market_data_warning(self) -> None:
        diag = IbDiagnostic.from_callback(9001, 10167, "Displaying delayed market data")
        assert diag.tier is ErrorTier.MARKET_DATA_WARNING

    def test_in_band_broken_message_override(self) -> None:
        diag = IbDiagnostic.from_callback(-1, 2108, "Market data farm connection is broken:usfarm")
        assert diag.tier is ErrorTier.CONNECTIVITY_DEGRADED
        assert diag.farm is not None
        assert diag.farm.is_ok is False

    def test_frozen_immutability(self) -> None:
        diag = IbDiagnostic.from_callback(-1, 2104, "OK:usfarm")
        with pytest.raises((AttributeError, TypeError)):
            diag.code = 999  # type: ignore[misc]  # testing runtime immutability
