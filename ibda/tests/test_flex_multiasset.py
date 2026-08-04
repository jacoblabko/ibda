"""Tests for multi-asset Flex mapping: STK, OPT, FUT, and CASH (FX) rows.

Runs parse_statement + flex_sections_to_canonical over report_multiasset.xml
and verifies that all asset categories map without error, and that SecType and
Multiplier are populated correctly for each asset class.

Design choices exercised by the fixture:
- STK (NVDA)        — ibExecID present; BUY; standard equity; Multiplier=1.0
- OPT (AAPL call)   — ibExecID present; BUY; multiplier=100 in XML; Multiplier=100.0
- FUT (ESM6)        — NO ibExecID but tradeID present; SELL (quantity=-2); multiplier=50
- CASH/FX (EUR.USD) — NO ibExecID, NO tradeID → synthetic id; BUY; quantity=100000;
                       no multiplier attribute → Multiplier=1.0

OPT notional check:
  Price × Qty × Multiplier = 3.20 × 5 × 100 = 1600.0 USD
  (matches ``proceeds="-1600.00"`` in the fixture).

FX "price" semantics note:
  For assetCategory="CASH" (FX), ``tradePrice`` is an exchange rate (e.g. 1.08520
  USD per EUR), not a per-unit price.  ``quantity`` is in base-currency units
  (100,000 EUR).  Canonical ``Qty`` and ``Price`` are populated from these as-is
  — which is structurally correct but semantically distinct from equity trades.
  Consumers can gate on ``SecType == "CASH"`` to branch on FX rows.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ibda.adapters.ibkr.flex.mapping import flex_sections_to_canonical
from ibda.adapters.ibkr.flex.parse import parse_statement
from ibda.schema import EXECUTION

_FIXTURE = Path(__file__).parent / "fixtures" / "flex" / "report_multiasset.xml"


def _sections() -> dict[str, Any]:
    parsed = parse_statement(_FIXTURE.read_text())
    assert parsed["status"] == "ok", f"parse failed: {parsed}"
    result: dict[str, Any] = parsed["sections"]
    return result


def _execs() -> list[dict[str, Any]]:
    return flex_sections_to_canonical(_sections())["execution"]


# ---------------------------------------------------------------------------
# Basic structural tests
# ---------------------------------------------------------------------------


def test_multiasset_fixture_parses_to_four_execution_rows() -> None:
    """report_multiasset.xml has 4 Trade elements; all should map."""
    execs = _execs()
    assert len(execs) == 4


def test_all_rows_have_required_execution_columns() -> None:
    """Every mapped row must contain only canonical EXECUTION column names."""
    execs = _execs()
    cols = set(EXECUTION.column_names)
    for row in execs:
        assert set(row).issubset(cols), (
            f"Row for {row.get('Sym')!r} has unexpected columns: "
            f"{set(row) - cols}"
        )


def test_all_rows_have_non_empty_exec_id() -> None:
    """ExecId must be non-empty for every asset category."""
    for row in _execs():
        assert row["ExecId"], f"Empty ExecId for row: {row!r}"


def test_all_rows_have_unsigned_qty() -> None:
    """Canonical Qty is always >= 0 for all asset categories."""
    for row in _execs():
        qty = row["Qty"]
        assert isinstance(qty, float), f"Qty not float for {row.get('Sym')!r}: {qty!r}"
        assert qty >= 0.0, f"Negative Qty for {row.get('Sym')!r}: {qty}"


def test_all_rows_have_valid_side() -> None:
    """Side must be 'BUY' or 'SELL' for all asset categories."""
    for row in _execs():
        assert row["Side"] in ("BUY", "SELL"), (
            f"Invalid Side for {row.get('Sym')!r}: {row['Side']!r}"
        )


# ---------------------------------------------------------------------------
# Per-asset-category assertions
# ---------------------------------------------------------------------------


class TestStk:
    """STK row: NVDA BUY with ibExecID present."""

    def _row(self) -> dict[str, Any]:
        return next(r for r in _execs() if r["Sym"] == "NVDA")

    def test_sym(self) -> None:
        assert self._row()["Sym"] == "NVDA"

    def test_side_buy(self) -> None:
        assert self._row()["Side"] == "BUY"

    def test_qty_unsigned(self) -> None:
        assert self._row()["Qty"] == 50.0

    def test_price(self) -> None:
        assert abs(self._row()["Price"] - 875.50) < 1e-9

    def test_exec_id_is_ibexecid(self) -> None:
        """ibExecID is present, so ExecId should equal it (not synthetic)."""
        row = self._row()
        assert row["ExecId"] == "0000e9bf.667fe2a3.01.01"

    def test_sec_type_is_stk(self) -> None:
        assert self._row()["SecType"] == "STK"

    def test_multiplier_is_one(self) -> None:
        """STK rows have no multiplier attribute in Flex; must default to 1.0."""
        assert self._row()["Multiplier"] == 1.0


class TestOpt:
    """OPT row: AAPL call BUY with ibExecID; multiplier=100."""

    def _row(self) -> dict[str, Any]:
        return next(r for r in _execs() if "AAPL" in (r["Sym"] or ""))

    def test_sym_preserved(self) -> None:
        """Option symbol including expiry/strike/right is preserved as-is."""
        sym = self._row()["Sym"] or ""
        assert "AAPL" in sym

    def test_side_buy(self) -> None:
        assert self._row()["Side"] == "BUY"

    def test_qty_unsigned(self) -> None:
        assert self._row()["Qty"] == 5.0

    def test_price_is_premium(self) -> None:
        """tradePrice for OPT is per-contract premium (not including multiplier)."""
        assert abs(self._row()["Price"] - 3.20) < 1e-9

    def test_exec_id_is_ibexecid(self) -> None:
        assert self._row()["ExecId"] == "0000e9bf.667fe2a3.02.01"

    def test_sec_type_is_opt(self) -> None:
        assert self._row()["SecType"] == "OPT"

    def test_multiplier_is_100(self) -> None:
        assert self._row()["Multiplier"] == 100.0

    def test_notional_price_times_qty_times_multiplier(self) -> None:
        """Contract notional = Price × Qty × Multiplier = 3.20 × 5 × 100 = 1600.0.

        Matches ``proceeds="-1600.00"`` in the fixture.
        """
        row = self._row()
        notional = float(row["Price"]) * float(row["Qty"]) * float(row["Multiplier"])
        assert abs(notional - 1600.0) < 1e-9


class TestFut:
    """FUT row: ESM6 SELL with tradeID only (no ibExecID); quantity=-2 in XML."""

    def _row(self) -> dict[str, Any]:
        return next(r for r in _execs() if r["Sym"] == "ESM6")

    def test_sym(self) -> None:
        assert self._row()["Sym"] == "ESM6"

    def test_side_sell(self) -> None:
        assert self._row()["Side"] == "SELL"

    def test_qty_unsigned_for_sell(self) -> None:
        """Fixture has quantity=-2; canonical Qty must be abs(−2) = 2.0."""
        assert self._row()["Qty"] == 2.0

    def test_price(self) -> None:
        assert abs(self._row()["Price"] - 5305.00) < 1e-9

    def test_exec_id_is_trade_id_fallback(self) -> None:
        """No ibExecID → tradeID ('123456789') used as ExecId."""
        assert self._row()["ExecId"] == "123456789"

    def test_sec_type_is_fut(self) -> None:
        assert self._row()["SecType"] == "FUT"

    def test_multiplier_is_50(self) -> None:
        assert self._row()["Multiplier"] == 50.0


class TestCashFx:
    """CASH/FX row: EUR.USD BUY with no ibExecID/tradeID → synthetic id."""

    def _row(self) -> dict[str, Any]:
        return next(r for r in _execs() if r["Sym"] == "EUR.USD")

    def test_sym_fx_pair_preserved(self) -> None:
        """FX pair symbol is preserved exactly."""
        assert self._row()["Sym"] == "EUR.USD"

    def test_side_buy(self) -> None:
        assert self._row()["Side"] == "BUY"

    def test_qty_unsigned(self) -> None:
        """Qty is base-currency units (100,000 EUR) — unsigned."""
        assert self._row()["Qty"] == 100_000.0

    def test_price_is_exchange_rate(self) -> None:
        """Price is the FX exchange rate (USD per EUR)."""
        assert abs(self._row()["Price"] - 1.08520) < 1e-6

    def test_exec_id_synthetic_when_no_ids(self) -> None:
        """No ibExecID and no tradeID → synthetic id; must be non-empty and marked.

        The synthetic form is a ``synx-`` prefixed content hash of
        Account/Sym/DateTime/Qty/Price, not the old symbol-embedding string — see
        ``_synthetic_exec_id`` for why the account had to be folded in and why the id
        must not end in a price.
        """
        exec_id = self._row()["ExecId"]
        assert exec_id  # non-empty
        assert exec_id.startswith("synx-")

    def test_sec_type_is_cash(self) -> None:
        """CASH/FX rows carry SecType='CASH'."""
        assert self._row()["SecType"] == "CASH"

    def test_multiplier_defaults_to_one_when_absent(self) -> None:
        """FX rows omit the multiplier attribute in Flex; must default to 1.0."""
        assert self._row()["Multiplier"] == 1.0


# ---------------------------------------------------------------------------
# ExecId fallback-chain completeness test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sym,expected_exec_id_or_starts_with,is_synthetic",
    [
        ("NVDA", "0000e9bf.667fe2a3.01.01", False),       # ibExecID
        ("ESM6", "123456789", False),                      # tradeID fallback
        ("EUR.USD", "synx-", True),                        # synthetic prefix
    ],
)
def test_exec_id_fallback_chain(
    sym: str,
    expected_exec_id_or_starts_with: str,
    is_synthetic: bool,
) -> None:
    """Verify ibExecID > tradeID > synthetic precedence per row."""
    row = next(r for r in _execs() if r["Sym"] == sym)
    exec_id: str = row["ExecId"]
    if is_synthetic:
        assert exec_id.startswith(expected_exec_id_or_starts_with), (
            f"Expected synthetic ExecId to start with {expected_exec_id_or_starts_with!r}; "
            f"got {exec_id!r}"
        )
    else:
        assert exec_id == expected_exec_id_or_starts_with, (
            f"Expected ExecId={expected_exec_id_or_starts_with!r}; got {exec_id!r}"
        )
