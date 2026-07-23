"""JVM contract test: DataPort.register_derived / table(name) resolution.

Covers the named-registry primitive: a derived table is registered once and
resolved by name; every caller resolving it by name shares the SAME underlying
live computation (not recomputed per lookup). Also exercises a worked
custom-Beta example end to end (as_of_join + derive, composed via the public
port surface): a small Beta *reference* table (one row per symbol, computed
offline — Beta itself is a time-series statistic the safe-DSL cannot compute
inline) is as-of-joined onto the live ``position`` table, then a
``BetaAdjExposure`` column is computed over the joined result via the
validated safe-DSL and registered under a name, so every consumer resolving
it shares one live engine computation. ``_register_beta_adjusted_position``
below is a self-contained example CONSUMER of ibda (not part of the package
itself), inlined here rather than imported from a separate example module so
this test carries no first-party dependency outside this package.

Run with:
    uv run pytest ibda/tests_jvm/test_register_derived.py -v
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pyarrow as pa
import pytest

from ibda import connect
from ibda.result import Result
from ibda.schema import Column, DType, Schema

#: Schema of the caller-supplied reference table: one Beta estimate per symbol.
#: Not a canonical ibda schema — the caller owns this table's shape.
_BETA_REF_SCHEMA = Schema(
    name="beta_ref",
    doc="Per-symbol Beta reference, computed offline (e.g. via ibda.analytics.benchmark).",
    columns=(
        Column("Sym", DType.STRING, nullable=False, doc="instrument symbol"),
        Column("ConId", DType.INT64, nullable=False, doc="IBKR contract id"),
        Column("Beta", DType.FLOAT64, nullable=False, doc="beta estimate vs the chosen benchmark"),
    ),
)


def _register_beta_adjusted_position(tables: Mapping[str, Any]) -> Result:
    """Register + return the ``beta_adjusted_position`` derived table.

    *tables* must include ``"position"`` (canonical) and ``"beta_ref"`` (a live
    engine table matching ``_BETA_REF_SCHEMA`` — ``Sym``, ``ConId``, ``Beta``,
    one row per symbol the caller cares about).
    """
    port = connect(tables)
    position = port.table("position")
    # beta_ref isn't a canonical table, so it can't be resolved via
    # port.table(); wrap the caller-supplied handle directly with its schema.
    beta_ref = Result(handle=tables["beta_ref"], schema=_BETA_REF_SCHEMA, port=port)

    joined = port.as_of_join(position, beta_ref, on=["Sym", "ConId"], joins=["Beta"])

    return port.register_derived(
        "beta_adjusted_position",
        base=joined,
        columns={"BetaAdjExposure": "MarketValue * Beta"},
    )


def _position_table() -> Any:
    from deephaven import new_table  # noqa: PLC0415 — JVM-gated
    from deephaven.column import double_col, long_col, string_col  # noqa: PLC0415

    return new_table([
        string_col("Account", ["DU1", "DU1"]),
        long_col("ConId", [1, 2]),
        string_col("Sym", ["AAPL", "MSFT"]),
        string_col("SecType", ["STK", "STK"]),
        double_col("Qty", [10.0, -5.0]),
        double_col("AvgCost", [150.0, 300.0]),
        double_col("MarketPrice", [155.0, 290.0]),
        double_col("MarketValue", [1550.0, -1450.0]),
        double_col("UnrealizedPnl", [50.0, -50.0]),
    ])


def _beta_ref_table() -> Any:
    """A small caller-supplied reference table: one Beta per (Sym, ConId)."""
    from deephaven import new_table  # noqa: PLC0415 — JVM-gated
    from deephaven.column import double_col, long_col, string_col  # noqa: PLC0415

    return new_table([
        string_col("Sym", ["AAPL", "MSFT"]),
        long_col("ConId", [1, 2]),
        double_col("Beta", [1.2, 0.9]),
    ])


def _adapter() -> Any:
    from ibda.adapters.deephaven.adapter import DeephavenPort  # noqa: PLC0415

    return DeephavenPort({"position": _position_table()})


# ---------------------------------------------------------------------------
# Named registry
# ---------------------------------------------------------------------------


def test_register_derived_is_resolvable_by_name() -> None:
    port = _adapter()
    base = port.table("position")
    port.register_derived("beta_custom", base=base, columns={"Doubled": "Qty * 2"})

    result = port.table("beta_custom")
    snapshot: pa.Table = result.snapshot()
    assert snapshot.column("Doubled").to_pylist() == pytest.approx([20.0, -10.0])


def test_register_derived_two_subscribers_share_one_handle() -> None:
    """Both `port.table("beta_custom")` calls must return the SAME live handle —
    the unified-calculation requirement (one engine computation, N consumers)."""
    port = _adapter()
    base = port.table("position")
    port.register_derived("beta_custom", base=base, columns={"Doubled": "Qty * 2"})

    first = port.table("beta_custom")
    second = port.table("beta_custom")
    assert first.handle is second.handle


def test_register_derived_rejects_canonical_name_collision() -> None:
    port = _adapter()
    base = port.table("position")
    with pytest.raises(ValueError, match="collides with a canonical table name"):
        port.register_derived("position", base=base, columns={"Doubled": "Qty * 2"})


def test_register_derived_rejects_duplicate_registration() -> None:
    port = _adapter()
    base = port.table("position")
    port.register_derived("beta_custom", base=base, columns={"Doubled": "Qty * 2"})
    with pytest.raises(ValueError, match="already registered"):
        port.register_derived("beta_custom", base=base, columns={"Tripled": "Qty * 3"})


def test_table_unknown_derived_name_raises_unknown_table() -> None:
    from ibda.errors import UnknownTable  # noqa: PLC0415

    port = _adapter()
    with pytest.raises(UnknownTable):
        port.table("never_registered")


def test_register_derived_tags_untrusted_origin() -> None:
    port = _adapter()
    base = port.table("position")
    port.register_derived("beta_custom", base=base, columns={"Doubled": "Qty * 2"})
    assert port.derived_origin("beta_custom") == "untrusted"
    assert port.derived_origin("never_registered") is None


# ---------------------------------------------------------------------------
# Worked example: custom Beta-adjusted exposure (as_of_join + derive + register)
# ---------------------------------------------------------------------------


def test_worked_custom_beta_derivation_example() -> None:
    """Exercises `_register_beta_adjusted_position` (this module's worked example) end to end."""
    tables = {"position": _position_table(), "beta_ref": _beta_ref_table()}
    result = _register_beta_adjusted_position(tables)
    snapshot: pa.Table = result.snapshot()

    # AAPL: MarketValue 1550.0 * Beta 1.2 = 1860.0
    # MSFT: MarketValue -1450.0 * Beta 0.9 = -1305.0
    by_sym = dict(zip(snapshot.column("Sym").to_pylist(), snapshot.column("BetaAdjExposure").to_pylist()))
    assert by_sym["AAPL"] == pytest.approx(1860.0)
    assert by_sym["MSFT"] == pytest.approx(-1305.0)


# ---------------------------------------------------------------------------
# NOTE: a downstream query layer built on top of `ibda` (resolving a
# `register_derived` table by name, not just canonical tables, for snapshot,
# filter, and unknown-table-rejection paths) is exercised separately, outside
# this package. The underlying `ibda` contract it depends on (`port.table(name)`
# resolves BOTH canonical and registered-derived names identically, per
# `derived_schema`/`derived_origin`) is covered directly by the tests above —
# any additional coverage needed for a downstream consumer's own routing is
# specific to that consumer, not to `ibda` itself.
# ---------------------------------------------------------------------------
