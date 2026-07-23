"""Tests for ibda's public surface.

Covers two invariants:
  1. `import ibda` must NOT pull in a compute engine (deephaven).
  2. The public names are present and DeephavenPort is hidden.
"""

from __future__ import annotations

import sys


def _is_purge_target(mod: str) -> bool:
    return (
        mod == "ibda"
        or mod.startswith("ibda.")
        or mod == "deephaven"
        or mod.startswith(("deephaven.", "deephaven_"))
    )


def test_importing_ibda_does_not_import_deephaven() -> None:
    """`import ibda` must not pull in a compute engine.

    HERMETIC: the purge below is restored afterwards. Without restoration this test
    corrupted the whole pytest session — every module imported later got a *fresh*
    `ibda.schema`, so dtype/enum identity checks against objects captured earlier
    silently failed (e.g. ibda_mcp's to_predicate + register_derived tests). It only
    looked harmless because `ibda/tests` happened to run last in `testpaths`.
    """
    # Snapshot the real module objects so we can put them back byte-for-byte.
    saved = {m: sys.modules[m] for m in list(sys.modules) if _is_purge_target(m)}
    for mod in saved:
        del sys.modules[mod]

    try:
        import ibda  # noqa: F401  — must re-execute __init__ to catch a top-level leak

        leaked = [
            m
            for m in sys.modules
            if m == "deephaven" or m.startswith(("deephaven.", "deephaven_"))
        ]
        assert leaked == [], f"importing ibda pulled in the engine: {leaked}"
    finally:
        # Drop the freshly-imported copies, then restore the originals, so no other
        # test observes a second, non-identical `ibda` module tree.
        for mod in [m for m in sys.modules if _is_purge_target(m)]:
            del sys.modules[mod]
        sys.modules.update(saved)


def test_public_names_present() -> None:
    import ibda

    assert hasattr(ibda, "connect")
    assert hasattr(ibda, "Result")
    assert hasattr(ibda, "Stream")
    assert hasattr(ibda, "schema")
    assert hasattr(ibda, "errors")
    assert hasattr(ibda, "DataPort")
    assert not hasattr(ibda, "DeephavenPort")


def test_timeseries_functions_on_top_level_surface() -> None:
    import ibda

    assert hasattr(ibda, "rolling_performance")
    assert hasattr(ibda, "periodic_returns")
    assert "rolling_performance" in ibda.__all__
    assert "periodic_returns" in ibda.__all__


def test_benchmark_relative_on_top_level_surface() -> None:
    import ibda

    for name in ("relative_summary", "relative_metrics", "RelativeSummary"):
        assert hasattr(ibda, name), name
        assert name in ibda.__all__, name
    # ibda.benchmarks (Yahoo) was removed — benchmark bars are IB-only now
    # (ibda.adapters.ibkr.marketdata.historical_bars), not a top-level public-surface symbol.
    assert not hasattr(ibda, "benchmarks")
    # Clean rename (no back-compat alias): relative_performance no longer exists.
    assert not hasattr(ibda, "relative_performance")


def test_rolling_relative_on_top_level_surface() -> None:
    import ibda

    assert hasattr(ibda, "rolling_relative")
    assert "rolling_relative" in ibda.__all__


def test_roundtrips_on_top_level_surface() -> None:
    import ibda

    for name in (
        "Fill", "RoundTrip", "ReconstructResult", "JournalResult",
        "reconstruct_round_trips", "aggregate_round_trips",
    ):
        assert hasattr(ibda, name), name
        assert name in ibda.__all__, name
