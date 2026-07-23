"""Pure tests for DeephavenPort.live_table — the in-engine escape hatch (Pass A).

No JVM: DeephavenPort.table() resolution is engine-agnostic (it just looks up
a name in a dict and wraps whatever handle was stored), so these tests use a
plain sentinel object in place of a real deephaven Table. JVM-level proof that
a *real* live Table round-trips through live_table() lives in
ibda/tests_jvm/test_live_table_accessor_jvm.py.

``live_table`` is a deliberate, dated design decision (2026-07-08): a documented in-engine
escape hatch, declared only on ``DeephavenPort`` (never on the ``DataPort`` ABC) so
out-of-process consumers never see it.
"""
from __future__ import annotations

import pytest

from ibda.adapters.deephaven.adapter import DeephavenPort
from ibda.errors import UnknownTable
from ibda.port import DataPort
from ibda.schema import ALL as ALL_SCHEMAS


def test_live_table_returns_the_underlying_handle_for_a_canonical_name() -> None:
    handle = object()
    canonical_name = next(iter(ALL_SCHEMAS))
    port = DeephavenPort({canonical_name: handle})

    assert port.live_table(canonical_name) is handle


def test_live_table_matches_table_dot_handle() -> None:
    """live_table(name) must be exactly table(name).handle — same resolution path."""
    handle = object()
    canonical_name = next(iter(ALL_SCHEMAS))
    port = DeephavenPort({canonical_name: handle})

    assert port.live_table(canonical_name) is port.table(canonical_name).handle


def test_live_table_resolves_registered_derived_names() -> None:
    """A name registered via register_derived must also resolve through live_table."""
    handle = object()
    canonical_name = next(iter(ALL_SCHEMAS))
    port = DeephavenPort({canonical_name: handle})

    class _FakeDerivedResult:
        def __init__(self, h: object) -> None:
            self.handle = h

    # Bypass register_derived's engine-side derive() call (no JVM here) by
    # directly populating the internal derived-name maps it would populate.
    derived_handle = object()
    port._tables["my_derived"] = derived_handle
    port._derived_schemas["my_derived"] = ALL_SCHEMAS[canonical_name]

    assert port.live_table("my_derived") is derived_handle


def test_live_table_raises_unknown_table_on_bad_name() -> None:
    port = DeephavenPort({})

    with pytest.raises(UnknownTable):
        port.live_table("not_a_real_table")


def test_live_table_is_not_on_dataport_abc() -> None:
    """Engine-hiding invariant: DataPort (the ABC external consumers hold) must
    not expose live_table — only the concrete DeephavenPort does."""
    assert not hasattr(DataPort, "live_table")


def test_live_table_is_on_deephaven_port_concrete_class() -> None:
    assert hasattr(DeephavenPort, "live_table")


# Note: an "ArrowDataPort" adapter (mentioned in the ADR as a second concrete
# DataPort implementation that must also lack live_table) does not exist yet
# in this tree — there is currently exactly one concrete DataPort adapter
# (DeephavenPort). The invariant that matters structurally is enforced by the
# two tests above: live_table is declared only on DeephavenPort, never on the
# ABC, so ANY future adapter (Arrow-backed or otherwise) that subclasses
# DataPort will not inherit it.
