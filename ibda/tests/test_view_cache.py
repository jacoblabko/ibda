"""Tests for _VIEW_CACHE WeakKeyDictionary across positions, orders, and executions.

Verifies the two key correctness properties of the WeakKeyDictionary cache:

1. Isolation: two distinct session objects never share cache entries.
2. Eviction: a cache entry is GC'd when its session is garbage-collected — no
   memory leak and no stale-view hazard from id() reuse.

All three modules (positions, orders, executions) are tested separately because
each carries its own module-level _VIEW_CACHE instance.

No JVM, no TWS, no engine imports — the tests manipulate the WeakKeyDictionary
directly using SimpleNamespace sessions (which are weak-referenceable).
"""
from __future__ import annotations

import gc
from typing import Any

from ibda.adapters.ibkr import executions as executions_mod
from ibda.adapters.ibkr import orders as orders_mod
from ibda.adapters.ibkr import positions as positions_mod


class _WeakableSession:
    """Minimal fake session that supports weak references.

    ``types.SimpleNamespace`` is C-implemented in CPython 3.12 and does NOT
    support ``weakref.ref``, so it cannot be used as a WeakKeyDictionary key.
    A plain Python class supports weak references by default.  The production
    session (``deephaven_ib.IbSessionTws``) is also a plain Python class, so
    this matches the real-world type.
    """


def _fresh_session() -> _WeakableSession:
    """Return a fresh, unique, weak-referenceable fake session object."""
    return _WeakableSession()


# ---------------------------------------------------------------------------
# Helper: parametric tests run against each module's _VIEW_CACHE
# ---------------------------------------------------------------------------


def _assert_cache_isolation(cache: Any) -> None:
    """Two distinct session keys must store and retrieve independent values."""
    session_a = _fresh_session()
    session_b = _fresh_session()
    view_a = object()
    view_b = object()

    cache[session_a] = view_a
    cache[session_b] = view_b

    assert cache[session_a] is view_a, "session_a must retrieve its own view"
    assert cache[session_b] is view_b, "session_b must retrieve its own view"
    assert cache[session_a] is not view_b, "session_a must not see session_b's view"
    assert cache[session_b] is not view_a, "session_b must not see session_a's view"


def _assert_cache_eviction(cache: Any) -> None:
    """A cache entry must be automatically removed when its session is GC'd."""
    session = _fresh_session()
    sentinel = object()
    before_len = len(cache)

    cache[session] = sentinel
    assert len(cache) == before_len + 1, "entry must appear in cache after insertion"
    assert cache[session] is sentinel

    # Drop the only reference to the session and force GC.
    del session
    gc.collect()

    assert len(cache) == before_len, (
        "entry must be evicted from WeakKeyDictionary after session is GC'd"
    )


# ---------------------------------------------------------------------------
# positions._VIEW_CACHE
# ---------------------------------------------------------------------------


def test_positions_view_cache_isolates_distinct_sessions() -> None:
    _assert_cache_isolation(positions_mod._VIEW_CACHE)


def test_positions_view_cache_evicts_entry_on_session_gc() -> None:
    _assert_cache_eviction(positions_mod._VIEW_CACHE)


# ---------------------------------------------------------------------------
# orders._VIEW_CACHE
# ---------------------------------------------------------------------------


def test_orders_view_cache_isolates_distinct_sessions() -> None:
    _assert_cache_isolation(orders_mod._VIEW_CACHE)


def test_orders_view_cache_evicts_entry_on_session_gc() -> None:
    _assert_cache_eviction(orders_mod._VIEW_CACHE)


# ---------------------------------------------------------------------------
# executions._VIEW_CACHE
# ---------------------------------------------------------------------------


def test_executions_view_cache_isolates_distinct_sessions() -> None:
    _assert_cache_isolation(executions_mod._VIEW_CACHE)


def test_executions_view_cache_evicts_entry_on_session_gc() -> None:
    _assert_cache_eviction(executions_mod._VIEW_CACHE)
