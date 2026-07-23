"""Non-JVM tests for the idempotent start_server helper.

The tests inject a fake ``Server`` class into ``deephaven_server`` so that no
JVM is ever started.  The real ``start_server`` implementation is exercised
directly with the fake class in place.

Three scenarios:

1. ``Server.instance`` is already set → ``start_server`` returns it and does
   NOT construct or start a new Server.
2. ``Server.instance`` is None → ``start_server`` constructs once, calls
   ``.start()``, and returns the new instance.
3. Construction raises but ``Server.instance`` is non-None at that point →
   ``start_server`` recovers and returns the existing instance instead of
   propagating.
"""
from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fake Server helpers
# ---------------------------------------------------------------------------

class _FakeServerBase:
    """Base for fake Server classes; sets ``instance`` on construction."""

    instance: Any = None  # mirrors the real deephaven_server.Server class attr

    def __init__(self, port: int = 10000, jvm_args: list[str] | None = None) -> None:
        type(self).instance = self
        self.port = port
        self.started: bool = False

    def start(self) -> None:
        self.started = True


def _make_fresh_server_class() -> type[_FakeServerBase]:
    """Return a fresh subclass so each test gets an isolated instance state."""

    class FreshServer(_FakeServerBase):
        instance: Any = None

    return FreshServer


def _make_fake_dh_module(server_cls: type[_FakeServerBase]) -> types.ModuleType:
    mod = types.ModuleType("deephaven_server")
    mod.Server = server_cls  # type: ignore[attr-defined]
    return mod


# ---------------------------------------------------------------------------
# The actual start_server function under test
# ---------------------------------------------------------------------------

def _run_start_server(server_cls: type[_FakeServerBase], **kwargs: Any) -> Any:
    """Call the real start_server with deephaven_server patched to use *server_cls*."""
    fake_mod = _make_fake_dh_module(server_cls)
    with patch.dict(sys.modules, {"deephaven_server": fake_mod}):
        import ibda.adapters.deephaven.server as srv_mod
        return srv_mod.start_server(**kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_start_server_reuses_existing_instance() -> None:
    """When Server.instance is non-None, start_server returns it; no new construction."""
    ServerCls = _make_fresh_server_class()
    sentinel: Any = object()
    ServerCls.instance = sentinel  # pre-populate as if a server was already started

    # Track instantiation attempts — the __init__ of the class should NOT be called.
    init_mock = MagicMock()
    original_init = ServerCls.__init__

    def tracking_init(self: Any, **kwargs: Any) -> None:
        init_mock(**kwargs)
        original_init(self, **kwargs)

    ServerCls.__init__ = tracking_init  # type: ignore[assignment]

    result = _run_start_server(ServerCls, port=9999)

    assert result is sentinel
    init_mock.assert_not_called()


def test_start_server_constructs_when_no_instance() -> None:
    """When Server.instance is None, start_server constructs once and starts it."""
    ServerCls = _make_fresh_server_class()
    assert ServerCls.instance is None  # precondition

    result = _run_start_server(ServerCls, port=10002)

    assert ServerCls.instance is not None
    assert result is ServerCls.instance
    assert result.started is True  # .start() was called


def test_start_server_recovers_from_init_exception() -> None:
    """If Server() raises and Server.instance is set, return the existing instance."""
    ServerCls = _make_fresh_server_class()
    existing: Any = object()
    ServerCls.instance = existing  # already running

    def raising_init(self: Any, **kwargs: Any) -> None:
        raise RuntimeError("already one instance")

    ServerCls.__init__ = raising_init  # type: ignore[assignment]

    result = _run_start_server(ServerCls, port=10003)

    assert result is existing
