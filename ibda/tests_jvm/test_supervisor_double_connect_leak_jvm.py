"""JVM regression test for a prior regression — IbkrSupervisor.connect() leaks the
prior session when called again while already connected.

Root cause: connect() always constructed a new session and unconditionally
overwrote self._session, with no check of whether a prior session was still
live. An earlier fix (see ``test_supervisor_connect_leak_jvm.py``) fixed the
*failed*-connect leak (a locally-created session never assigned to self._session
on a failure path); this is the separate, narrower case that fix explicitly left
open: a caller invoking connect() a second time on an already-CONNECTED
supervisor (no is_connected() guard first — e.g. a reconnect attempt after a
farm blip that did not actually drop the old session) previously leaked the
still-live prior session (open socket + native threads) with no disconnect()
call.

Caller-path finding: every production caller
(``ibda.adapters.ibkr.live.connect_live``) constructs a brand-new
``IbkrSupervisor()`` per connect attempt, so this exact re-entrant call
pattern is not currently exercised by any production code path — it is a
defensive fix, not a confirmed live leak. It is still worth closing: the
class is a general-purpose lifecycle manager (``IbkrSupervisor()`` +
``connect()`` is public API), and nothing prevents a future caller from
reusing one supervisor instance across a reconnect.

Fix (chosen semantics: disconnect-first / "connect() forces a fresh
connection"): at the top of connect(), if self._session is not None and
still reports connected (guarded — an unreadable session is treated as
still-live), self.disconnect() is called before building the new session.
This keeps every connect() call's parameters (host/port/client_id/
read_only) authoritative on every invocation, rather than silently
no-op'ing (and ignoring possibly-different parameters) when a session
already exists.

``deephaven_ib`` requires the embedded Deephaven server to be constructed
before it can be imported (bare ``import deephaven_ib`` outside a JVM-backed
test raises "The Deephaven Server has not been initialized"), so this test
must live under ``ibda/tests_jvm/`` even though it never touches a real
Deephaven Table — it only needs the real, JVM-backed ``deephaven_ib`` module
object to monkeypatch ``IbSessionTws`` on. Mirrors the fake-session approach
of ``test_supervisor_connect_leak_jvm.py``.

Run with:
    uv run pytest ibda/tests_jvm/test_supervisor_double_connect_leak_jvm.py -q
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ibda.adapters.ibkr.supervisor import IbkrSupervisor


# ---------------------------------------------------------------------------
# Fake deephaven_ib.IbSessionTws
# ---------------------------------------------------------------------------


class _FakeIbSessionTws:
    """Fakes the IbSessionTws surface touched by IbkrSupervisor.connect().

    Each instance is appended to *created* (in construction order) so a test
    can inspect every session connect() ever built. ``_connected`` is drawn
    from *outcomes* by construction index, simulating a scripted sequence of
    connect attempts (e.g. two successes in a row).
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        client_id: int,
        read_only: bool,
        download_short_rates: bool,
        outcomes: list[bool],
        created: list[_FakeIbSessionTws],
    ) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id
        self.read_only = read_only
        self.download_short_rates = download_short_rates
        self.connect_calls = 0
        self.disconnect_calls = 0
        self._connected = outcomes[len(created)]
        created.append(self)

    def connect(self) -> None:
        self.connect_calls += 1

    def is_connected(self) -> bool:
        return self._connected

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self._connected = False


def _make_factory(
    outcomes: list[bool],
    created: list[_FakeIbSessionTws],
    session_cls: type[_FakeIbSessionTws] = _FakeIbSessionTws,
) -> Callable[..., _FakeIbSessionTws]:
    def factory(
        *,
        host: str,
        port: int,
        client_id: int,
        read_only: bool,
        download_short_rates: bool,
    ) -> _FakeIbSessionTws:
        return session_cls(
            host=host,
            port=port,
            client_id=client_id,
            read_only=read_only,
            download_short_rates=download_short_rates,
            outcomes=outcomes,
            created=created,
        )

    return factory


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_connect_twice_while_already_connected_disconnects_prior_session(
    monkeypatch: Any,
) -> None:
    """connect() called a second time while the first session is still
    CONNECTED must disconnect the first session exactly once before
    establishing the second — zero leaked sessions, mirrors the earlier
    fix's zero-leak guarantee for the double-connect case."""
    import deephaven_ib

    outcomes = [True, True]
    created: list[_FakeIbSessionTws] = []
    monkeypatch.setattr(deephaven_ib, "IbSessionTws", _make_factory(outcomes, created))
    monkeypatch.setattr("ibda.adapters.ibkr.supervisor.time.sleep", lambda _: None)

    sup = IbkrSupervisor()
    sup.connect(settle_secs=0)
    assert len(created) == 1
    assert created[0].disconnect_calls == 0
    first_session = sup._session

    # Second connect() call while still connected — no is_connected() guard
    # by the caller, simulating a reconnect after a farm blip.
    sup.connect(settle_secs=0)

    assert len(created) == 2, "a new session is still constructed for the second connect()"
    assert first_session.disconnect_calls == 1, (
        "the first (still-live) session must be disconnected exactly once "
        "before the second connect() establishes a new session"
    )
    assert created[1].disconnect_calls == 0, "the second (newly successful) session must not be disconnected"
    assert sup._session is created[1]
    assert sup._session is not first_session


def test_connect_twice_disconnects_prior_session_even_when_is_connected_raises(
    monkeypatch: Any,
) -> None:
    """If the prior session's is_connected() itself raises on the SECOND
    connect() call's guard check (unreadable / broken state, distinct from
    the first connect()'s own post-connect check, which must still succeed
    normally), the guard must treat it as still-live and disconnect it
    defensively rather than skip cleanup or propagate the read failure."""
    import deephaven_ib

    created: list[_FakeIbSessionTws] = []
    monkeypatch.setattr(deephaven_ib, "IbSessionTws", _make_factory([True, True], created))
    monkeypatch.setattr("ibda.adapters.ibkr.supervisor.time.sleep", lambda _: None)

    sup = IbkrSupervisor()
    sup.connect(settle_secs=0)
    first_session = sup._session
    assert first_session.disconnect_calls == 0

    # Only AFTER the first connect() has already completed successfully
    # (its own post-connect is_connected() check already passed), break
    # is_connected() on that same instance to simulate it going unreadable
    # by the time the second connect() call's leak guard checks it. The
    # single [True, True] factory (same as the previous test) still supplies
    # a normal, non-raising session for the second connect() call.
    def _raise_is_connected() -> bool:
        raise RuntimeError("simulated JNI session read failure")

    monkeypatch.setattr(first_session, "is_connected", _raise_is_connected)

    sup.connect(settle_secs=0)

    assert first_session.disconnect_calls == 1, (
        "an unreadable prior session must still be disconnected defensively"
    )
    assert len(created) == 2
    assert sup._session is created[1]


def test_connect_when_not_previously_connected_does_not_disconnect_anything(
    monkeypatch: Any,
) -> None:
    """The very first connect() call (self._session is None) must not touch
    disconnect() at all — no regression on the common, non-reentrant path."""
    import deephaven_ib

    created: list[_FakeIbSessionTws] = []
    monkeypatch.setattr(deephaven_ib, "IbSessionTws", _make_factory([True], created))
    monkeypatch.setattr("ibda.adapters.ibkr.supervisor.time.sleep", lambda _: None)

    sup = IbkrSupervisor()
    assert sup._session is None
    sup.connect(settle_secs=0)

    assert len(created) == 1
    assert created[0].disconnect_calls == 0
    assert sup._session is created[0]
