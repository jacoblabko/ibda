"""JVM regression test for a prior regression — IbkrSupervisor.connect() leaks a
JNI session on every failed retry.

Root cause (verified by reading connect()): the local ``session = dhib.IbSessionTws(...)`` was only assigned
to ``self._session`` AFTER the post-connect connectivity check passed. On the
failure branch (``not session.is_connected()``), the method raised without
ever calling ``session.disconnect()`` — the locally-created, JNI-backed
session (open socket + native threads) was dropped with no cleanup. The
retry loop in the feed calls ``connect()`` every 2-30s, so each failed
attempt leaked one native session, which (unlike a Java heap leak) does not
show up as OOM/hs_err — it grows RSS until the OS/cgroup SIGKILLs the
process, matching every symptom of the observed silent death.

Fix: ``self._session`` is now assigned immediately after construction
(before ``connect()``/the connectivity check), inside a ``try/finally`` that
calls the existing idempotent ``self.disconnect()`` on any path that does
not reach ``connected = True`` — covering both the connectivity-check
failure AND ``session.connect()`` itself raising.

``deephaven_ib`` requires the embedded Deephaven server to be constructed
before it can be imported (bare ``import deephaven_ib`` outside a JVM-backed
test raises "The Deephaven Server has not been initialized"), so this test
must live under ``ibda/tests_jvm/`` even though it never touches a real
Deephaven Table — it only needs the real, JVM-backed ``deephaven_ib`` module
object to monkeypatch ``IbSessionTws`` on.

Run with:
    uv run pytest ibda/tests_jvm/test_supervisor_connect_leak_jvm.py -q
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from ibda.adapters.ibkr.supervisor import IbkrSupervisor


# ---------------------------------------------------------------------------
# Fake deephaven_ib.IbSessionTws
# ---------------------------------------------------------------------------


class _FakeIbSessionTws:
    """Fakes the IbSessionTws surface touched by IbkrSupervisor.connect().

    Each instance is appended to *created* (in construction order) so a test
    can inspect every session connect() ever built, including ones that were
    never assigned to self._session. ``_connected`` is drawn from *outcomes*
    by construction index, simulating a scripted sequence of connect
    attempts (e.g. two failures then a success).
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


class _FakeIbSessionTwsConnectRaises(_FakeIbSessionTws):
    """Fails inside connect() itself, not just the post-check — proves the
    fix's try/finally covers this path too, not only the is_connected()
    check."""

    def connect(self) -> None:
        super().connect()
        raise ConnectionError("simulated TWS handshake failure")


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


def test_connect_n_failed_attempts_leak_zero_sessions(monkeypatch: Any) -> None:
    """N consecutive failed connect() calls must each disconnect their own
    created session exactly once — the zero-leak guarantee from the fix above."""
    import deephaven_ib

    n = 3
    outcomes = [False] * n
    created: list[_FakeIbSessionTws] = []
    monkeypatch.setattr(deephaven_ib, "IbSessionTws", _make_factory(outcomes, created))
    monkeypatch.setattr("ibda.adapters.ibkr.supervisor.time.sleep", lambda _: None)

    sup = IbkrSupervisor()
    for _ in range(n):
        with pytest.raises(RuntimeError, match="disconnected immediately after connect"):
            sup.connect(settle_secs=0)

    assert len(created) == n, "one session object should be constructed per attempt"
    assert all(s.disconnect_calls == 1 for s in created), (
        "every failed-attempt session must be disconnected exactly once — "
        f"got {[s.disconnect_calls for s in created]}"
    )
    assert sup._session is None, "supervisor must not hold a reference to a failed session"


def test_connect_disconnects_when_session_connect_itself_raises(monkeypatch: Any) -> None:
    """The failure branch also covers session.connect() raising directly
    (not just the post-connect is_connected() check failing) — the created
    session must still be disconnected and the real error must propagate."""
    import deephaven_ib

    created: list[_FakeIbSessionTws] = []
    monkeypatch.setattr(
        deephaven_ib,
        "IbSessionTws",
        _make_factory([False], created, session_cls=_FakeIbSessionTwsConnectRaises),
    )
    monkeypatch.setattr("ibda.adapters.ibkr.supervisor.time.sleep", lambda _: None)

    sup = IbkrSupervisor()
    with pytest.raises(ConnectionError, match="simulated TWS handshake failure"):
        sup.connect(settle_secs=0)

    assert len(created) == 1
    assert created[0].disconnect_calls == 1, "session must be disconnected even when connect() itself raises"
    assert sup._session is None


def test_connect_success_does_not_disconnect_and_leaves_session_set(monkeypatch: Any) -> None:
    """A successful connect() must NOT call disconnect() on the session and
    must leave self._session pointing at it."""
    import deephaven_ib

    created: list[_FakeIbSessionTws] = []
    monkeypatch.setattr(deephaven_ib, "IbSessionTws", _make_factory([True], created))
    monkeypatch.setattr("ibda.adapters.ibkr.supervisor.time.sleep", lambda _: None)

    sup = IbkrSupervisor()
    sup.connect(settle_secs=0)

    assert len(created) == 1
    assert created[0].disconnect_calls == 0, "a successful connect must not disconnect its session"
    assert sup._session is created[0]
    assert sup._last_error == ""


def test_connect_retries_then_succeeds_only_disconnects_failed_attempts(monkeypatch: Any) -> None:
    """Two failures followed by a success: the first two sessions are each
    disconnected once; the third (successful) session is never disconnected
    and becomes self._session — mirrors the real feed's backoff retry loop."""
    import deephaven_ib

    outcomes = [False, False, True]
    created: list[_FakeIbSessionTws] = []
    monkeypatch.setattr(deephaven_ib, "IbSessionTws", _make_factory(outcomes, created))
    monkeypatch.setattr("ibda.adapters.ibkr.supervisor.time.sleep", lambda _: None)

    sup = IbkrSupervisor()
    for _ in range(2):
        with pytest.raises(RuntimeError, match="disconnected immediately after connect"):
            sup.connect(settle_secs=0)
    sup.connect(settle_secs=0)

    assert len(created) == 3
    assert created[0].disconnect_calls == 1
    assert created[1].disconnect_calls == 1
    assert created[2].disconnect_calls == 0
    assert sup._session is created[2]
