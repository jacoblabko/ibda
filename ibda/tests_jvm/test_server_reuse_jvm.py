"""JVM regression test for a suspected silent-death mechanism in a long-lived feed
consumer.

A recoverable connect failure in a long-lived feed consumer was once suspected to
escalate into a silent process death, caused by ``connect_live`` constructing a
SECOND ``deephaven_server.Server`` on each retry (only one may exist per process).
``connect_live`` calls ``ibda.adapters.deephaven.server.start_server`` on every
(re)connect attempt — see a typical feed consumer's backoff-retry loop
(``connect_tws`` -> ``connect_live`` -> ``start_server``).

This test exercises ``start_server`` directly against a REAL running JVM
(the session-scoped ``_dh_server`` fixture in this directory's ``conftest.py``
already constructed one ``Server``) and confirms repeated calls — simulating
several retry-loop passes — reuse that same instance rather than attempting a
second construction. It does not (and cannot, without a live TWS) reproduce
the full silent-death scenario; it isolates and confirms the one piece of the
suspected mechanism that a JVM-backed test can actually verify.
"""
from __future__ import annotations

from typing import Any


def test_start_server_reuses_existing_instance_under_real_jvm(_dh_server: Any) -> None:
    """start_server() must return the SAME Server instance across repeated calls.

    Simulates several retry-loop passes (a long-lived feed consumer calls
    connect_live, which calls start_server, on every reconnect attempt). No second
    ``deephaven_server.Server(...)`` construction — and therefore no risk of the
    suspected abort — occurs on any of these calls.
    """
    from deephaven_server import Server

    from ibda.adapters.deephaven.server import start_server

    existing = Server.instance
    assert existing is not None, (
        "the session-scoped _dh_server fixture should have already constructed a Server"
    )

    for _ in range(5):  # simulate several retry-loop passes
        reused = start_server(port=10000)
        assert reused is existing, "start_server must reuse, not reconstruct, the existing Server"


def test_server_reconstruction_raises_catchable_error_not_jvm_abort(_dh_server: Any) -> None:
    """A genuine second ``deephaven_server.Server(...)`` construction (bypassing
    start_server's reuse check entirely) must raise a catchable Python exception,
    not abort the JVM. This directly falsifies the "second Server() aborts the
    JVM" half of the suspected mechanism above: this test process is still alive
    and able to make further assertions after the raise.
    """
    from deephaven_server import Server

    raised: Exception | None = None
    try:
        Server(port=19999, jvm_args=["-Xmx256m"])
    except Exception as exc:  # noqa: BLE001 — asserting on the raised exception itself
        raised = exc

    assert raised is not None, "a second Server() construction should have raised"
    # The process is still alive and the original Server instance is untouched —
    # proof the raise did not abort the JVM.
    assert Server.instance is not None
