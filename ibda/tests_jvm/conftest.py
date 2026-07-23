"""JVM-backed fixtures for ibda adapter tests.

Constructs exactly one Deephaven Server for the whole test session (only one
Server may exist per process). Run with: uv run pytest ibda/tests_jvm
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any

import pytest


def _free_port() -> int:
    """Return an OS-assigned free TCP port.

    Binds a socket to port 0, reads the assigned ephemeral port, then closes
    the socket.  The port is free at the instant of return; there is a brief
    benign race window before the Server binds it, which is acceptable for
    test-fixture use.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="session", autouse=True)
def _dh_server() -> Iterator[Any]:
    from deephaven_server import Server

    port = _free_port()
    server = Server(port=port, jvm_args=["-Xmx2g"])
    server.start()
    yield server
    # deephaven_server.Server exposes no public stop() / close() API — the
    # embedded JVM terminates with the Python process.  The ephemeral port
    # selected above is what prevents back-to-back test-run collisions.
