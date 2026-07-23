"""ibda.adapters.deephaven.server — thin wrapper around deephaven_server.Server.

ENGINE-CONFINED: only adapters/deephaven/ may import deephaven_server.

Usage::

    from ibda.adapters.deephaven.server import start_server

    start_server()          # binds port 10000, default JVM opts
    start_server(port=9999, jvm_args=["-Xmx4g"])
"""
from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)

# Local-only pre-shared key for the embedded dev IDE, so an embedded Deephaven
# server exposes its IDE under a consistent token instead of a random
# per-process PSK. This default is for local development only — override it
# via the ``DH_PSK`` env var for anything beyond a local dev box.
DEFAULT_DH_PSK: str = os.environ.get("DH_PSK", "trading")


def start_server(
    port: int = 10000,
    jvm_args: Sequence[str] | None = None,
) -> Any:
    """Start (or reuse) an in-process Deephaven Server and return the Server object.

    This function is idempotent: if a ``deephaven_server.Server`` instance has
    already been started in the current process it is returned as-is and the
    *port* and *jvm_args* arguments are ignored.  This makes it safe to call
    from a notebook or script that may already have a live engine (e.g. when
    ``connect_live`` is called after a server was started elsewhere).

    Parameters
    ----------
    port:
        Port the Deephaven web API will listen on (default 10000).  Ignored
        when reusing an existing server.
    jvm_args:
        Extra JVM arguments (e.g. ``["-Xmx4g"]``).  When *None*, uses a
        safe default of ``["-Xmx2g"]``.  Ignored when reusing an existing
        server.

    Returns
    -------
    The running ``deephaven_server.Server`` instance (untyped; ``Any``).
    """
    from deephaven_server import Server

    # Fast path: reuse an already-running in-process server.
    existing: Any = getattr(Server, "instance", None)
    if existing is not None:
        logger.debug("start_server: reusing existing Server instance (port/jvm_args ignored)")
        return existing

    effective_args: list[str] = list(jvm_args) if jvm_args is not None else ["-Xmx2g"]
    # Ensure a shared, deterministic IDE token unless the caller already set one.
    if not any(a.startswith("-Dauthentication.psk=") for a in effective_args):
        effective_args.append(f"-Dauthentication.psk={DEFAULT_DH_PSK}")

    try:
        server = Server(port=port, jvm_args=effective_args)
        server.start()
    except Exception as exc:  # noqa: BLE001 — belt-and-suspenders: recover if already running
        recovered: Any = getattr(Server, "instance", None)
        if recovered is not None:
            logger.warning(
                "start_server: Server() raised %r but an instance already exists; reusing it",
                exc,
            )
            return recovered
        raise

    return server
