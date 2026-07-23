"""Adapters: the only place in ``ibda`` allowed to touch a third-party engine or vendor SDK.

The package is split so that the rest of ``ibda`` stays free of both Deephaven and IBKR imports,
which is what makes the core carve-able. A pre-commit hook (``tools/check_ibda_boundary.py``)
enforces exactly three rules:

1. Only ``ibda/adapters/deephaven/`` may import ``deephaven*`` / ``pydeephaven``.
2. Only ``ibda/adapters/ibkr/`` may import ``ibapi`` / ``deephaven_ib``.
3. **No** ``ibda`` module may import first-party code from outside this package — the
   dependency is one-way, always.

``ibda/tests`` and ``ibda/tests_jvm`` are exempt. If a new adapter needs a fourth engine, it gets
its own subpackage and its own rule — do not widen an existing one.

This boundary was decided deliberately and has been enforced since the package's design.
"""

from __future__ import annotations
