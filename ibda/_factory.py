"""ibda._factory — wiring that selects the (hidden) engine adapter.

The deephaven import is done lazily inside connect() so that `import ibda`
never loads the engine (enforced by test_public_surface).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ibda.port import DataPort


def connect(tables: Mapping[str, Any]) -> DataPort:
    """Return a DataPort backed by the default (hidden) compute engine.

    `tables` maps canonical table names to engine-native live tables. In SP2 the
    IBKR adapter supplies these; in tests, fixtures do. The return type is the
    abstract DataPort — callers never name or see the engine.
    """
    from ibda.adapters.deephaven.adapter import DeephavenPort  # noqa: PLC0415 — lazy: keep engine out of import

    return DeephavenPort(tables)
