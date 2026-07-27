"""ibda.adapters.ibkr.positions — canonical position snapshot helper.

VENDOR-CONFINED: this module lives under ``adapters/ibkr/`` and may touch the
deephaven engine **only** through the deephaven adapter helpers
(``apply_canonical_view`` / ``snapshot_rows`` in
``ibda.adapters.deephaven.views``).  It MUST NOT import ``deephaven``
(engine) directly — those imports stay inside the deephaven adapter, which is
why the engine import is lazy/indirect here (the boundary guard only allows
``deephaven``/``pydeephaven`` roots under ``adapters/deephaven/``).

Public function:

``snapshot_positions(supervisor, *, account=None) -> pa.Table``
    Build (once, cached per session) a canonical ``position`` live view over the
    supervisor's ``accounts_positions`` raw table via ``apply_canonical_view``,
    snapshot it to row dicts on each call, and project those rows into a
    canonical ``POSITION`` Arrow table.  ``account`` optionally filters the rows
    to a single account.  ``MarketPrice``/``MarketValue``/``UnrealizedPnl`` are
    null-filled (the raw table does not carry them); ``SecType`` is populated.

The live-view construction (``apply_canonical_view``) requires the JVM/engine,
so it is exercised by the paper flip-compare; the pure Arrow-assembly +
account-filter logic is factored into ``_positions_arrow_from_rows`` and
unit-tested with canned rows (no JVM).
"""
from __future__ import annotations

import weakref
from typing import Any, cast

import pyarrow as pa

from ibda.adapters.ibkr.supervisor import IbkrSupervisor
from ibda.schema import POSITION

# Per-session cache of the built canonical ``position`` live view.
# WeakKeyDictionary keys the cache on the session OBJECT (not id(session)), so
# entries are automatically evicted when the session is garbage-collected.  The
# previous dict[int, Any] keyed by id() was never evicted and risked serving a
# stale view if a new session happened to reuse a freed memory address.
_VIEW_CACHE: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()


def _canonical_position_view(supervisor: IbkrSupervisor) -> Any:
    """Return the cached canonical ``position`` live view for *supervisor*.

    Built once per session via the deephaven adapter's ``apply_canonical_view``
    over the raw ``accounts_positions`` table; cached in a ``WeakKeyDictionary``
    keyed by the session OBJECT itself (not ``id(session)`` — see the module
    comment above ``_VIEW_CACHE``), so entries are evicted automatically when
    the session is garbage-collected.  The engine import lives inside the
    lazily-imported view helper, not here.
    """
    session = supervisor._session
    view = _VIEW_CACHE.get(session)
    if view is None:
        # Lazy import: the deephaven engine import lives inside this adapter
        # module (adapters/deephaven), keeping this vendor module engine-free.
        from ibda.adapters.deephaven.views import apply_canonical_view  # noqa: PLC0415
        from ibda.adapters.ibkr.specs import CANONICAL_SPECS  # noqa: PLC0415

        spec = CANONICAL_SPECS["position"]
        raw = supervisor.raw_table(spec.raw_table)
        view = apply_canonical_view(raw, spec, POSITION)
        _VIEW_CACHE[session] = view
    return view


def _positions_arrow_from_rows(
    rows: list[dict[str, Any]],
    account: str | None = None,
) -> pa.Table:
    """Project canonical ``position`` snapshot *rows* into a ``POSITION`` Arrow table.

    Pure (no engine, no JVM): *rows* are plain dicts keyed by canonical POSITION
    column names (as produced by snapshotting the canonical view).  When
    *account* is given, only rows whose ``Account`` equals it are kept.  An empty
    result yields a zero-row table with the canonical ``POSITION`` schema.

    Parameters
    ----------
    rows:
        Canonical position rows (keys are POSITION column names).
    account:
        Optional account id to filter rows by (``Account == account``).
    """
    selected = (
        [r for r in rows if r.get("Account") == account]
        if account is not None
        else rows
    )
    columns: list[pa.Array] = []
    for col in POSITION.columns:
        columns.append(
            pa.array([r.get(col.name) for r in selected], type=col.dtype.to_arrow())
        )
    return cast(
        pa.Table,
        pa.table(
            dict(zip(POSITION.column_names, columns, strict=True)),
            schema=POSITION.to_arrow_schema(),
        ),
    )


def snapshot_positions(
    supervisor: IbkrSupervisor,
    *,
    account: str | None = None,
) -> pa.Table:
    """Snapshot the canonical ``position`` view to a ``POSITION`` Arrow table.

    Builds the canonical position live view once (cached per session), snapshots
    it to row dicts via the supervisor's injected snapshot helper, and projects
    the rows into a canonical ``POSITION`` Arrow table (``SecType`` populated;
    ``MarketPrice``/``MarketValue``/``UnrealizedPnl`` null-filled as today).

    Parameters
    ----------
    supervisor:
        A connected IbkrSupervisor (e.g. wrapped via ``from_session``).
    account:
        Optional account id; when given, only rows for that account are returned.

    Returns
    -------
    pa.Table
        A canonical ``POSITION`` Arrow table (all POSITION columns, schema order).
    """
    view = _canonical_position_view(supervisor)
    rows = supervisor._snapshot_rows(view)
    return _positions_arrow_from_rows(rows, account)
