"""ibda.adapters.ibkr.orders — canonical working-order snapshot helper.

VENDOR-CONFINED: this module lives under ``adapters/ibkr/`` and may touch the
deephaven engine **only** through the deephaven adapter helpers
(``apply_canonical_view`` / ``snapshot_rows`` in
``ibda.adapters.deephaven.views``).  It MUST NOT import ``deephaven``
(engine) directly — those imports stay inside the deephaven adapter.

Public function:

``snapshot_working_orders(supervisor) -> pa.Table``
    Build (once, cached per session) a canonical ``order`` live view over the
    supervisor's ``orders_submitted`` raw table via ``apply_canonical_view``,
    snapshot it to row dicts on each call, and project those rows into a
    canonical ``ORDER`` Arrow table.

    NOTE: ``orders_submitted`` is only populated in read-write sessions
    (``read_only=False`` + TWS "Read-Only API" unchecked).  In read-only
    sessions the underlying table is empty so this function returns a zero-row
    table with the canonical ``ORDER`` schema.

Mirrors ``snapshot_executions`` in ``executions.py`` and ``snapshot_positions``
in ``positions.py``.
"""
from __future__ import annotations

import weakref
from typing import Any, cast

import pyarrow as pa

from ibda.adapters.ibkr.supervisor import IbkrSupervisor
from ibda.schema import ORDER

# Per-session cache of the canonical ``order`` live view.
# WeakKeyDictionary keys the cache on the session OBJECT (not id(session)), so
# entries are automatically evicted when the session is garbage-collected.  The
# previous dict[int, Any] keyed by id() was never evicted and risked serving a
# stale view if a new session happened to reuse a freed memory address.
_VIEW_CACHE: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()


def _canonical_order_view(supervisor: IbkrSupervisor) -> Any:
    """Return the cached canonical ``order`` live view for *supervisor*.

    Built once per session via the deephaven adapter's ``apply_canonical_view``
    over the raw ``orders_submitted`` table; cached keyed by the session OBJECT.
    The engine import lives inside the lazily-imported view helper.
    """
    session = supervisor._session
    view = _VIEW_CACHE.get(session)
    if view is None:
        from ibda.adapters.deephaven.views import apply_canonical_view  # noqa: PLC0415
        from ibda.adapters.ibkr.specs import CANONICAL_SPECS  # noqa: PLC0415

        spec = CANONICAL_SPECS["order"]
        raw = supervisor.raw_table(spec.raw_table)
        view = apply_canonical_view(raw, spec, ORDER)
        _VIEW_CACHE[session] = view
    return view


def _working_orders_arrow_from_rows(rows: list[dict[str, Any]]) -> pa.Table:
    """Project canonical ``order`` snapshot *rows* into an ``ORDER`` Arrow table.

    Pure (no engine, no JVM): *rows* are plain dicts keyed by canonical ORDER
    column names (as produced by snapshotting the canonical view).  An empty
    result yields a zero-row table with the canonical ``ORDER`` schema.

    Parameters
    ----------
    rows:
        Canonical order rows (keys are ORDER column names:
        OrderId, PermId, Account, Sym, Side, Qty, FilledQty,
        LimitPrice, Status, Timestamp).
    """
    columns: list[pa.Array] = []
    for col in ORDER.columns:
        columns.append(
            pa.array([r.get(col.name) for r in rows], type=col.dtype.to_arrow())
        )
    return cast(
        pa.Table,
        pa.table(
            dict(zip(ORDER.column_names, columns, strict=True)),
            schema=ORDER.to_arrow_schema(),
        ),
    )


def snapshot_working_orders(supervisor: IbkrSupervisor) -> pa.Table:
    """Snapshot the canonical ``order`` view to an ``ORDER`` Arrow table.

    Builds the canonical order live view once (cached per session), snapshots
    it to row dicts via the supervisor's injected snapshot helper, and projects
    the rows into a canonical ``ORDER`` Arrow table.

    In read-only sessions ``orders_submitted`` is empty, so this returns a
    zero-row table with the canonical ORDER schema.

    Parameters
    ----------
    supervisor:
        A connected IbkrSupervisor (e.g. wrapped via ``from_session``).

    Returns
    -------
    pa.Table
        A canonical ``ORDER`` Arrow table (all ORDER columns, schema order).
    """
    view = _canonical_order_view(supervisor)
    rows = supervisor._snapshot_rows(view)
    return _working_orders_arrow_from_rows(rows)
