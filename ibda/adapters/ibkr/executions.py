"""ibda.adapters.ibkr.executions — canonical execution snapshot helper.

VENDOR-CONFINED: this module lives under ``adapters/ibkr/`` and may touch the
deephaven engine **only** through the deephaven adapter helpers
(``apply_canonical_view`` / ``snapshot_rows`` in
``ibda.adapters.deephaven.views``).  It MUST NOT import ``deephaven``
(engine) directly — those imports stay inside the deephaven adapter, which is
why the engine import is lazy/indirect here.

Public function:

``snapshot_executions(supervisor) -> pa.Table``
    Build (once, cached per session) a canonical ``execution`` live view over the
    supervisor's ``orders_exec_details`` raw table, joined against
    ``orders_exec_commission_report`` via ``apply_canonical_view``, snapshot it to
    row dicts on each call, and project those rows into a canonical ``EXECUTION``
    Arrow table.  ``Commission``/``RealizedPnl`` come from the join (null for a
    fill whose commission report has not arrived — e.g. a read-only session,
    where the commission table stays empty); ``Liquidity`` is null-filled (no
    source in deephaven-ib); ``OrderId`` is populated.

Mirrors ``snapshot_positions`` in ``positions.py``.  The live-view construction
(``apply_canonical_view``) requires the JVM/engine, so it is exercised by the
paper flip-compare; the pure Arrow-assembly logic is factored into
``_executions_arrow_from_rows`` and unit-tested with canned rows (no JVM).
"""
from __future__ import annotations

import weakref
from typing import Any, cast

import pyarrow as pa

from ibda.adapters.ibkr.supervisor import IbkrSupervisor
from ibda.schema import EXECUTION

# Per-session cache of the built canonical ``execution`` live view.
# WeakKeyDictionary keys the cache on the session OBJECT (not id(session)), so
# entries are automatically evicted when the session is garbage-collected.  The
# previous dict[int, Any] keyed by id() was never evicted and risked serving a
# stale view if a new session happened to reuse a freed memory address.
_VIEW_CACHE: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()


def _canonical_execution_view(supervisor: IbkrSupervisor) -> Any:
    """Return the cached canonical ``execution`` live view for *supervisor*.

    Built once per session via the deephaven adapter's ``apply_canonical_view``
    over the raw ``orders_exec_details`` table, joined against
    ``orders_exec_commission_report`` (per the ``execution`` spec's
    ``join_table``) so Commission/RealizedPnl populate; cached keyed by the
    session OBJECT.  The engine import lives inside the lazily-imported view
    helper.
    """
    session = supervisor._session
    view = _VIEW_CACHE.get(session)
    if view is None:
        from ibda.adapters.deephaven.views import apply_canonical_view  # noqa: PLC0415
        from ibda.adapters.ibkr.specs import CANONICAL_SPECS  # noqa: PLC0415

        spec = CANONICAL_SPECS["execution"]
        raw = supervisor.raw_table(spec.raw_table)
        join_raw = supervisor.raw_table(spec.join_table) if spec.join_table else None
        view = apply_canonical_view(raw, spec, EXECUTION, join_raw=join_raw)
        _VIEW_CACHE[session] = view
    return view


def _executions_arrow_from_rows(rows: list[dict[str, Any]]) -> pa.Table:
    """Project canonical ``execution`` snapshot *rows* into an ``EXECUTION`` Arrow table.

    Pure (no engine, no JVM): *rows* are plain dicts keyed by canonical EXECUTION
    column names (as produced by snapshotting the canonical view).  An empty
    result yields a zero-row table with the canonical ``EXECUTION`` schema.

    Parameters
    ----------
    rows:
        Canonical execution rows (keys are EXECUTION column names).
    """
    columns: list[pa.Array] = []
    for col in EXECUTION.columns:
        columns.append(
            pa.array([r.get(col.name) for r in rows], type=col.dtype.to_arrow())
        )
    return cast(
        pa.Table,
        pa.table(
            dict(zip(EXECUTION.column_names, columns, strict=True)),
            schema=EXECUTION.to_arrow_schema(),
        ),
    )


def snapshot_executions(supervisor: IbkrSupervisor) -> pa.Table:
    """Snapshot the canonical ``execution`` view to an ``EXECUTION`` Arrow table.

    Builds the canonical execution live view once (cached per session), snapshots
    it to row dicts via the supervisor's injected snapshot helper, and projects the
    rows into a canonical ``EXECUTION`` Arrow table (``OrderId`` populated;
    ``Commission``/``Liquidity`` null-filled as today).

    Parameters
    ----------
    supervisor:
        A connected IbkrSupervisor (e.g. wrapped via ``from_session``).

    Returns
    -------
    pa.Table
        A canonical ``EXECUTION`` Arrow table (all EXECUTION columns, schema order).
    """
    view = _canonical_execution_view(supervisor)
    rows = supervisor._snapshot_rows(view)
    return _executions_arrow_from_rows(rows)
