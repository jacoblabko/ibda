"""ibda.adapters.deephaven.builders — canonical rows -> deephaven Table.

ENGINE-CONFINED: this is the only new file outside adapter.py that may import
deephaven. The boundary guard enforces this.

Usage::

    from ibda.schema import EXECUTION
    from ibda.adapters.deephaven.builders import table_from_rows

    dh_table = table_from_rows(EXECUTION, rows)

The returned object is a live deephaven Table. Consumers must not type-hint it
as anything other than ``Any`` (deephaven is fully untyped in this project).

Implementation notes
--------------------
deephaven is configured as ``follow_imports = "skip"`` in pyproject.toml, so
mypy treats all deephaven symbols as ``Any`` — no per-call ``# type: ignore``
annotations are needed or permitted (unused ignores are errors under strict).
Correctness is verified by the JVM test in ibda/tests_jvm/test_flex_loader.py.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ibda.schema._core import DType, Schema


def _format_instant_iso(dt: datetime) -> str:
    """Return an ISO-8601 string with microseconds for Deephaven's ``to_j_instant``.

    This is the pure-Python portion of :func:`_to_j_instant`, extracted so it
    can be unit-tested without a JVM.  The ``%f`` directive produces 6 decimal
    digits (microseconds), preserving sub-second precision in the
    ``TIMESTAMP_NS`` column — no truncation to whole seconds.

    Naive datetimes are treated as UTC per the project convention.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f UTC")


def _to_j_instant(dt: datetime) -> Any:
    """Convert a Python datetime to a deephaven Instant.

    Deephaven's ``to_j_instant`` accepts ISO-8601 strings with a UTC suffix.
    The ISO string is produced by :func:`_format_instant_iso`, which includes
    microseconds (``%f``) to preserve sub-second precision.  The canonical
    DType is ``TIMESTAMP_NS``; Python ``datetime`` has microsecond resolution,
    so this avoids silently truncating to whole seconds.

    Note: correctness with Deephaven's ``to_j_instant`` is verified by the JVM
    contract tests (``tests_jvm/test_adapter_contract.py``).  The pure-Python
    ISO-formatting behaviour is covered by unit tests that import
    :func:`_format_instant_iso` directly (no JVM required).
    """
    from deephaven.time import to_j_instant

    return to_j_instant(_format_instant_iso(dt))


def table_from_rows(schema: Schema, rows: list[dict[str, Any]]) -> Any:
    """Build a deephaven Table from a list of canonical row dicts.

    Parameters
    ----------
    schema:
        The canonical Schema describing expected columns and their types.
    rows:
        List of dicts, each keyed by canonical column names. Missing keys are
        treated as None for that row. Extra keys are ignored.

    Returns
    -------
    A deephaven Table (untyped; ``Any``). An empty Table with correct columns
    is returned when ``rows`` is empty.
    """
    from deephaven import new_table
    from deephaven.column import (
        bool_col,
        datetime_col,
        double_col,
        long_col,
        string_col,
    )

    col_objects: list[Any] = []

    for col in schema.columns:
        values: list[Any] = [row.get(col.name) for row in rows]

        if col.dtype == DType.STRING:
            # None stays None (nullable string column)
            col_objects.append(string_col(col.name, values))

        elif col.dtype == DType.INT64:
            # deephaven long_col accepts None for nullable longs (NULL_LONG sentinel)
            col_objects.append(long_col(col.name, values))

        elif col.dtype == DType.FLOAT64:
            # deephaven double_col accepts None -> NULL_DOUBLE
            col_objects.append(double_col(col.name, values))

        elif col.dtype == DType.BOOL:
            col_objects.append(bool_col(col.name, values))

        elif col.dtype == DType.TIMESTAMP_NS:
            # Convert each datetime value to a Java Instant; None -> None (null)
            instants: list[Any] = [
                _to_j_instant(v) if isinstance(v, datetime) else None
                for v in values
            ]
            col_objects.append(datetime_col(col.name, instants))

        else:  # pragma: no cover — exhaustive over DType members
            raise ValueError(f"unsupported DType {col.dtype!r} for column {col.name!r}")

    return new_table(col_objects)
