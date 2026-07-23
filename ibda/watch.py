"""ibda.watch — no-poll callbacks over the data surface.

watch(result, callback, predicate=None, once=False) registers `callback` to be
invoked with an Arrow table of newly-appearing rows that match `predicate`. It
is sugar over the port's filter + Result.subscribe; no engine import.

Typing note: ``result.port`` is typed as ``_PortLike``, which includes ``filter``
(added to the Protocol in result.py). This keeps watch.py engine-agnostic —
no import of DataPort or any adapter is needed.
"""

from __future__ import annotations

import pyarrow as pa

from ibda.result import DeltaCallback, Result, Stream


def watch(
    result: Result,
    callback: DeltaCallback,
    *,
    predicate: str | None = None,
    once: bool = False,
) -> Stream:
    """Call ``callback(arrow_table)`` when rows matching ``predicate`` appear in ``result``.

    Args:
        result: the query to watch (e.g. ``port.table("execution")``).
        callback: invoked with an Arrow table of the newly-added matching rows.
        predicate: optional engine-neutral boolean filter (e.g. ``"Last > 150"``);
            when given, only matching rows trigger the callback.
        once: when True, the subscription auto-cancels after the first delivery.

    Returns:
        A Stream; call ``.cancel()`` to stop receiving callbacks.
    """
    target = result.port.filter(result, predicate) if predicate is not None else result

    if not once:
        return target.subscribe(callback)

    # once=True: cancel after the first delivery. Hold the Stream in a mutable
    # cell so the wrapper can cancel the subscription it is itself part of.
    box: dict[str, Stream | None] = {"stream": None}
    fired: dict[str, bool] = {"done": False}

    def _wrapped(tbl: pa.Table) -> None:
        if fired["done"]:
            return
        fired["done"] = True
        callback(tbl)
        stream = box["stream"]
        if stream is not None:
            stream.cancel()

    stream = target.subscribe(_wrapped)
    box["stream"] = stream
    return stream
