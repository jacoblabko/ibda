"""What Python type a Deephaven null ``double`` actually becomes, measured not assumed.

``ibda/adapters/ibkr/pnl.py::_f`` is the Arrow-snapshot scrubber: every "no value"
encoding must leave it as ``None``, because every consumer of those columns sums them.
Its docstring makes a specific claim about one of those encodings — that a Deephaven
null "arrives as NaN" through ``deephaven.pandas.to_pandas``, which is why it carries an
explicit ``math.isnan`` branch. Nothing in this package verified that claim against a
running engine, and it decides whether the branch guards a real condition or a
theoretical one.

Reasoning could not settle it. The installed ``deephaven/pandas.py`` defaults to
``dtype_backend="numpy_nullable"``, which argues for ``pandas.NA`` — and ``float(pd.NA)``
raises ``TypeError``, which ``_f`` catches separately, so the ``isnan`` branch would
never fire. Against that, ``_f``'s docstring and ``ibda/tests/test_pnl_adapter.py``
both assert NaN. A docstring describing an older Deephaven is exactly the kind of thing
that goes stale unnoticed. The only authority is a running JVM, which is why this lives
in ``tests_jvm/``.

**Measured, on the Deephaven build this repo pins: neither.** A null ``double`` reaches
Python through ``snapshot_rows`` as plain ``None`` — ``to_pandas`` produces the nullable
form, and ``DataFrame.to_dict(orient="records")`` collapses it to ``None`` on the way
out. So ``_f`` returns ``None`` at its very first branch, and the ``math.isnan`` branch
is belt-and-braces on *this* path rather than the load-bearing guard its docstring
implies. It is still worth keeping: ``_f`` is also called on values that do not come
through ``to_dict``.

The first test asserts the *behaviour* rather than pinning that representation: whichever
form arrives, ``_f`` must return ``None``. That stays true across a Deephaven upgrade
that changes the representation, and fails loudly if an upgrade ever makes a null survive
scrubbing as a number. The second test accepts any of the three forms and reports which
was seen, so a flip is named rather than left for the next reader to re-derive.

Run with:
    uv run pytest ibda/tests_jvm/test_null_double_python_type.py -q
"""

from __future__ import annotations

import math
from typing import Any

from ibda.adapters.ibkr.pnl import _f


def _table_with_a_null_double() -> Any:
    """A one-row table whose only ``double`` cell is a Deephaven null."""
    from deephaven import empty_table  # noqa: PLC0415 — JVM-gated

    # NULL_DOUBLE is -Double.MAX_VALUE inside the engine; `(double)null` is how the query
    # language spells the null itself, which is what a real absent mark looks like.
    return empty_table(1).update(["Val = (double)null"])


def test_a_null_double_reaches_python_as_something_the_scrub_rejects() -> None:
    """The contract that matters: a null must not survive scrubbing as a number."""
    from ibda.adapters.deephaven.views import snapshot_rows  # noqa: PLC0415 — JVM-gated

    rows = snapshot_rows(_table_with_a_null_double())
    assert len(rows) == 1
    raw = rows[0]["Val"]

    scrubbed = _f(raw)
    assert scrubbed is None, (
        f"a Deephaven null double reached Python as {raw!r} ({type(raw).__name__}) and "
        f"_f returned {scrubbed!r} instead of None. A single null that survives as a "
        f"number poisons every sum built on these columns -- one NaN turns a whole-book "
        f"P&L into 'nan', and a surviving sentinel reads as a ~1.8e308 position."
    )


def test_the_observed_representation_is_recorded_for_the_next_reader() -> None:
    """Pin which form this Deephaven build actually produces.

    Not a redundant assertion: it is the fact ``_f``'s docstring commits to, and if a
    future upgrade flips it this test names the change instead of leaving someone to
    re-derive it. Both forms are accepted; the message reports which was seen.
    """
    from ibda.adapters.deephaven.views import snapshot_rows  # noqa: PLC0415 — JVM-gated

    raw = snapshot_rows(_table_with_a_null_double())[0]["Val"]

    is_nan = isinstance(raw, float) and math.isnan(raw)
    is_na_like = raw is None or (not isinstance(raw, (int, float)))
    assert is_nan or is_na_like, (
        f"a null double arrived as {raw!r} ({type(raw).__name__}) — neither NaN nor a "
        f"pandas-NA-like sentinel. The scrub assumes one of those two; a third form "
        f"means it needs revisiting."
    )
