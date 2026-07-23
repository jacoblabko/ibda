"""Pure tests for DeephavenPort.group_by's `how` validation (Fix #2).

No JVM: the `how` vocabulary is validated BEFORE `deephaven.agg` is imported or
`result.handle` is ever touched, so an unsupported `how` raises `ValueError`
without needing a live engine. JVM-level proof that the *supported* aggregations
(sum/mean/last/count) actually compute correctly lives in
ibda/tests_jvm/test_adapter_contract.py.
"""
from __future__ import annotations

import pytest

from ibda.adapters.deephaven.adapter import DeephavenPort
from ibda.result import Result
from ibda.schema import ALL as ALL_SCHEMAS


def _fake_result() -> Result:
    """A Result wrapping an opaque handle — group_by's validation gate must
    raise before `result.handle` is ever inspected, so a plain sentinel suffices."""
    canonical_name = next(iter(ALL_SCHEMAS))
    port = DeephavenPort({canonical_name: object()})
    return port.table(canonical_name)


def test_group_by_unsupported_how_raises_valueerror_not_keyerror() -> None:
    """Before the fix, `_list_fn[how]` raised a bare KeyError for an unsupported
    `how` (e.g. 'max'), leaking an internal dispatch-table detail through the
    DataPort abstraction. It must now raise ValueError."""
    port = DeephavenPort({})
    result = _fake_result()

    with pytest.raises(ValueError) as exc_info:
        port.group_by(result, by=["Sym"], aggs={"Qty": "max"})

    assert not isinstance(exc_info.value, KeyError)


def test_group_by_unsupported_how_names_the_bad_value_and_supported_set() -> None:
    port = DeephavenPort({})
    result = _fake_result()

    with pytest.raises(ValueError, match=r"'max'"):
        port.group_by(result, by=["Sym"], aggs={"Qty": "max"})

    with pytest.raises(ValueError, match=r"supported.*count.*last.*mean.*sum"):
        port.group_by(result, by=["Sym"], aggs={"Qty": "max"})


@pytest.mark.parametrize("how", ["sum", "mean", "last", "count"])
def test_group_by_supported_hows_do_not_raise_validation_error(how: str) -> None:
    """The four supported values must pass the validation gate (they will still
    fail past that point without a JVM/real handle, but NOT with the ValueError
    this fix introduces).

    Order-independent (Fix I): what happens PAST the gate depends on whether a
    Deephaven Server is already up in this pytest process --
      - no Server yet: `from deephaven import agg` raises
        ``RuntimeError("... Deephaven Server has not been initialized ...")``.
      - a Server already booted earlier in the same process (e.g. another
        JVM-backed test suite that constructs one in its conftest): the
        import succeeds and dispatch reaches
        ``result.handle.agg_by(...)`` on the sentinel ``object()`` handle
        from ``_fake_result()``, raising ``AttributeError``.
    Either way proves the gate accepted `how` — it is only the code PAST the
    gate that differs by process state. So this test asserts the one property
    it actually owns: no ``ValueError`` (the gate's "unsupported aggregation"
    rejection) is raised. Any other exception is caught, asserted to be one of
    the two known no-JVM / has-JVM shapes above, and otherwise re-raised
    (an unrecognized exception type is a real signal worth surfacing, not
    something to silently swallow).
    """
    port = DeephavenPort({})
    result = _fake_result()

    try:
        port.group_by(result, by=["Sym"], aggs={"Qty": how})
    except ValueError as exc:
        pytest.fail(
            f"how={how!r} should have passed the validation gate but was "
            f"rejected with ValueError: {exc}"
        )
    except (RuntimeError, AttributeError):
        # Gate accepted `how`; failure past that point is the expected
        # no-JVM / has-JVM-but-sentinel-handle shape, not the gate's own
        # ValueError -- exactly what this test verifies.
        pass
