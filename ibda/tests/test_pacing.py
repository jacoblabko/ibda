"""Tests for ibda.adapters.ibkr.pacing — fixed-interval PacingGovernor.

All tests use a fake (injected) clock so they are deterministic and fast.
No real time.sleep() calls are made.
"""
from __future__ import annotations

import pytest

from ibda.adapters.ibkr.pacing import PacingGovernor


class FakeClock:
    """Monotonic fake clock: advances by explicit delta."""

    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def advance(self, delta: float) -> None:
        self.t += delta

    def now(self) -> float:
        return self.t


# ---------------------------------------------------------------------------
# Basic rate enforcement at max_hz=5 (interval = 0.2s)
# ---------------------------------------------------------------------------

def test_first_request_proceeds_immediately() -> None:
    gov = PacingGovernor(max_hz=5.0)
    sched = gov.acquire(now=0.0)
    assert sched == 0.0


def test_sequential_requests_are_spaced_by_interval() -> None:
    gov = PacingGovernor(max_hz=5.0)
    interval = 1.0 / 5.0  # 0.2s
    t = 0.0
    schedules = [gov.acquire(now=t) for _ in range(5)]
    for i, s in enumerate(schedules):
        assert abs(s - i * interval) < 1e-12, f"schedule[{i}] = {s}, expected {i * interval}"


def test_wait_for_returns_zero_when_no_prior_request() -> None:
    gov = PacingGovernor(max_hz=5.0)
    assert gov.wait_for(0.0) == 0.0


def test_wait_for_returns_positive_after_saturated_burst() -> None:
    gov = PacingGovernor(max_hz=5.0)
    # Issue 3 requests at t=0 (all "at once" from the governor's perspective)
    for _ in range(3):
        gov.acquire(now=0.0)
    # 4th request at t=0 must wait at least 3 * interval
    wait = gov.wait_for(0.0)
    assert wait >= 3.0 / 5.0 - 1e-12


def test_gap_larger_than_interval_resets_schedule() -> None:
    gov = PacingGovernor(max_hz=5.0)
    gov.acquire(now=0.0)
    # After a large gap the governor should not accumulate "credit"
    sched = gov.acquire(now=10.0)
    assert sched == 10.0

    # This read `assert gov.wait_for(10.0 + 1e-9) == 0.0 or True`, and the trailing comment
    # ("next after 10.0 is 10.2") said outright that the disabled assertion was FALSE. The
    # slot at 10.0 is taken, so the next one is 10.0 + 1/5 = 10.2 and wait_for returns ~0.2.
    # Asserting the real value is what proves the point the test is named for: no credit
    # accumulated over the gap (a governor that banked the idle time would return 0.0 here
    # and let a burst through), and the spacing is exactly one interval.
    assert gov.wait_for(10.0 + 1e-9) == pytest.approx(0.2, abs=1e-6)

    # wait_for CALLS acquire, so it is not a read-only query — it consumes the slot it
    # reports. That is the property most likely to surprise a caller who polls it, and
    # nothing pinned it: a second call must report the slot after the one just taken.
    assert gov.wait_for(10.0 + 1e-9) == pytest.approx(0.4, abs=1e-6)


def test_ceiling_hz_clamps_max_hz() -> None:
    """When max_hz > ceiling_hz, the ceiling wins."""
    gov = PacingGovernor(max_hz=100.0, ceiling_hz=45.0)
    interval = 1.0 / 45.0
    gov.acquire(now=0.0)
    second = gov.acquire(now=0.0)
    assert abs(second - interval) < 1e-12


def test_max_hz_below_ceiling_is_used() -> None:
    gov = PacingGovernor(max_hz=5.0, ceiling_hz=45.0)
    interval = 1.0 / 5.0
    gov.acquire(now=0.0)
    second = gov.acquire(now=0.0)
    assert abs(second - interval) < 1e-12


def test_n_requests_spacing_invariant(
    n: int = 10, max_hz: float = 5.0
) -> None:
    """All N schedules are >= (k * interval) from time 0."""
    gov = PacingGovernor(max_hz=max_hz)
    interval = 1.0 / max_hz
    schedules = [gov.acquire(now=0.0) for _ in range(n)]
    for k, s in enumerate(schedules):
        assert s >= k * interval - 1e-12


def test_interleaved_with_real_time_advancement() -> None:
    """Issue half the requests, advance clock past them, then issue rest."""
    gov = PacingGovernor(max_hz=5.0)
    clock = FakeClock(start=0.0)

    # Issue 3 at t=0
    for _ in range(3):
        gov.acquire(now=clock.now())

    # Advance past them: all 3 should have been scheduled in [0, 0.4]
    clock.advance(2.0)
    # Now the governor's next_allowed is <= 0.4, so requesting at t=2.0 proceeds immediately
    sched = gov.acquire(now=clock.now())
    assert sched == 2.0, f"expected 2.0, got {sched}"


def test_invalid_max_hz_raises() -> None:
    with pytest.raises(ValueError, match="max_hz"):
        PacingGovernor(max_hz=0.0)


def test_invalid_ceiling_hz_raises() -> None:
    with pytest.raises(ValueError, match="ceiling_hz"):
        PacingGovernor(ceiling_hz=0.0)
