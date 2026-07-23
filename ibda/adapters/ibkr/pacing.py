"""ibda.adapters.ibkr.pacing — fixed-interval pacing governor.

PURE: no engine, no vendor imports.  The governor is deterministic when a
clock-injection function is provided (required for tests).  Production code
passes ``time.monotonic`` as the clock.

Limits (from deephaven-ib / IB API documentation):
- deephaven-ib internal ceiling: ~45 requests/s (``ceiling_hz``).
- IB API practical limit for market-data subscriptions: ~5 requests/s
  (``max_hz`` default).  Exceeding this risks pacing violations (error 10089).

Usage::

    import time
    from ibda.adapters.ibkr.pacing import PacingGovernor

    gov = PacingGovernor(max_hz=5.0)
    wait = gov.wait_for(time.monotonic())
    if wait > 0:
        time.sleep(wait)
    # … issue request …
"""
from __future__ import annotations


class PacingGovernor:
    """Fixed-interval pacing governor for IBKR market-data subscription requests.

    This is a **fixed-interval spacer**, not a token bucket: each call to
    :meth:`acquire` schedules the next request exactly ``1 / max_hz`` seconds
    after the previous one.  There is no burst capacity — a caller that was
    idle for many seconds cannot send a burst; the first request is allowed
    immediately, but each subsequent request is held for one full interval.

    Parameters
    ----------
    max_hz:
        Maximum sustained request rate in requests per second (default 5.0).
        This is the IB API practical limit for market-data subscriptions.
    ceiling_hz:
        Hard ceiling on the effective rate (~45/s, the deephaven-ib internal
        limit).  ``max_hz`` is silently clamped to this value.
    """

    def __init__(self, max_hz: float = 5.0, ceiling_hz: float = 45.0) -> None:
        if ceiling_hz <= 0:
            raise ValueError(f"ceiling_hz must be positive, got {ceiling_hz}")
        if max_hz <= 0:
            raise ValueError(f"max_hz must be positive, got {max_hz}")
        self._interval: float = 1.0 / min(max_hz, ceiling_hz)
        self._next_allowed: float | None = None  # None = no prior request

    def acquire(self, now: float) -> float:
        """Return the earliest timestamp at which the next request may proceed.

        If the returned value is <= *now*, the caller may proceed immediately.
        If it is > *now*, the caller must wait ``acquire(now) - now`` seconds.

        Parameters
        ----------
        now:
            Current time (e.g. ``time.monotonic()``).  Injected for
            deterministic testing.

        Returns
        -------
        Absolute timestamp (same epoch as *now*) of the next allowed request.
        Calling ``acquire`` also advances the internal interval schedule: successive
        calls at the same *now* are correctly spaced apart.
        """
        if self._next_allowed is None or now >= self._next_allowed:
            # First request or a gap larger than the interval: proceed from now.
            scheduled = now
        else:
            scheduled = self._next_allowed
        # Advance the interval schedule: next request may not come sooner than one interval later.
        self._next_allowed = scheduled + self._interval
        return scheduled

    def wait_for(self, now: float) -> float:
        """Return seconds the caller should sleep before issuing the next request.

        Returns 0.0 if the caller may proceed immediately.

        Parameters
        ----------
        now:
            Current time (same epoch as used in ``acquire``).
        """
        scheduled = self.acquire(now)
        return max(0.0, scheduled - now)
