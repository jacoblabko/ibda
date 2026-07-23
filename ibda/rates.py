"""ibda.rates — risk-free rate proxy, a documented constant (no network).

A Sharpe ratio's risk-free term should reflect a real short-term rate, not 0%.
The **industry-standard proxy is the 3-month US Treasury yield** (the shortest
widely-quoted constant-maturity tenor, matching the daily return horizon).

:data:`DEFAULT_RISK_FREE_ANNUAL` is a **documented, hand-refreshed parameter** —
a recent 3-month Treasury yield used as the default everywhere the performance
API takes ``risk_free_annual``. It is NOT a live feed: ``ibda``'s connectivity
is IB-only, and there is no IB market-data request for a Treasury par yield, so
this value is updated by hand (bump the constant + docstring) rather than
fetched. The canonical parameter name across the performance API is
``risk_free_annual`` — it accepts either the ``"auto"`` sentinel (resolving to
this documented constant) or an explicit decimal (e.g. ``0.05``).

Importing this module performs **no I/O** and this module makes **no network
calls** at all — pure stdlib, no third-party or engine imports.
"""
from __future__ import annotations

from typing import Final

# 3-Month US Treasury par yield (investment basis), as of DEFAULT_RISK_FREE_AS_OF, from
# the Treasury.gov Daily Treasury Par Yield Curve Rates. A documented offline default,
# refreshed by hand (bump this constant + DEFAULT_RISK_FREE_AS_OF below) — not a live feed.
DEFAULT_RISK_FREE_ANNUAL: Final[float] = 0.0378

# The as-of date for DEFAULT_RISK_FREE_ANNUAL, as an ISO-8601 string — interpolated into
# resolve_risk_free()'s provenance string so a consumer logging provenance can gauge
# staleness at a glance. Bump alongside DEFAULT_RISK_FREE_ANNUAL when refreshed by hand.
DEFAULT_RISK_FREE_AS_OF: Final[str] = "2026-06-12"

# Single source of truth for the ``periods_per_year`` default across the performance
# API: trading days in a year. Every ``periods_per_year: int = ...`` default reads
# this constant rather than repeating the literal ``252``.
DEFAULT_PERIODS_PER_YEAR: Final[int] = 252


def resolve_risk_free(spec: str | float | None) -> tuple[float, str]:
    """Resolve a risk-free-rate *spec* into ``(rate, human_source)``.

    Accepts:
      * ``"auto"`` / ``None`` -> :data:`DEFAULT_RISK_FREE_ANNUAL` (the documented
        offline constant; no network call is made);
      * a number (or numeric string) -> used verbatim.

    Returns the rate plus a short human-readable provenance string suitable for
    logging (e.g. ``"offline default (documented constant, as of 2026-06-12)"`` or
    ``"user-specified"``) — the as-of date lets a consumer logging provenance gauge
    staleness of the offline default (:data:`DEFAULT_RISK_FREE_AS_OF`).

    Raises:
        ValueError: if *spec* is a non-numeric, non-``"auto"`` string.
    """
    if spec is None or (isinstance(spec, str) and spec.strip().lower() == "auto"):
        return (
            DEFAULT_RISK_FREE_ANNUAL,
            f"offline default (documented constant, as of {DEFAULT_RISK_FREE_AS_OF})",
        )
    return float(spec), "user-specified"
