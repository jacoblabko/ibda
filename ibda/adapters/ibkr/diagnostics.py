"""Typed IBKR error/diagnostic classifier.

The IBKR TWS API delivers status notifications, warnings, and genuine errors all
through the same ``EWrapper.error(reqId, errorCode, errorString)`` callback. This
module provides a structured classifier so callers can handle each tier correctly
without inline ad-hoc membership tests.

The full classification rule (the authoritative tier specification this module
implements) is: classify by ``errorCode``, never by the ibapi client's log severity
(the stock ``ibapi`` library logs every one of these callbacks, including benign
status notices, at ``ERROR`` level). Codes 2100-2169 are system status
notifications, not request failures, with two exceptions inside that band: text
containing "broken" or "lost" means degraded connectivity (WARNING), and 1101/1102
mean connectivity restored (INFORMATIONAL). Three codes sit outside the 2100-2169
band but are still informational, not genuine errors: 2176 (a fractional-share
order-size advisory — the order is still accepted), and 10167/10091 (delayed
market data being substituted for a non-entitled real-time request — data is still
delivered, just degraded). See the ``ErrorTier`` enum below for the full per-tier
detail this module implements.

This module is a deliberate second implementation of that rule. The canonical one
lives in the vendored ``deephaven-ib`` fork (``_internal/error_tiers.py``), which
``ibda`` may not import — the package boundary is one-way. A parity test holds the
two to the same answers and requires any divergence to be declared.

The one declared divergence is :attr:`ErrorTier.MARKET_DATA_WARNING`, a fourth tier
splitting *delivered but degraded* out of the canonical three. It is a severity
refinement only: every code in it is INFORMATIONAL to the canonical classifier, so
the two never disagree about whether a request succeeded.

Engine-free: no deephaven, no ibapi import. Pure stdlib only.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Tier enum
# ---------------------------------------------------------------------------


class ErrorTier(enum.Enum):
    """Classification tier for an IBKR ``error()`` callback.

    The IB API routes notifications, warnings, and genuine errors through the
    same callback; this enum lets callers branch on meaning rather than raw
    code.
    """

    INFORMATIONAL = "INFORMATIONAL"
    """System status notifications that require no action.

    Codes in the 2100–2169 band that are NOT degradation signals: "…connection
    is OK", subscription notes, order-entry advisory messages (e.g. bond
    order-size), and session-level restore confirmations (1101, 1102).  Log at
    INFO or drop entirely. Never abort or retry on these.
    """

    CONNECTIVITY_DEGRADED = "CONNECTIVITY_DEGRADED"
    """A farm or TWS↔IB-server link has dropped — often transient.

    Specific codes: 2103, 2105, 2157 ("…connection is broken"), 2110
    (TWS↔server broken), 1100 (connectivity lost). Also any in-band 2100–2169
    message whose text contains "broken" or "lost".  Log at WARNING; surface to
    the user but do not crash — data pauses until 1101/1102 restores it.
    """

    MARKET_DATA_WARNING = "MARKET_DATA_WARNING"
    """Market data was DELIVERED, but degraded — substituted or partial.

    The tier exists because "you got delayed data instead of real-time" deserves
    more attention than a farm-status note (the canonical classifier calls both
    INFORMATIONAL) without being a failure. The dividing line is **delivery**:

    * 10167 — "not subscribed. Displaying delayed market data" — delayed ticks
      follow; the subscription silently stopped being real-time.
    * 10091 — "Part of requested market data requires additional subscription…
      Delayed market data is available" — partial, with a delayed fallback.

    Both carry an explicit delayed-data clause — but **that clause is not the
    test**, and must never be used as one. Measured live against TWS paper, a
    ``10089`` arrived reading, verbatim:

        "Requested market data requires additional subscription for API. See link
        in 'Market Data Connections' dialog for more details.**Delayed market data
        is available.**SPY ARCA/TOP/ALL"

    …and delivered **nothing** — 0 ticks over 45s under ``market_data_type=1``.
    Re-requesting the same contract under type 3 produced ``10167`` and ticks did
    arrive. So on a denial the clause advertises *that a different market-data
    type would work*, not that this request degraded gracefully.

    The test is the **code**, which is how this module and the canonical
    classifier both implement it. Delivery is the underlying truth the codes
    encode; message text is not a proxy for it, and neither is the ``100xx``
    prefix the two groups share. A caller receiving 10167/10091 has data in hand
    and should keep going with its provenance downgraded. Log at WARNING; don't
    abort.

    Codes that deny the request outright — 10089, 10168, 10197 — are NOT here.
    Nothing arrives for them, so they are :attr:`GENUINE_ERROR`; see
    :func:`is_market_data_denied` for the per-contract distinction that matters
    to a caller. Classifying a permanent denial as a non-fatal warning is what
    produced the observed retry loops.
    """

    GENUINE_ERROR = "GENUINE_ERROR"
    """A request or connection actually failed — requires action.

    Low/sub-1100 codes such as 502 (can't connect to TWS), 504 (not
    connected), 326 (duplicate client id), 200 (no security definition), 162
    (historical data error).  Log at ERROR; fail or retry as appropriate.  This
    is also the default for any unrecognised code.

    **Scope: the request, not necessarily the session.** The market-data denials
    (10089/10168/10197) live here because nothing was delivered, but they are
    per-contract facts — the connection is healthy and other subscriptions are
    unaffected. Tear a session down on :func:`is_connection_lost` /
    :func:`is_connectivity_event`, never on "some error was GENUINE". And note
    the denials are typically **permanent** for the account until a subscription
    changes: retrying one on a timer is the observed failure mode, not the fix.
    """


# ---------------------------------------------------------------------------
# Classification tables
# ---------------------------------------------------------------------------

# Codes in the 2100–2169 band that specifically indicate degradation.
_CONNECTIVITY_DEGRADED_CODES: frozenset[int] = frozenset({
    2103,   # Market data farm connection is broken:<farm>
    2105,   # Historical data farm connection is broken:<farm>
    2110,   # Connectivity between Trader Workstation and server is broken
    2157,   # Sec-def data farm connection is broken:<farm>
    1100,   # Connectivity between IB and TWS has been lost
})

# Explicit restore notifications — treated as informational (connection is back).
_CONNECTIVITY_RESTORED_CODES: frozenset[int] = frozenset({1101, 1102})

# Individually-classified informational codes OUTSIDE the 2100-2169 band that would
# otherwise fall through to the GENUINE_ERROR default. Classified by code, never by
# ibapi's raw ERROR: log severity (see the module docstring above).
#   2176 — fractional-share order requires API version >= 163 (order-entry advisory,
#          the order is still accepted; same character as 2113's bond order-size note,
#          just numbered outside the 2100-2169 band).
_INFORMATIONAL_CODES: frozenset[int] = frozenset({2176})

# In-band (2100–2169) range boundaries.
_SYSTEM_BAND_LO: int = 2100
_SYSTEM_BAND_HI: int = 2169  # inclusive

# Market data DELIVERED but degraded. Membership is decided by one question: did the
# caller get ticks? Both of these say so in the message itself, in IB's own words:
#   10167 — "Requested market data is not subscribed. Displaying delayed market data."
#   10091 — "Part of requested market data requires additional subscription… Delayed
#            market data is available."
#
# 10089/10168/10197 deliberately do NOT belong here: each is a refusal with nothing
# delivered, confirmed against a live account rather than inferred.
_MARKET_DATA_WARNING_CODES: frozenset[int] = frozenset({
    10091,
    10167,
})

# Market data DENIED — the request returned nothing. These fall through to GENUINE_ERROR
# (the request failed), and this set exists so a caller can tell a per-contract denial
# from a connection failure without re-deriving the codes:
#   10089 — "Requested market data requires additional subscription for API." Observed
#           on liquid US equities every cycle under market_data_type 1 AND 3; no
#           bid/ask ever arrived.
#   10168 — "not subscribed. Delayed market data is not enabled" — the 10167 sibling with
#           no fallback (observed on an index: "not entitled").
#   10197 — "No market data during competing live session." IBKR admits one live
#           market-data session per login; observed 158 times in a ~31s retry loop.
_MARKET_DATA_DENIED_CODES: frozenset[int] = frozenset({
    10089,
    10168,
    10197,
})

# Regex to detect degradation language in in-band messages.
_DEGRADED_RE: re.Pattern[str] = re.compile(r"\b(broken|lost)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Core classifier
# ---------------------------------------------------------------------------


def classify_error(code: int, message: str | None = None) -> ErrorTier:
    """Classify an IBKR ``error()`` callback into its semantic tier.

    Parameters
    ----------
    code:
        The ``errorCode`` delivered by ``EWrapper.error``.
    message:
        The ``errorString`` from the same callback. Optional; used to refine
        classification for in-band (2100–2169) messages that signal degradation
        via their text ("broken", "lost") rather than their code alone.

    Returns
    -------
    :class:`ErrorTier`
        The semantic tier for this code/message pair.

    Examples
    --------
    >>> classify_error(2104)
    <ErrorTier.INFORMATIONAL: 'informational'>
    >>> classify_error(2103)
    <ErrorTier.CONNECTIVITY_DEGRADED: 'connectivity_degraded'>
    >>> classify_error(1100)
    <ErrorTier.CONNECTIVITY_DEGRADED: 'connectivity_degraded'>
    >>> classify_error(502)
    <ErrorTier.GENUINE_ERROR: 'genuine_error'>
    >>> classify_error(10167)
    <ErrorTier.MARKET_DATA_WARNING: 'market_data_warning'>
    >>> classify_error(10089)
    <ErrorTier.GENUINE_ERROR: 'genuine_error'>
    >>> classify_error(2176)
    <ErrorTier.INFORMATIONAL: 'informational'>
    """
    # Explicit connectivity-degraded codes (some are in-band, 1100 is not).
    if code in _CONNECTIVITY_DEGRADED_CODES:
        return ErrorTier.CONNECTIVITY_DEGRADED

    # Restore notifications are informational (connection is back).
    if code in _CONNECTIVITY_RESTORED_CODES:
        return ErrorTier.INFORMATIONAL

    # Individually-classified informational codes outside the 2100-2169 band (2176).
    if code in _INFORMATIONAL_CODES:
        return ErrorTier.INFORMATIONAL

    # 2100–2169: system-notification band.
    if _SYSTEM_BAND_LO <= code <= _SYSTEM_BAND_HI:
        # Message-content override: in-band codes whose text signals degradation.
        if message is not None and _DEGRADED_RE.search(message):
            return ErrorTier.CONNECTIVITY_DEGRADED
        return ErrorTier.INFORMATIONAL

    # 10000-band notices that still DELIVERED data, just degraded.
    if code in _MARKET_DATA_WARNING_CODES:
        return ErrorTier.MARKET_DATA_WARNING

    # Default: genuine error — low codes, the market-data denials
    # (_MARKET_DATA_DENIED_CODES, which deliver nothing), and anything unrecognised.
    return ErrorTier.GENUINE_ERROR


# ---------------------------------------------------------------------------
# Convenience predicates
# ---------------------------------------------------------------------------


def is_fatal(code: int, message: str | None = None) -> bool:
    """Return True iff this code is a :attr:`~ErrorTier.GENUINE_ERROR`.

    Use this to gate **request-level** abort/retry: the call this code arrived on
    did not succeed. It is not a session verdict — a market-data denial
    (:func:`is_market_data_denied`) fails one contract while the connection stays
    healthy. For connection teardown, branch on :func:`is_connection_lost` or
    :func:`is_connectivity_event`.

    Parameters
    ----------
    code:
        The ``errorCode`` from the callback.
    message:
        The ``errorString``; forwarded to :func:`classify_error`.
    """
    return classify_error(code, message) is ErrorTier.GENUINE_ERROR


def is_market_data_denied(code: int) -> bool:
    """Return True iff market data was REFUSED for this request — nothing delivered.

    ``10089`` (subscription required), ``10168`` (not subscribed, no delayed
    fallback) and ``10197`` (competing live session). All three are
    :attr:`~ErrorTier.GENUINE_ERROR`; this predicate is how a caller tells "this
    contract has no data on this account" from "the request was malformed" or "the
    connection died", without re-deriving the code list.

    Practical consequence, from live evidence: these are **persistent** — the same
    contract returns the same denial on every cycle until an account subscription
    changes (or, for 10197, until the competing session ends). A caller should stop
    requesting that contract and surface the reason, not schedule a retry. Contrast
    :func:`classify_error` ``is MARKET_DATA_WARNING`` (10167/10091), where ticks do
    arrive and the right response is to carry on with degraded provenance.
    """
    return code in _MARKET_DATA_DENIED_CODES


def is_connectivity_event(code: int) -> bool:
    """Return True iff this code relates to connectivity (degraded or restored).

    Covers both loss (:attr:`~ErrorTier.CONNECTIVITY_DEGRADED`) and restore
    (1101 / 1102 — :attr:`~ErrorTier.INFORMATIONAL` but connectivity-class).
    Use this to trigger reconnection state machines.
    """
    return code in _CONNECTIVITY_DEGRADED_CODES or code in _CONNECTIVITY_RESTORED_CODES


def is_connection_lost(code: int) -> bool:
    """Return True iff 1100 (connectivity between IB and TWS lost).

    This is the hard-loss signal; data will pause until 1101/1102 restores.
    """
    return code == 1100


def is_connection_restored(code: int) -> bool:
    """Return True iff 1101 or 1102 (connectivity restored after loss).

    1101 — restored; any subscriptions that were lost need re-subscribing.
    1102 — restored; existing subscriptions are preserved.
    """
    return code in _CONNECTIVITY_RESTORED_CODES


# ---------------------------------------------------------------------------
# Farm-status parser
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FarmStatus:
    """Parsed content of an IBKR data-farm status message.

    IBKR sends farm status through the ``error()`` callback with messages like:

    * ``"Market data farm connection is OK:usfarm"``
    * ``"Historical data farm connection is broken:ushmds"``
    * ``"HMDS data farm connection is inactive but should be available upon demand.ushmds"``

    Attributes
    ----------
    farm:
        The farm identifier extracted from the message suffix (e.g. ``"usfarm"``,
        ``"hfarm"``), or ``None`` when the message has no farm suffix.
    kind:
        Whether this is a ``"market_data"`` or ``"historical_data"`` farm, or
        ``"unknown"`` when the prefix doesn't match either pattern.
    is_ok:
        ``True`` when the farm reported OK/available; ``False`` when broken or
        inactive.
    """

    farm: str | None
    kind: Literal["market_data", "historical_data", "unknown"]
    is_ok: bool


# Patterns for the two farm kinds seen in the wild.
# "Market data farm" also matches "Sec-def data farm" (we call that market_data for
# simplicity since it carries the same operational implication).
_FARM_PATTERN: re.Pattern[str] = re.compile(
    r"""
    ^
    (?P<kind>Market\ data|Historical\ data|HMDS\ data|Sec-def\ data)
    \ farm\ connection\ is\ (?P<status>[^:.]+)     # "OK", "broken", "inactive but..."
    (?:[.:](?P<farm>\S+))?                          # optional ":usfarm" or ".ushmds"
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Terms that indicate the farm is NOT OK.
_NOT_OK_RE: re.Pattern[str] = re.compile(r"\b(broken|inactive|lost|not available)\b", re.IGNORECASE)


def classify_farm(message: str) -> FarmStatus | None:
    """Parse an IBKR farm-status ``errorString`` into a :class:`FarmStatus`.

    Returns ``None`` when *message* is not a farm-status line.

    Parameters
    ----------
    message:
        The raw ``errorString`` from ``EWrapper.error`` — e.g.
        ``"Market data farm connection is OK:usfarm"``.

    Returns
    -------
    :class:`FarmStatus` or ``None``
        Parsed farm status, or ``None`` for non-farm messages.

    Examples
    --------
    >>> classify_farm("Market data farm connection is OK:usfarm")
    FarmStatus(farm='usfarm', kind='market_data', is_ok=True)
    >>> classify_farm("Historical data farm connection is broken:ushmds")
    FarmStatus(farm='ushmds', kind='historical_data', is_ok=False)
    >>> classify_farm("[502] Could not connect to TWS")
    None
    """
    m = _FARM_PATTERN.match(message.strip())
    if m is None:
        return None

    raw_kind = m.group("kind").lower()
    if "historical" in raw_kind or "hmds" in raw_kind:
        kind: Literal["market_data", "historical_data", "unknown"] = "historical_data"
    elif "market" in raw_kind or "sec-def" in raw_kind:
        kind = "market_data"
    else:
        kind = "unknown"

    status_text = m.group("status")
    is_ok = not bool(_NOT_OK_RE.search(status_text))

    raw_farm = m.group("farm")
    farm: str | None = raw_farm.strip() if raw_farm else None

    return FarmStatus(farm=farm, kind=kind, is_ok=is_ok)


# ---------------------------------------------------------------------------
# Typed diagnostic event
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IbDiagnostic:
    """Typed representation of one IBKR ``EWrapper.error()`` invocation.

    Bundles the raw callback arguments with their semantic classification and
    optional farm parsing. Consumers can stream these through queues, log them,
    or pattern-match on :attr:`tier` without re-implementing the classifier.

    Attributes
    ----------
    code:
        The raw ``errorCode``.
    tier:
        Semantic tier from :func:`classify_error`.
    message:
        The raw ``errorString``.
    farm:
        Parsed :class:`FarmStatus` if *message* is a farm-status line, else
        ``None``.
    req_id:
        The ``reqId`` from the callback, or ``None`` when constructed without
        one. ``-1`` is the TWS sentinel for "not associated with a request".
    """

    code: int
    tier: ErrorTier
    message: str
    farm: FarmStatus | None
    req_id: int | None = None

    @classmethod
    def from_callback(cls, req_id: int, code: int, message: str) -> IbDiagnostic:
        """Construct an :class:`IbDiagnostic` from raw ``EWrapper.error`` arguments.

        Runs :func:`classify_error` and :func:`classify_farm` and returns a
        fully populated, immutable diagnostic event. This is the primary
        factory; call it from your ``error()`` override.

        Parameters
        ----------
        req_id:
            The ``reqId`` argument from ``EWrapper.error``.
        code:
            The ``errorCode`` argument.
        message:
            The ``errorString`` argument.
        """
        return cls(
            code=code,
            tier=classify_error(code, message),
            message=message,
            farm=classify_farm(message),
            req_id=req_id,
        )


# ---------------------------------------------------------------------------
# Connection-health surface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConnectionHealth:
    """Snapshot of live IBKR connection health derived from the ``errors`` table.

    Attributes
    ----------
    connected:
        Result of ``supervisor.is_connected()``; ``True`` when that call raises.
    tier_counts:
        Mapping from :class:`ErrorTier` name to count of rows in that tier.
    genuine_errors:
        List of ``{"code": int, "description": str}`` dicts for every row
        classified as :attr:`~ErrorTier.GENUINE_ERROR`.
    farms:
        Mapping from farm name (e.g. ``"usfarm"``) to latest ``is_ok`` value.
        When the same farm appears multiple times, the last row wins.
    connection_lost:
        ``True`` iff a code-1100 (connectivity lost) event appears in the error
        rows without a subsequent code-1101 or code-1102 (restore) event.
    summary:
        Human-readable one-line digest of the health state.
    """

    connected: bool
    tier_counts: dict[str, int] = field(default_factory=dict)
    genuine_errors: list[dict[str, Any]] = field(default_factory=list)
    farms: dict[str, bool] = field(default_factory=dict)
    connection_lost: bool = False
    summary: str = ""


def connection_health(supervisor: Any) -> ConnectionHealth:
    """Classify every row in the live ``errors`` table and return a health snapshot.

    Reads ``supervisor.snapshot_raw_rows("errors")`` and classifies each row
    using :func:`classify_error` and :func:`classify_farm`.  Both the connected
    query and the table read are wrapped so that an unavailable supervisor or a
    missing ``errors`` table degrade gracefully rather than raising.

    Parameters
    ----------
    supervisor:
        A live ``IbkrSupervisor`` (or any object exposing
        ``snapshot_raw_rows(name: str) -> list[dict]`` and
        ``is_connected() -> bool``).

    Returns
    -------
    :class:`ConnectionHealth`
        Immutable snapshot — safe to cache or serialise.

    Examples
    --------
    >>> ch = connection_health(supervisor)
    >>> ch.connected
    True
    >>> ch.tier_counts
    {'INFORMATIONAL': 5}
    """
    # --- connected ----------------------------------------------------------
    connected: bool = True
    try:
        connected = bool(supervisor.is_connected())
    except Exception:  # noqa: BLE001 — unknown supervisor implementation; default safe
        pass

    # --- rows ---------------------------------------------------------------
    rows: list[dict[str, Any]] = []
    try:
        rows = list(supervisor.snapshot_raw_rows("errors"))
    except Exception:  # noqa: BLE001 — table may not exist yet; graceful degradation
        pass

    # --- iterate ------------------------------------------------------------
    tier_counts: dict[str, int] = {}
    genuine_errors: list[dict[str, Any]] = []
    farms: dict[str, bool] = {}
    connection_lost: bool = False

    for row in rows:
        raw_code = row.get("ErrorCode")
        try:
            code = int(raw_code)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue

        msg = str(row.get("Error") or row.get("ErrorDescription") or "")
        tier = classify_error(code, msg)
        tier_counts[tier.name] = tier_counts.get(tier.name, 0) + 1

        if tier is ErrorTier.GENUINE_ERROR:
            desc = str(row.get("ErrorDescription") or msg)
            genuine_errors.append({"code": code, "description": desc})

        farm = classify_farm(msg)
        if farm is not None and farm.farm is not None:
            farms[farm.farm] = farm.is_ok

        if is_connection_lost(code):
            connection_lost = True
        elif is_connection_restored(code):
            connection_lost = False

    # --- summary ------------------------------------------------------------
    parts: list[str] = []
    parts.append("connected" if connected else "DISCONNECTED")
    if genuine_errors:
        codes = ", ".join(str(g["code"]) for g in genuine_errors)
        parts.append(f"{len(genuine_errors)} genuine error(s) [{codes}]")
    else:
        parts.append("no genuine errors")
    if farms:
        farm_strs = [f"{name}={'OK' if ok else 'BROKEN'}" for name, ok in sorted(farms.items())]
        parts.append("farms: " + ", ".join(farm_strs))
    if connection_lost:
        parts.append("connectivity LOST")

    summary = "; ".join(parts)

    return ConnectionHealth(
        connected=connected,
        tier_counts=tier_counts,
        genuine_errors=genuine_errors,
        farms=farms,
        connection_lost=connection_lost,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "ConnectionHealth",
    "ErrorTier",
    "FarmStatus",
    "IbDiagnostic",
    "classify_error",
    "classify_farm",
    "connection_health",
    "is_connection_lost",
    "is_connection_restored",
    "is_connectivity_event",
    "is_fatal",
    "is_market_data_denied",
]
