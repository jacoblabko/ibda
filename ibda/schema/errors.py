"""Canonical `errors` table — the IB API error/notification stream.

Populated from deephaven-ib's ``errors`` table, auto-populated from connect:
every ``EWrapper.error()`` callback is captured, including benign data-farm
status notifications delivered on the same channel as genuine failures. The
IBKR TWS API multiplexes status notices, warnings, and real errors onto that
one callback — classify by ``errorCode``, not log severity: 2100-2169 is the
system-status band (with "broken"/"lost" text and 1100/1101/1102 refining it
to connectivity-degraded / restored), and 2176/10167/10091 are informational
outliers outside that band (delayed-data substitution, order-entry advisories)
that still deliver data and must not be treated as failures.

``Severity`` carries the informational / connectivity-degraded / genuine-error
classification the vendored ``deephaven-ib`` fork's tier classifier computes
(``deephaven_ib._internal.error_tiers.classify_error_tier``) — the raw
``errors`` table's ``Tier`` column is renamed straight through, so no
reclassification happens in ``ibda``.  Filter on ``Severity`` to drop benign
farm-status noise rather than branching on ``Code`` yourself.

Time-ordered event log; no dedupe — every callback is retained.
"""

from __future__ import annotations

from ibda.schema._core import Column, DType, Schema

ERRORS = Schema(
    name="errors",
    doc=(
        "IB API error/notification stream, auto-populated from connect. "
        "Severity classifies each row as INFORMATIONAL, "
        "CONNECTIVITY_DEGRADED, or GENUINE_ERROR — filter on Severity to "
        "drop benign farm-status noise instead of branching on Code."
    ),
    columns=(
        Column(
            "Timestamp",
            DType.TIMESTAMP_NS,
            nullable=False,
            doc="receive time of the error() callback (UTC)",
        ),
        Column(
            "ReqId",
            DType.INT64,
            doc="request id this message relates to; null for general/system notifications",
        ),
        Column("Code", DType.INT64, nullable=False, doc="IB errorCode"),
        Column("Message", DType.STRING, nullable=False, doc="short IB-supplied error description"),
        Column("Detail", DType.STRING, doc="full error string, including contract/context suffix"),
        Column(
            "Note",
            DType.STRING,
            doc="deephaven-ib's explanatory note for this code (populated for informational codes)",
        ),
        Column(
            "Severity",
            DType.STRING,
            nullable=False,
            doc=(
                "INFORMATIONAL | CONNECTIVITY_DEGRADED | GENUINE_ERROR — classified by "
                "errorCode (module docstring above has the full rule), never by ibapi's "
                "raw ERROR: log severity"
            ),
        ),
        Column("Sym", DType.STRING, doc="related instrument symbol, if any"),
        Column("ConId", DType.INT64, doc="related IBKR contract id, if any"),
    ),
)
