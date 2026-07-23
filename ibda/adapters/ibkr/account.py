"""ibda.adapters.ibkr.account — account_overview snapshot helper.

VENDOR-CONFINED: this module lives under ``adapters/ibkr/`` and may touch the
deephaven engine **only** through the supervisor's injected ``snapshot_rows``
helper (``ibda.adapters.deephaven.views.snapshot_rows``).  It MUST NOT
import ``deephaven`` directly.

Public function:

``snapshot_account_overview(supervisor) -> pd.DataFrame``
    Snapshot the raw ``accounts_overview`` table via the supervisor's injected
    ``_snapshot_rows`` helper and return it as a pandas DataFrame with the
    original column names (Account, Key, Value, DoubleValue, Currency, ...).

Unlike the position / execution / order helpers, ``accounts_overview`` needs no
canonical-view transformation: its raw structure is already the key-value form
``parse_account_summary`` expects.  There is no renaming spec and no
``apply_canonical_view`` call.

Returns an empty DataFrame on any read failure so callers get a zero-filled
``AccountSummary`` rather than an exception.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ibda.adapters.ibkr.supervisor import IbkrSupervisor

logger = logging.getLogger(__name__)


def snapshot_account_overview(supervisor: IbkrSupervisor) -> pd.DataFrame:
    """Snapshot ``accounts_overview`` to a pandas DataFrame.

    Reads the raw ``accounts_overview`` table from the supervisor's session via
    the injected ``snapshot_rows`` helper (so the deephaven engine import lives
    inside that helper, not here).  Returns the rows as a DataFrame with the
    original deephaven-ib column names:
    ``RequestId``, ``ReceiveTime``, ``Account``, ``Currency``, ``Key``,
    ``ModelCode``, ``Value``, ``DoubleValue``.

    Returns an empty DataFrame (no columns) when the table is absent or any
    read error occurs, so downstream callers receive a zero-filled summary
    rather than an exception.

    Parameters
    ----------
    supervisor:
        A connected ``IbkrSupervisor`` instance.

    Returns
    -------
    pd.DataFrame
        Raw ``accounts_overview`` rows as a DataFrame.  May be empty.
    """
    try:
        raw: Any = supervisor.raw_table("accounts_overview")
        rows: list[dict[str, Any]] = supervisor._snapshot_rows(raw)
    except Exception as exc:  # noqa: BLE001 — return empty on any read failure
        logger.warning(
            "snapshot_account_overview: error reading accounts_overview: %s", exc
        )
        supervisor._last_error = str(exc)
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)
