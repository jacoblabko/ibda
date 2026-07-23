"""IBKR contract construction + small contract/exchange helpers.

Pure where possible: ``parse_expiry`` is stdlib-only; the builders import
``ibapi.contract.Contract`` (vendor type, fine in the adapter layer). No deephaven
(engine) import — these never start a JVM.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any

from ibapi.contract import Contract


def parse_expiry(value: str | _dt.date) -> _dt.date:
    """Parse an IBKR expiry into a ``date``.

    Accepts a ``date`` (returned unchanged), an 8-char "YYYYMMDD", or a 6-char
    "YYYYMM" (first of the month).
    """
    if isinstance(value, _dt.date):
        return value
    s = value.strip()
    if len(s) == 8:
        return _dt.datetime.strptime(s, "%Y%m%d").date()
    if len(s) == 6:
        return _dt.datetime.strptime(s, "%Y%m").date()
    raise ValueError(f"unrecognised expiry format: {value!r}")


def build_stock_contract(
    symbol: str,
    exchange: str = "SMART",
    currency: str = "USD",
    primary_exchange: str | None = None,
) -> Contract:
    """Build a US-stock ``Contract`` (SecType=STK)."""
    c: Contract = Contract()
    c.symbol = symbol.strip().upper()
    c.secType = "STK"
    c.currency = currency
    c.exchange = exchange
    if primary_exchange:
        c.primaryExchange = primary_exchange
    return c


def build_fx_contract(
    base_currency: str,
    quote_currency: str = "USD",
    exchange: str = "IDEALPRO",
) -> Contract:
    """Build an FX spot ``Contract`` (SecType=CASH) on IDEALPRO.

    IBKR encodes an FX pair as ``symbol=<base>``, ``currency=<quote>`` — e.g.
    ``base_currency="EUR", quote_currency="USD"`` is the EUR/USD pair. Used by
    callers that need an IB-sourced prior-close FX reference rate when no
    external FX feed is otherwise available.
    """
    c: Contract = Contract()
    c.symbol = base_currency.strip().upper()
    c.secType = "CASH"
    c.currency = quote_currency.strip().upper()
    c.exchange = exchange
    return c


def build_option_contract(
    symbol: str,
    expiry: str | _dt.date,
    strike: float,
    right: str,
    exchange: str = "SMART",
    currency: str = "USD",
    multiplier: str = "100",
    trading_class: str | None = None,
) -> Contract:
    """Build an option ``Contract`` (SecType=OPT).

    ``expiry`` accepts a ``date`` or "YYYYMMDD"/"YYYYMM"; ``right`` is "C" or "P".
    """
    r = right.strip().upper()
    if r not in ("C", "P"):
        raise ValueError(f"right must be 'C' or 'P', got {right!r}")
    c: Contract = Contract()
    c.symbol = symbol.strip().upper()
    c.secType = "OPT"
    c.currency = currency
    c.exchange = exchange
    c.lastTradeDateOrContractMonth = parse_expiry(expiry).strftime("%Y%m%d")
    c.strike = float(strike)
    c.right = r
    c.multiplier = multiplier
    if trading_class:
        c.tradingClass = trading_class
    return c


def build_contract_from_conid(conid: int, exchange: str = "SMART") -> Contract:
    """Build a bare ``Contract`` from a known contract id.

    Creates the minimal contract needed to re-use an already-known ConId
    without a symbol-to-ConId lookup (e.g. market-data subscriptions,
    reqPnLSingle).  ``secType`` is intentionally left blank so TWS resolves
    it from the ConId.

    Parameters
    ----------
    conid:
        IBKR contract id.
    exchange:
        Routing exchange (default ``"SMART"``).
    """
    c: Contract = Contract()
    c.conId = conid
    c.exchange = exchange
    return c


def _conid_of(rc: Any) -> int | None:
    """Extract the contract id from a deephaven-ib RegisteredContract.

    Returns ``None`` on any failure (missing attribute, type mismatch, empty
    ``contract_details``) so callers can treat the result as optional without
    catching exceptions.
    """
    cds: Any = getattr(rc, "contract_details", None)
    cd0: Any = cds[0] if isinstance(cds, (list, tuple)) and cds else cds
    try:
        return int(cd0.contract.conId)
    except (AttributeError, TypeError, ValueError, IndexError):
        return None


def _contract_of(rc: Any) -> Any:
    """Extract the underlying ibapi Contract from a deephaven-ib RegisteredContract.

    Mirrors _conid_of's access path (contract_details[0].contract).
    """
    cds: Any = getattr(rc, "contract_details", None)
    cd0: Any = cds[0] if isinstance(cds, (list, tuple)) and cds else cds
    return getattr(cd0, "contract", None)


def get_primary_exchange(supervisor: Any, symbol: str) -> str | None:
    """Resolve a symbol's primary exchange, or None if unknown / unresolvable."""
    session: Any = supervisor._session
    try:
        rc = session.get_registered_contract(build_stock_contract(symbol))
    except Exception:  # noqa: BLE001 — never raise into callers; unknown → None
        return None
    contract = _contract_of(rc)
    primary: str | None = getattr(contract, "primaryExchange", None)
    return primary or None
