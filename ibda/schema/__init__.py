"""ibda.schema - canonical, vendor-neutral table definitions.

Pure module tree: no engine, no vendor imports. These Schema instances are the
contract every adapter targets and every consumer codes against.
"""

from __future__ import annotations

from ibda.schema._core import Column, DType, Schema
from ibda.schema.account import ACCOUNT
from ibda.schema.account_pnl import ACCOUNT_PNL
from ibda.schema.bar import BAR
from ibda.schema.book import BOOK
from ibda.schema.cash import CASH
from ibda.schema.cash_balance import CASH_BALANCE
from ibda.schema.commission import COMMISSION
from ibda.schema.definition import DEFINITION
from ibda.schema.errors import ERRORS
from ibda.schema.execution import EXECUTION
from ibda.schema.nav import NAV
from ibda.schema.news import NEWS
from ibda.schema.news_provider import NEWS_PROVIDER
from ibda.schema.order import ORDER
from ibda.schema.position import POSITION
from ibda.schema.position_pnl import POSITION_PNL
from ibda.schema.price_tick import PRICE_TICK
from ibda.schema.quote import QUOTE
from ibda.schema.trade import TRADE

ALL: dict[str, Schema] = {
    s.name: s
    for s in (
        ACCOUNT, ACCOUNT_PNL, POSITION, POSITION_PNL, ORDER, EXECUTION,
        CASH, CASH_BALANCE, NAV, DEFINITION, QUOTE, PRICE_TICK, BAR, TRADE, BOOK,
        COMMISSION, NEWS, NEWS_PROVIDER, ERRORS,
    )
}

__all__ = [
    "Column", "DType", "Schema", "ALL",
    "ACCOUNT", "ACCOUNT_PNL", "POSITION", "POSITION_PNL",
    "ORDER", "EXECUTION", "CASH", "CASH_BALANCE", "NAV",
    "DEFINITION", "QUOTE", "PRICE_TICK", "BAR", "TRADE", "BOOK",
    "COMMISSION", "NEWS", "NEWS_PROVIDER", "ERRORS",
]
