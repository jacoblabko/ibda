"""ibda.errors — typed errors raised at the API boundary.

Pure module: no engine, no vendor imports.
"""

from __future__ import annotations


class IbdaError(Exception):
    """Base class for every error surfaced by ibda."""


class UnknownTable(IbdaError):
    """A canonical table name was requested that the adapter does not provide."""


class SchemaMismatch(IbdaError):
    """Data produced by an adapter does not conform to its declared schema."""


class FlexParseError(IbdaError):
    """A Flex XML report could not be parsed or was not yet ready.

    Raised by :func:`ibda.load_flex_xml` and :func:`ibda.load_flex_file`
    when ``parse_statement`` returns a non-``"ok"`` status.
    """
