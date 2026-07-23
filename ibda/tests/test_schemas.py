from __future__ import annotations

import pyarrow as pa

from ibda import schema as S


ALL_SCHEMAS = [
    S.ACCOUNT, S.POSITION, S.ORDER, S.EXECUTION, S.CASH, S.DEFINITION,
    S.QUOTE, S.BAR, S.TRADE, S.BOOK,
]


def test_every_schema_is_wellformed_and_documented() -> None:
    for sch in ALL_SCHEMAS:
        assert sch.columns, f"{sch.name} has no columns"
        assert sch.doc, f"{sch.name} has no doc"
        arrow = sch.to_arrow_schema()
        assert isinstance(arrow, pa.Schema)
        for col in sch.columns:
            assert col.doc, f"{sch.name}.{col.name} undocumented"


def test_execution_has_settlement_and_core_fields() -> None:
    names = S.EXECUTION.column_names
    for required in ("ExecId", "Timestamp", "Sym", "Side", "Qty", "Price", "Commission"):
        assert required in names
    assert S.EXECUTION.settled_flag is False
