from __future__ import annotations

import pyarrow as pa

from ibda import schema as S


# Taken from the registry rather than retyped: the hand-written list had drifted to
# 10 of the 19 registered schemas, leaving 9 schemas (52 columns) outside the only
# `col.doc` assertion in the repo. Reading `S.ALL` means a newly registered schema is
# documented-or-red on the commit that adds it.
ALL_SCHEMAS = list(S.ALL.values())


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
