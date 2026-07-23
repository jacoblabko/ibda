from __future__ import annotations

import pytest

from ibda.port import DataPort


def test_dataport_is_abstract() -> None:
    with pytest.raises(TypeError):
        DataPort()  # type: ignore[abstract]
