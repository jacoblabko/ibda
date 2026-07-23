from __future__ import annotations

import importlib.util
import types
from pathlib import Path

_GUARD = Path(__file__).resolve().parents[2] / "tools" / "check_ibda_boundary.py"


def _load() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("check_ibda_boundary", _GUARD)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_engine_import_outside_adapter_is_flagged(tmp_path: Path) -> None:
    guard = _load()
    f = tmp_path / "ibda" / "port.py"
    f.parent.mkdir(parents=True)
    f.write_text("import deephaven\n")
    violations = guard.scan_file(f, package_root=tmp_path / "ibda")
    assert any("deephaven" in v for v in violations)


def test_engine_import_inside_adapter_is_allowed(tmp_path: Path) -> None:
    guard = _load()
    f = tmp_path / "ibda" / "adapters" / "deephaven" / "adapter.py"
    f.parent.mkdir(parents=True)
    f.write_text("import deephaven\n")
    violations = guard.scan_file(f, package_root=tmp_path / "ibda")
    assert violations == []


def test_internal_import_is_flagged(tmp_path: Path) -> None:
    guard = _load()
    f = tmp_path / "ibda" / "result.py"
    f.parent.mkdir(parents=True)
    f.write_text("from internal.code import strategy\n")
    violations = guard.scan_file(f, package_root=tmp_path / "ibda")
    assert any("internal" in v for v in violations)


def test_deephaven_ib_allowed_under_ibkr_adapter(tmp_path: Path) -> None:
    guard = _load()
    f = tmp_path / "ibda" / "adapters" / "ibkr" / "supervisor.py"
    f.parent.mkdir(parents=True)
    f.write_text("import deephaven_ib\n")
    assert guard.scan_file(f, package_root=tmp_path / "ibda") == []


def test_deephaven_engine_still_blocked_under_ibkr(tmp_path: Path) -> None:
    guard = _load()
    f = tmp_path / "ibda" / "adapters" / "ibkr" / "x.py"
    f.parent.mkdir(parents=True)
    f.write_text("import deephaven\n")
    v = guard.scan_file(f, package_root=tmp_path / "ibda")
    assert any("deephaven" in s for s in v)


def test_deephaven_ib_blocked_outside_ibkr(tmp_path: Path) -> None:
    guard = _load()
    f = tmp_path / "ibda" / "port.py"
    f.parent.mkdir(parents=True)
    f.write_text("import deephaven_ib\n")
    v = guard.scan_file(f, package_root=tmp_path / "ibda")
    assert any("deephaven_ib" in s for s in v)
