#!/usr/bin/env python3
"""Boundary guard for the ibda package (pre-commit).

Enforces the one-way / engine-confinement / vendor-confinement rules this package was
designed around from the start:

  1. Only ibda/adapters/deephaven/ may import deephaven*/pydeephaven.
  2. No ibda module may import first-party code from outside this package
     (import roots the package must never depend on).
  3. Only ibda/adapters/ibkr/ may import ibapi/deephaven_ib.

Tests (ibda/tests, ibda/tests_jvm) are exempt. Exit non-zero on any violation. Run it as
a pre-commit hook or standalone: ``python tools/check_ibda_boundary.py``.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ENGINE_PREFIXES: tuple[str, ...] = ("deephaven", "pydeephaven")
VENDOR_PREFIXES: tuple[str, ...] = ("ibapi", "deephaven_ib")
# First-party import roots that ibda must never depend on, keeping the one-way
# boundary intact (ibda is a standalone package; nothing outside it may leak in).
FORBIDDEN_LOCAL: tuple[str, ...] = ("internal", "shared", "scripts")

ENGINE_ALLOWED: tuple[str, ...] = ("adapters", "deephaven")   # relative path parts under the package
VENDOR_ALLOWED: tuple[str, ...] = ("adapters", "ibkr")
EXEMPT_DIRS: tuple[str, ...] = ("tests", "tests_jvm")


def _imported_roots(tree: ast.AST) -> list[str]:
    roots: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.append(node.module.split(".")[0])
    return roots


def scan_file(path: Path, *, package_root: Path) -> list[str]:
    """Return a list of human-readable violation strings for one file."""
    rel = path.relative_to(package_root)
    if rel.parts and rel.parts[0] in EXEMPT_DIRS:
        return []
    in_engine_adapter = rel.parts[: len(ENGINE_ALLOWED)] == ENGINE_ALLOWED
    in_vendor_adapter = rel.parts[: len(VENDOR_ALLOWED)] == VENDOR_ALLOWED

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:  # pragma: no cover
        return [f"{rel}: syntax error: {exc}"]

    violations: list[str] = []
    for root in _imported_roots(tree):
        is_vendor = root.startswith(VENDOR_PREFIXES)
        is_engine = root.startswith(ENGINE_PREFIXES) and not is_vendor
        if is_engine and not in_engine_adapter:
            violations.append(f"{rel}: engine import {root!r} outside adapters/deephaven/")
        if is_vendor and not in_vendor_adapter:
            violations.append(f"{rel}: vendor import {root!r} outside adapters/ibkr/")
        if root in FORBIDDEN_LOCAL:
            violations.append(f"{rel}: ibda must not import {root!r} (one-way boundary)")
    return violations


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    package_root = repo_root / "ibda"
    if not package_root.exists():
        return 0
    all_violations: list[str] = []
    for path in package_root.rglob("*.py"):
        all_violations += scan_file(path, package_root=package_root)
    if all_violations:
        print("ibda boundary violations:", file=sys.stderr)
        for v in all_violations:
            print(f"  {v}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
