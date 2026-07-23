"""Build a pinned ``ibapi`` wheel from Interactive Brokers' own TWS API distribution.

Why this exists
---------------
PyPI's ``ibapi`` has exactly one release -- ``9.81.1.post1``, uploaded 2020-12-06.
IB has never published 10.x there. Every modern consumer (including upstream
``deephaven-ib``, via its ``dhib_env.py ib-wheel``) builds the wheel from the
zip IB hosts at ``interactivebrokers.github.io``.

Leaving ``ibapi`` unpinned therefore silently resolves to a 2020 client. See
``SETUP.md`` and ``DEPENDENCIES.md`` for why this project needs the newer,
locally-built wheel instead, and why it is pinned to exactly ``10.19.04``.

Usage
-----
    uv run python scripts/build_ibapi_wheel.py                 # IB "Stable"
    uv run python scripts/build_ibapi_wheel.py --version 10.48.01
    uv run python scripts/build_ibapi_wheel.py --check         # verify vendored wheel

The wheel lands in ``vendor/`` and is committed, so ``uv sync`` stays hermetic and
offline. ``pyproject.toml`` points ``[tool.uv.sources] ibapi`` at it.

Choosing a version
------------------
**Do not bump past 10.19.04 while ``deephaven-server``/``pydeephaven`` are pinned to
41.6.** From TWS API 10.30 onward, ibapi ships 203 generated protobuf modules, imports
them eagerly from ``ibapi.client``, and declares ``protobuf==5.29.5``. ``pydeephaven``
41.6's generated code is protobuf 6.x gencode, and protobuf refuses a gencode/runtime
major-version mismatch. The two cannot share an environment::

    VersionError: Detected mismatched Protobuf Gencode/Runtime major versions when
    loading deephaven_core/proto/table.proto: gencode 6.31.1 runtime 5.29.5.

10.19.04 has no protobuf dependency, and is the version upstream ``deephaven-ib``
v0.6.3 is built and tested against.

IB also publishes newer "Stable" (10.45) and "Latest" (10.48) builds at
https://interactivebrokers.github.io/. The vendored deephaven-ib fork is already
version-adaptive across 9.81 / 10.19 / 10.45 (see ``_internal/ib_compat.py``), so the
*API* surface is not the blocker -- protobuf is. Before any bump, re-run
``deephaven-ib/tests/test_ib_compat.py`` **and** confirm ``import pydeephaven`` still works.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

#: The newest ibapi that can share an environment with pydeephaven 41.6 -- see the
#: module docstring. Also upstream deephaven-ib v0.6.3's tested target. Newer IB builds
#: exist (Stable 10.45.01, Latest 10.48.01) but pull in an incompatible protobuf.
DEFAULT_VERSION: str = "10.19.04"

#: Where the committed wheel lives, relative to the repo root.
VENDOR_DIR: Path = Path(__file__).resolve().parent.parent / "vendor"


def _download_url(version: str) -> str:
    """IB encodes ``10.45.01`` in its filenames as ``1045.01``.

    Args:
        version: dotted TWS API version, e.g. ``"10.45.01"``.

    Returns:
        The URL of the mac/unix TWS API zip for that version.

    Raises:
        ValueError: if ``version`` is not three dotted numeric components.
    """
    parts = version.split(".")

    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise ValueError(f"expected a version like '10.45.01', got {version!r}")

    major, minor, micro = (int(p) for p in parts)
    encoded = f"{major:02d}{minor:02d}.{micro:02d}"
    return f"https://interactivebrokers.github.io/downloads/twsapi_macunix.{encoded}.zip"


def _sha256(path: Path) -> str:
    """Hex digest of ``path``."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()


def build(version: str) -> Path:
    """Download IB's distribution, build the ibapi wheel, and place it in ``vendor/``.

    Args:
        version: dotted TWS API version to build.

    Returns:
        Path to the wheel written into ``vendor/``.

    Raises:
        RuntimeError: if the distribution has no Python client, or the build produced
            no wheel.
    """
    url = _download_url(version)

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        archive = work / "api.zip"

        print(f"downloading {url}")
        urllib.request.urlretrieve(url, archive)  # noqa: S310 -- fixed https host

        with zipfile.ZipFile(archive) as zf:
            zf.extractall(work)

        client = work / "IBJts" / "source" / "pythonclient"

        if not (client / "setup.py").exists():
            raise RuntimeError(f"no pythonclient in the {version} distribution")

        print(f"building wheel from {client}")
        subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--outdir", str(work / "dist")],
            cwd=client,
            check=True,
        )

        wheels = sorted((work / "dist").glob("ibapi-*.whl"))

        if not wheels:
            raise RuntimeError("build produced no wheel")

        VENDOR_DIR.mkdir(exist_ok=True)
        dest = VENDOR_DIR / wheels[0].name
        shutil.copy2(wheels[0], dest)

    print(f"wrote {dest}")
    print(f"sha256 {_sha256(dest)}")
    return dest


def check() -> int:
    """Report the vendored wheel(s) and the ibapi version currently importable.

    Returns:
        0 if exactly one wheel is vendored and it matches the installed ibapi, else 1.
    """
    wheels = sorted(VENDOR_DIR.glob("ibapi-*.whl")) if VENDOR_DIR.exists() else []

    if len(wheels) != 1:
        print(f"expected exactly one vendored ibapi wheel, found {len(wheels)}")
        return 1

    wheel = wheels[0]
    print(f"vendored: {wheel.name}  sha256={_sha256(wheel)}")

    try:
        import ibapi
    except ImportError:
        print("ibapi is not importable in this environment")
        return 1

    installed = ibapi.get_version_string()
    print(f"installed: ibapi {installed}")

    # Wheel name is `ibapi-10.45.1-py3-none-any.whl`; ibapi reports `10.45.1`.
    expected = wheel.name.split("-")[1]

    if installed != expected:
        print(f"MISMATCH: vendored wheel is {expected}, installed is {installed}")
        return 1

    return 0


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", default=DEFAULT_VERSION,
                        help=f"TWS API version to build (default: {DEFAULT_VERSION})")
    parser.add_argument("--check", action="store_true",
                        help="verify the vendored wheel matches the installed ibapi")
    args = parser.parse_args()

    if args.check:
        return check()

    build(args.version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
