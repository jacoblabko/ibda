# Setup

This walks through getting `ibda` running end to end against a real (paper or live) IBKR
account: TWS configuration, the two non-PyPI pieces you have to build/fetch yourself (a specific
`ibapi` wheel and a patched `deephaven-ib`), and a first-run verification. See `DEPENDENCIES.md`
for exactly *why* the patched fork is required and what each patch fixes.

## 0. Prerequisites

- Python 3.12+.
- [`uv`](https://docs.astral.sh/uv/) for dependency management (`pip install uv`, or see uv's
  own install instructions).
- Trader Workstation (TWS) or IB Gateway, installed and able to log into a paper or live IBKR
  account. Either works identically for `ibda` — Gateway is lighter weight if you don't need the
  TWS UI itself.

## 1. Build the `ibapi` wheel (10.19.04, not PyPI, and not IB's newest build)

PyPI's `ibapi` package has exactly one release, `9.81.1.post1`, uploaded 2020-12-06 — IB has
never published a 10.x build there. Every consumer of a modern TWS API (`ibda` included) has to
build the wheel from IB's own TWS API distribution.

**Use `tools/build_ibapi_wheel.py`** (vendored from the same build this project was extracted
from — it has no dependency on anything outside this file):

```bash
uv run python tools/build_ibapi_wheel.py            # builds 10.19.04 into vendor/
uv run python tools/build_ibapi_wheel.py --check     # confirms the vendored wheel matches what's installed
```

It downloads `https://interactivebrokers.github.io/downloads/twsapi_macunix.1019.04.zip`,
extracts `IBJts/source/pythonclient`, and runs `python -m build --wheel` against it (so you need
the `build` package: `uv pip install build`, or `pip install build`, before running it). The
wheel lands in `vendor/ibapi-10.19.4-py3-none-any.whl`, which is exactly where this project's
`pyproject.toml` `[tool.uv.sources]` entry expects it.

**Why 10.19.04 specifically, and not IB's current "Stable" (10.45.01) or "Latest" (10.48.01):**
from TWS API **10.30 onward**, `ibapi` ships 203 generated protobuf modules, imports them
eagerly from `ibapi.client`, and hard-pins `protobuf==5.29.5`. That version conflict only bites
you if something else in your environment needs `pydeephaven` (Deephaven's separate *remote*
Barrage wire client — for connecting to a Deephaven server running in another process) at a
version whose generated code needs protobuf 6.x: protobuf refuses a gencode/runtime major-version
mismatch, so `import pydeephaven` would raise
`VersionError: Detected mismatched Protobuf Gencode/Runtime major versions ... gencode 6.31.1
runtime 5.29.5`.

`ibda` **on its own never installs `pydeephaven`** — it runs `deephaven-server` in-process
(`connect_live()` boots an embedded JVM in the same Python process; see `ibda/adapters/deephaven/server.py`),
which needs no wire client at all. So if all you run is `ibda` by itself, this specific conflict
does not apply to you, and the fork's `ibapi` version-compatibility shims (`_internal/ib_compat.py`
in the fork) do genuinely support 10.45 as well. This project still pins `ibapi==10.19.4` rather
than the newer build, for a simpler reason: it's the exact version the vendored `deephaven-ib`
fork (upstream tag v0.6.3) is built and tested against, and it drags in no protobuf dependency at
all — one less thing that can silently break if you later add a Barrage client (e.g. to query the
same Deephaven server from a separate process) to the same environment. If you do bump `ibapi`
independently, re-run the fork's own `deephaven-ib/tests/test_ib_compat.py` first.

## 2. Get the patched `deephaven-ib`

`pip install deephaven-ib` / `uv add deephaven-ib` gets you the **plain upstream package**,
which is missing every patch `ibda` requires (read-only-login position fallback, option-chain
discovery, error-tier classification, and two crash/log-storm fixes at scale — see
`DEPENDENCIES.md` for the full list and why each one is load-bearing).

This project does not (yet) publish the patched fork as an installable package on its own; you
need a copy of it as a sibling directory (e.g. `deephaven-ib/` next to this package) built from
upstream `deephaven-examples/deephaven-ib` tag `v0.6.3` plus the patches in `DEPENDENCIES.md`.
Once you have it:

```toml
# pyproject.toml
[tool.uv.sources]
deephaven-ib = { path = "./deephaven-ib" }
```

(uncomment the equivalent line already present but commented out in this project's
`pyproject.toml`).

## 3. `uv sync`

```bash
uv sync
```

This installs `deephaven-server==41.6`, the patched `deephaven-ib`, the vendored `ibapi` wheel,
and `numpy`/`pandas`/`pyarrow`. No network access beyond PyPI is needed at this point — both
non-PyPI pieces (the `ibapi` wheel and the fork) are already local files/directories by now.

## 4. Configure TWS / IB Gateway

In TWS: **File -> Global Configuration -> API -> Settings**:

- **Enable ActiveX and Socket Clients** — checked.
- **Socket port** — `7497` for paper trading, `7496` for live trading (Gateway's defaults are
  `4002`/`4001` respectively; `ibda`'s `connect_live(port=...)` defaults to `7497`, so pass the
  right port explicitly for Gateway or live TWS).
- **Read-Only API** — checked, unless you specifically need order placement/cancellation.
  `ibda` itself never places an order regardless of this setting (there is no order-submission
  code path in this package) — this setting only controls whether TWS *would allow* it if some
  other client asked. Leaving it checked is the safer default for a pure analytics client.
- **Trusted IP Addresses** — add `127.0.0.1` if TWS and your Python process run on the same
  machine (the common case).
- Under **Master API client ID**, leave the default unless you have a specific reason to
  restrict it.

`connect_live`'s `client_id` parameter must be distinct from any other client already connected
to the same TWS/Gateway instance — a collision raises IB error 326 ("client id already in use").
Pick an uncommon, memorable integer (e.g. something outside TWS's own UI client ID range) and
keep it consistent across your own reconnects, but don't reuse a number another running client
(including TWS's own charts/mobile client, which also holds a client ID) might already hold.

## 5. First-run verification

**Offline, no TWS needed** — confirms the package installed correctly and the engine-free path
works:

```bash
uv run python examples/roundtrip_pnl_ibda.py ibda/tests/fixtures/flex/report_full.xml
```

This should print a one-line summary (`N closed round-trips   total realized P&L $...`) followed
by a per-symbol breakdown, parsed entirely from the bundled test fixture — no engine boot, no
network call.

**Against a live paper account** — confirms TWS connectivity and the full engine + vendor SDK
path:

```python
import ibda

supervisor, port = ibda.connect_live(
    port=7497,          # paper TWS; use 7496 for live TWS, or Gateway's 4002/4001
    client_id=7719,      # pick your own uncommon, unused client id
    read_only=True,
)
arrow = port.table("position").snapshot()
print(arrow.to_pandas())
```

Expect roughly **~30 seconds** end-to-end the first time (JVM boot + TWS handshake + waiting for
the position stream to settle) before this returns. A non-empty `position` table (even a single
row, or zero rows for a genuinely flat paper account) confirms the full path — TWS config, the
patched fork, and `ibda`'s canonical view layer — is wired correctly.

## Non-JVM test suite

Once set up, `ibda/tests` and `ibda/analytics/tests` run without a JVM or TWS:

```bash
uv run pytest
```

`ibda/tests_jvm` boots an in-process Deephaven server (no TWS) and is excluded from the default
run; run it explicitly:

```bash
uv run pytest ibda/tests_jvm
```
