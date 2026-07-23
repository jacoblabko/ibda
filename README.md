# ibda

`ibda` is a read-only data-access layer over an Interactive Brokers account. It turns the raw
TWS API — asynchronous callbacks, no tables — plus `deephaven-ib`'s 34 append-only event tables
into **19 canonical, typed, current-state tables** (`position`, `execution`, `nav`, `quote`, ...),
materialized as `pyarrow.Table`. The same query surface works from a live TWS connection or
offline from a Flex Activity XML statement. For a quant or developer who already has an IBKR
account and wants Arrow tables instead of callback plumbing.

Importing `ibda` never boots a compute engine — that happens lazily, on the first call that
actually needs data.

## Quickstart

### Offline, from a Flex Activity XML report (no TWS, no engine, no market-data subscription)

This is the path you can run right now, with nothing but a Flex XML file:

```python
from ibda.adapters.ibkr.flex.parse import parse_statement_or_raise
from ibda.adapters.ibkr.flex.arrow import flex_arrow_tables  # or: from ibda import flex_arrow_tables

with open("activity.xml", encoding="utf-8") as f:
    xml = f.read()

sections = parse_statement_or_raise(xml)   # raw Flex XML -> parsed sections
tables = flex_arrow_tables(sections)       # sections -> {"execution": pa.Table, "cash": pa.Table, ...}

df = tables["execution"].to_pandas()
```

No JVM, no network call — pure parse-then-map. If you just want Sharpe/Sortino/Calmar and don't
care about the intermediate tables:

```python
import ibda

perf = ibda.flex_performance("activity.xml", risk_free_annual=0.05)
print(perf.render())
```

If instead you want a live, queryable `DataPort` from that same file (`.filter`, `.group_by`,
`.as_of_join`, `.subscribe(...)`), use `ibda.load_flex_file(path)` — the trade-off is that it
boots the Deephaven engine (an in-process JVM) to back those tables. Prefer the engine-free path
above unless you need that query surface.

### Live, against a TWS connection (boots an engine + the vendor SDK)

Needs a running TWS or IB Gateway session — see `SETUP.md` before trying this.

```python
import ibda

supervisor, port = ibda.connect_live(port=7497, client_id=<uncommon>, read_only=True)
arrow = port.table("position").snapshot()   # pyarrow.Table
```

`port=7497` is paper TWS (`7496` for live TWS). `client_id` must be distinct from any other
client already connected to the same TWS/Gateway instance, or IB raises error 326. Expect
roughly 30 seconds end to end the first time (JVM boot + TWS handshake + waiting for the
position stream to settle).

Whichever way you connect, `port.table(name).snapshot()` always returns a `pyarrow.Table`, and
`.subscribe(callback)` gives you a push feed of the same rows instead of a one-shot snapshot.

## What this does and doesn't do

- **Read-only analytics. No order placement.** There is no order-submission code path in this
  package, regardless of how TWS's "Read-Only API" checkbox is set on the connecting session.
- **IB-only, despite the vendor-neutral *interface*.** The canonical schema itself is
  IB-specific (`conId`, `SecType`, and similar fields carry IBKR's own semantics) — "vendor-neutral"
  describes the query surface (`DataPort`, `Result.snapshot()`, Arrow everywhere), not a proven
  claim that this schema maps cleanly onto another broker. Only one engine port exists today:
  Deephaven.
- **A read-only TWS login withholds order state.** Connect with `read_only=True` (the default,
  and the recommended setting) and the `order` table has no open-order or arrival-time data —
  that requires a read-write login, which also means TWS would allow that session to place
  orders (this package still won't; see above).
- **No L2/L3 market depth.** The `book` table is defined in the schema but unwired — nothing in
  this package or its `deephaven-ib` dependency requests a market-depth feed.
- **Live quotes need per-exchange market-data entitlements.** `quote`/`price_tick`/`bar` only
  populate for contracts you hold a real-time data subscription for; otherwise expect delayed
  data or nothing at all, independent of anything `ibda` does.

## Install

**This is not `pip install ibda` today.** Two of the three non-PyPI-friendly dependencies have
to be built or fetched yourself before `uv sync` will succeed:

- `ibapi` — PyPI's only release is `9.81.1.post1` from 2020; a modern build has to be compiled
  from IB's own TWS API distribution. A vendoring script is included.
- `deephaven-ib` — the plain PyPI package is missing several patches this package's code calls
  directly (read-only-login position fallback, option-chain discovery, error-tier
  classification, two crash/log-storm fixes at scale). You need a patched fork, not the upstream
  package.

**Read `SETUP.md` first** — it walks through building the `ibapi` wheel, pointing `uv` at the
patched fork, and TWS configuration, end to end. **`DEPENDENCIES.md`** documents exactly which
fork patches are load-bearing and why, verified against this package's own call sites (not the
fork's changelog). Once both pieces are in place, `uv sync` installs everything else
(`deephaven-server`, `pyarrow`, `numpy`, `pandas`) from PyPI normally.

## Why this exists

A broker API surfaces a lot of incidental complexity that has nothing to do with the analytics
question you're actually asking. A few examples this layer settles once, so calling code doesn't
have to rediscover them:

- **`Multiplier` arrives as the string `"100"`**, not a number — every consumer of the raw feed
  either forgets to cast it or casts it inconsistently.
- **IBKR's "unset" marker is `Double.MAX_VALUE`** — a finite float that silently passes `isnan`/
  `isinf` guards, so a naive numeric check treats "no value" as a legitimate (enormous) value.
- **Status notifications, warnings, and genuine errors all arrive on the same `error()`
  callback.** Error code 2104 means a market-data farm connection came *up* — not a failure —
  and the raw feed logs it at `ERROR` severity regardless.
- **Fills and commissions arrive on separate callbacks**, keyed only by `ExecId`, and have to be
  joined back together before an execution record means anything.

None of this is exotic once you know it, but knowing it is exactly the kind of thing that has to
be re-learned by every new consumer of the raw API. The canonical tables here encode the answer
once.

## Testing

```
uv run pytest ibda/tests ibda/analytics/tests
```

898 passed (no JVM, no TWS, no network).

```
uv run pytest ibda/tests_jvm
```

145 passed. This lane boots an in-process Deephaven server — it needs a Java runtime, but no TWS
connection and no network access.

## Project layout

```
ibda/
  __init__.py           Public surface (see ibda.__all__)
  schema/                Canonical table definitions (ibda.schema.ALL)
  adapters/
    ibkr/                Live TWS + Flex XML adapters
    deephaven/           Engine-facing view construction
  analytics/             Performance, relative/benchmark, and round-trip P&L analytics
  tests/, tests_jvm/     Non-JVM and JVM test lanes (see Testing above)
examples/                Runnable end-to-end scripts
tools/                   ibapi wheel builder, boundary checker
SETUP.md                 Full environment setup, start to finish
DEPENDENCIES.md          Exactly which deephaven-ib fork patches are required, and why
```

See `ibda/README.md` for the fuller API reference: canonical table names, the full public
surface (`ibda.__all__`), and the error hierarchy (`ibda.errors.IbdaError` and its subclasses).

## License

Apache License 2.0 — see [`LICENSE`](LICENSE). Chosen to match Deephaven's own license so this
layer stays clean for upstream contribution.
