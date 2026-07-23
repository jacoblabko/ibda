# ibda

`ibda` is an engine-hidden, vendor-neutral data-access API over an IBKR account. It gives you
canonical tables (`position`, `execution`, `nav`, ...) as Apache Arrow, from either a live TWS
connection or an offline Flex Activity XML report. **Importing `ibda` never boots a compute
engine** — the Deephaven engine (and, for live connections, the IBKR vendor SDK) is imported and
started lazily, the first time you call a data-producing entry point.

That "does this boot an engine?" distinction is the one thing to get right before reaching for an
entry point — it's called out for each recipe below.

## Recipe 1 — offline, from a Flex XML file (no engine, no TWS)

The engine-free path: parse the XML to plain Python sections, then map those sections straight to
`pyarrow.Table` — no Deephaven, no JVM.

```python
from ibda.adapters.ibkr.flex.parse import parse_statement_or_raise
from ibda.adapters.ibkr.flex.arrow import flex_arrow_tables  # or: from ibda import flex_arrow_tables

with open("activity.xml", encoding="utf-8") as f:
    xml = f.read()

sections = parse_statement_or_raise(xml)   # str (raw Flex XML) -> parsed sections (a Mapping)
tables = flex_arrow_tables(sections)       # sections -> {"execution": pa.Table, "cash": pa.Table, ...}

df = tables["execution"].to_pandas()
```

`flex_arrow_tables` takes **parsed sections**, not a file path or raw XML string — `parse_statement_or_raise`
is always the step before it. This pairing is the whole engine-free recipe; nothing else is needed.

If you only want Sharpe/Sortino/Calmar and don't care about intermediate tables, skip straight to
`ibda.flex_performance("activity.xml")` — same engine-free path, one call, returns a
`PerformanceSummary` (see its docstring for what the Flex query must include).

Round-trip P&L reconstruction from the same `execution` table is demonstrated end-to-end in
`examples/roundtrip_pnl_ibda.py`.

### If you have a file path and want a live, queryable `DataPort` instead

`ibda.load_flex_file(path)` / `ibda.load_flex_xml(xml)` take a path or XML string directly (no
separate parse step) and hand back a `DataPort` you can `.table(name).snapshot()` or `.subscribe(...)`
against with the same op vocabulary as a live connection (`filter`, `group_by`, `as_of_join`, ...).
The trade-off: **this boots the Deephaven engine** (starts an in-process JVM) to back those tables,
which is real weight for a one-shot read. Prefer Recipe 1 above unless you need `DataPort`'s
query surface.

## Recipe 2 — live TWS connection (boots an engine + the vendor SDK)

```python
import ibda

supervisor, port = ibda.connect_live(port=7497, client_id=<uncommon>, read_only=True)
arrow = port.table("position").snapshot()   # pyarrow.Table
```

`connect_live` starts the Deephaven engine, connects to TWS, waits for the position stream to
settle, and builds every canonical live view — measured at **~29s** end-to-end (JVM boot + TWS
handshake + settle wait) before it returns. `port` (`.table("execution")`, `.table("nav")`, ...)
gives you the same `DataPort` surface as the Flex-loaded path.

- `port=7497` is paper TWS; use `7496` for live TWS.
- Always pass a `client_id` distinct from any other session connected to the same TWS instance —
  a collision raises IB error 326 (client id already in use).
- `read_only=True` for analytics-only sessions (the default and the safe choice unless you also
  need order placement/cancellation).

`supervisor` (an `IbkrSupervisor`) is returned alongside `port` for health checks and account
refresh; most read-only analytics code only needs `port`.

## Every `Result.snapshot()` returns `pyarrow.Table`

Whichever way you connected — `connect_live`, `load_flex_file`/`load_flex_xml`, or `connect(tables)`
directly — `port.table(name)` returns a `Result`, and `Result.snapshot()` always materializes as a
`pyarrow.Table` (`.to_pandas()`, `.to_pylist()`, etc. from there). `Result.subscribe(callback)` gives
you a push feed of the same rows instead of a one-shot snapshot.

## Canonical table names

`ibda.schema.ALL` is the authoritative registry. Today: `account`, `account_pnl`, `position`,
`position_pnl`, `order`, `execution`, `cash`, `cash_balance`, `nav`, `definition`, `quote`,
`price_tick`, `bar`, `trade`, `book`, `commission`, `news`, `news_provider`, `errors`. Not every
table is populated by every entry point — a Flex-loaded `DataPort` only has `execution`/`cash`
(plus `nav` if the report included the daily equity summary); a live `connect_live` session
populates the full set relevant to a live account.

## What's public

`ibda.__all__` is the supported surface: `connect`, `DataPort`, `load_flex_file`, `load_flex_xml`,
`flex_performance`, `flex_arrow_tables`, `connect_live`, the performance/relative/round-trip
analytics (`PerformanceSummary`, `compute_performance`, `RelativeSummary`, `RoundTrip`,
`reconstruct_round_trips`, ...), `Result`/`Stream`, `watch`, and the `schema`/`errors`/`rates`
submodules. Everything else (`ibda.adapters.*`, `ibda.analytics.*` internals not re-exported above)
is reachable but not part of the stability contract — the two flex-adapter functions used in
Recipe 1 (`parse_statement_or_raise`, `flex_arrow_tables`'s home module) are the one documented
exception, since they are the canonical engine-free pairing.

## Errors

Every error `ibda` raises deliberately derives from `ibda.errors.IbdaError` (see `ibda/errors.py`):
`FlexParseError` (malformed/not-ready/rejected Flex XML), `UnknownTable`, `SchemaMismatch`.
