# Dependencies

`ibda` depends on a **patched fork** of [deephaven-examples/deephaven-ib](https://github.com/deephaven-examples/deephaven-ib)
(upstream tag `v0.6.3`), not the plain package of the same name. This document lists exactly
which patches `ibda`'s own code calls or reads, verified directly against `ibda`'s call sites
(not just against the fork's own changelog) so this list can't silently drift from what the
code actually uses.

## Patches `ibda` genuinely requires

### 1. `request_sec_def_opt_params` client method + `securities_def_option_params` table

Upstream `deephaven-ib` does not wrap `reqSecDefOptParams` at all. The fork adds the
`securityDefinitionOptionParameter[End]` callbacks, a `securities_def_option_params` table, and
a `request_sec_def_opt_params()` method on the session's raw client.

**Called directly** by `ibda/adapters/ibkr/optionchain.py` (`session._client.request_sec_def_opt_params(...)`,
polling the `securities_def_option_params` table). This is `ibda`'s entire option-chain
discovery path (expirations/strikes for a symbol). **Without this patch:** `ibda.option_chain`
and everything built on it (ATM strike selection, greeks subscription) has no data source and
cannot function at all.

### 2. `accounts_portfolio` table (the `updatePortfolio`/`accountDownloadEnd` legacy-account-updates stream)

Upstream only exposes `accounts_positions` (from `reqPositions`/`reqPositionsMulti`), which a
**read-only** TWS API login does not populate. The fork adds `updatePortfolio` +
`accountDownloadEnd` callback handling, an `accounts_portfolio` TableWriter, and issues
`reqAccountUpdates(True, account)` on connect.

**Read directly** by `ibda/adapters/ibkr/supervisor.py` (`wait_for_positions_settled`'s
read-only-login row-count fallback), `ibda/adapters/ibkr/live.py` (`POSITION_PORTFOLIO_SPEC`
selection and live-marks enrichment), `ibda/adapters/deephaven/views.py`
(`enrich_position_with_marks`), and `ibda/adapters/ibkr/specs.py` (the fallback position spec).
**Without this patch:** a read-only-login connection (the default and recommended login mode --
see SETUP.md) gets an empty `position` table forever, since `accounts_positions` never
populates and there is no fallback source.

### 3. `errors` table `Tier` column (`_internal/error_tiers.py`)

Upstream logs every `EWrapper.error()` callback at `ERROR` severity unconditionally, with no
classification. The fork adds a tier classifier (INFORMATIONAL / CONNECTIVITY_DEGRADED /
GENUINE_ERROR) and a `Tier` column on the `errors` table.

**Read directly**: `ibda/adapters/ibkr/specs.py` renames the raw `Tier` column straight through
to ibda's canonical `errors.Severity` column (`ibda/schema/errors.py`) -- no reclassification
happens in `ibda` itself; it is a pure passthrough of the fork's own classification. **Without
this patch:** the canonical `errors` table has no `Severity` column at all (the rename source
column doesn't exist), so `errors` either fails schema validation or ships without the one
column that lets a caller distinguish a benign data-farm status message from a real failure.

### 4. `TableWriter.write_row`: `int` -> `float64` and `int` -> `bool` coercion

Upstream's `DynamicTableWriter.write_row` requires an exact Python type match per declared
column type. Two of `ibda`'s own usage patterns hit this directly:

- `reqPnLSingle`'s `Position` field arrives as a Python `int` at market open; the `Position`
  column is `float64`. Without the coercion, this raised a full stack-trace `ERROR` on *every*
  callback -- observed live as ~280,000 log lines / 1.5 GB in 20 minutes with 2,000+ simultaneous
  subscriptions, stalling the ibapi decode thread.
- Several boolean-typed columns (reached via `decode(bool, ...)`, bitmask expressions, and
  10.30+ protobuf field copies) do not always arrive as a Python `bool`. Without the coercion,
  `jpy` boxes the `int` as `java.lang.Byte`, and `DynamicTableWriter` raises
  `ClassCastException: class java.lang.Byte cannot be cast to class java.lang.Boolean` --
  **fatal**, not just noisy: it kills the row, then the connection.

`ibda/adapters/ibkr/pnl.py`'s `request_single_pnl_for_conids` is exactly the call path that
triggers the first case at the scale `ibda` is designed for (bounded batches of a large book).
**Without this patch:** subscribing PnL for more than a handful of conids risks stalling the
decode thread (best case) or killing the TWS connection outright (worst case) the first time a
`bool`-typed row arrives malformed.

### 5. `managedAccounts` no longer re-issues `request_account_positions(account)`

Upstream's `managedAccounts` callback calls `request_account_positions(account)` in addition to
the `connect()`-time `request_account_positions("All")`, which double-writes every row into the
append-only `accounts_positions` TableWriter for any account with a `managedAccounts` callback
(i.e. essentially always).

`ibda/adapters/ibkr/supervisor.py`'s `wait_for_positions_settled` determines "positions have
finished streaming in" purely by row-count stabilization against `accounts_positions`.
**Without this patch:** every position is double-counted, so `wait_for_positions_settled`
settles on (and callers report) exactly 2x the real position count, and any other row-count- or
uniqueness-sensitive read of `accounts_positions` is corrupted the same way.

## A dependency the fork's own changelog lists but `ibda` does NOT use

The fork also adds `IbSessionTws.wait_for_positions(timeout)` and
`wait_for_account_download(timeout)` -- reliable, event-driven completion signals meant to
*replace* count-polling. A prior pass at this document assumed `ibda` calls these. **Verified
false**: `ibda` never calls either method (confirmed by direct search over every module in this
package). `ibda`'s own `IbkrSupervisor.wait_for_positions_settled` implements its own
count-polling loop directly against the `accounts_positions`/`accounts_portfolio` row counts
(see patch #5 above) -- the exact workaround the fork's event-driven methods were meant to
replace, just not adopted here. This is not currently a correctness problem (the polling loop
works), but it does mean `ibda` does not benefit from the fork's more reliable signal, and a
future contributor should not assume `wait_for_positions`/`wait_for_account_download` are
load-bearing for `ibda` -- they aren't, today.

## Structural prerequisite, not a specific call site: the ibapi 9.81/10.19/10.30+ compatibility shims

The fork is version-adaptive across ibapi 9.81 / 10.19 / 10.45 via `_internal/ib_compat.py`
(the `error()` signature change, the `commissionReport`/`commissionAndFeesReport` rename, the
`TickTypeEnum.to_str` -> `toStr` rename, and `reqGlobalCancel`'s new required parameter, among
others -- TWS API 10.30 is a breaking boundary for all of these). `ibda` does not call any of
these shims directly; they are why the *rest* of the fork (and therefore every patch above)
still functions against whichever `ibapi` version is actually installed. This project pins
`ibapi==10.19.4` (see SETUP.md) rather than exercising this adaptivity, so if you bump `ibapi`
independently of the fork, re-run the fork's own `tests/test_ib_compat.py` first.

## How this list was produced

Every "read/called directly" claim above was verified with a direct source search
(`grep`) over this package's source for the table name, column name, or method name in
question -- not taken from the fork's own change-log prose. The one correction above
(`wait_for_positions`/`wait_for_account_download`) was found exactly this way: the claim
appeared plausible from the fork's documentation but did not survive a search for an actual
call site.
