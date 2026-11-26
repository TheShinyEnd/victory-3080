# Frame-time profiling

`victory_core::prof` is a scoped-timer facility present in every build (no
feature gate). A scope costs two clock reads, a thread-local touch and three
relaxed atomic adds when it closes (about 60 ns); nothing allocates or locks
on that path. `VICTORY_PROFILE_OFF=1` makes spans do nothing (to measure that
cost). It never changes what the title or the shaders observe.

## Declaring scopes

    let _p = victory_core::prof_scope!("watch.take");       // time, until dropped
    let _p = victory_core::prof_fine!("list.state_call");   // as above, only with fine scopes on
    victory_core::prof_count!("copy.jobs.bytes", n);        // a quantity
    let g = victory_core::prof::waits!(MUTEX, "x.wait");    // a MutexGuard; only a contended wait is timed
    let g = victory_core::prof::locked!(MUTEX, "x.wait", "x.hold"); // wait and hold, both
    victory_core::prof_top!("pipeline.make", ns, || label); // "slowest of the interval" list
    victory_core::prof_tally!("drop.x", key, || label);     // counts per key, top 10 per interval

Scopes of the same name share a column. `crate::timeline::span("text")` (the
spans the diagnostics timeline already has, about 45 across the submitting
thread, relay, landings, the title's calls) is also a scope named by its text,
in every build.

## What a scope hierarchy means

Scopes nest, so a frame's scopes add up to more than the frame. The per-thread
line shows time inside outermost scopes against wall time ("unattributed" is the
thread's own code or calls without a scope). Read the biggest scope of each
thread first, then its children.

## Output

* Every ~5 s (inside the frame-rate line's interval): `frame profile |` lines,
  all scopes by ms/frame (calls/frame, longest single-frame total), then
  quantities, then per thread time in scopes against wall time, then
  `slowest pipeline.make` / `slowest shader.translate` lists, then
  `drop report |` blocks (see below). `grep "frame profile |\|drop report |"`.
* `stall:` line when one frame takes longer than `VICTORY_STALL_MS` (default
  1000): `stall: frame N took X ms: <top 6 scopes with ms>, pipelines being made
  A->B, textures uploaded MB (hashed MB), video memory local used/budget MB,
  shared used/budget MB`. N counts presents.
* `long block: thread T inside scope S (now in I) for N s and counting` when a
  thread stays in one outermost scope longer than `VICTORY_LONG_BLOCK_MS`
  (default 2000), once per episode, while it is still stuck.
* `VICTORY_PROFILE=<dir>`: `frames.csv` (a row per presented frame: `world` =
  more than 300 draws, `wall_ms`, a column per scope in ms or units, then
  `name#n` call counts; thread columns are busy ms), `trace.json` (Chrome trace,
  "X" complete events = begin and end in one, per scope per thread; open in
  `chrome://tracing` or ui.perfetto.dev; events under 10 us left out) and
  `summary.txt`. `VICTORY_PROFILE_FRAMES=a-b` traces frames a..b; `n` or unset
  traces the n (default 4) slowest world frames. Files are written by the
  `victory-prof-monitor` thread a moment after each interval, never by the
  presenting thread. Up to 20,000 frames are kept. Setting `VICTORY_PROFILE`
  also turns on fine scopes.
* `VICTORY_PROFILE_SUMMARY=1`: `frame profile summary |` lines with each
  interval: p50 / p90 / max per scope over world frames so far (the last block
  of a log is the end-of-run table; `summary.txt` has the same).
* `VICTORY_PROFILE_FINE=1`: fine scopes (every draw-state call,
  `list.state_call`, `draw.pipeline_lookup`). They cost a few percent at ~50,000
  calls a frame, so they are off unless asked for (or `VICTORY_PROFILE` is set).

## Drop report

Per interval, for draws, mesh dispatches and dispatches recorded but not run:

    drop report | drop.no_host_pipeline: N in F frames (x/frame), top 10
    drop report |     count (per frame) draw: <why> <shader hashes>

`<why>`: `not made yet (queue state, waited ms)`, `shader does not translate:
<stage hash>`, `host pipeline refused`, `left out (<reason>)`, `has a host
pipeline, not run (bind or prepare refused)`, `no pipeline set`. Hashes are the
`stage+stage` shader hashes the pipeline logs use. `drop.indirect` lists dropped
`ExecuteIndirect` calls by signature (stride, argument layout), and
`drop.indirect.why` by the reason `State::execute_indirect` gave. All work in
every build (the older `dropped {why}` lines need `diagnostics`).

## Scope families in v7

* `list.*`: the title's list calls (`draw`, `draw_indexed`, `dispatch`,
  `dispatch_mesh`, `execute_indirect`, `close`, `reset`, `barrier`,
  `copy_*`, `set_pipeline`, `sync`, `wait_value`); `lock.list_state.wait`
  (contended only); fine: `list.state_call`.
* `draw.*`: `prepare` (per draw), `bind`, `tables`, `heap_buffers`,
  `heap_textures`; fine: `pipeline_lookup`.
* `pipe.*`: `make` (worker, per pipeline, calls = pipelines), `compile` (driver),
  `library.load`, `wait` (a draw waiting for a pipeline), `request`;
  `lock.pipe_queue.wait`; `shader.translate`; lists `slowest pipeline.make`,
  `slowest shader.translate`; `pipe.queue_depth` (average pipelines being made).
* `host.*`: `settle`, `settle_now`, `relay_wait`, `relay_card_back`,
  `relay_land`, `card_back`, `card_before`, `take_steps`, `read_backs`,
  `hand_back`, `refresh_front`, `refresh_ahead`, `run_deferred`, `flush_all`,
  `prelude`, `new_list`, `list_reset`, `present`, `execute_parts`, `place_copy`.
* `wait.*`: `line_drain`, `room_in_line`, `present_behind` (the title thread
  waiting on the submit thread at a present, by reason).
* `submit: ...`, `landing: ...`, `the title's ...` (timeline spans).
* `descriptor.*` (`sync_heaps`, `begin_sync_heaps`, `make_waiting`, `mirror`),
  `fence.*`, `compose.record`, `indirect.*`.
* `sub.*`, `prep.*`, `frame.*`, `tex.*`: the host's own counters handed over each
  frame (submission preparing and GPU wait, copy-back, copies, texture refresh,
  draws, drops).

## Reading a run quickly

1. `grep "stall:"` for the multi-second frames and what filled them.
2. Last `frame profile summary |` block: p50 against p90/max shows steady cost
   against spikes.
3. `drop report |` for what is not drawn and why.
4. `frames.csv` in a spreadsheet: sort on `wall_ms`, read across the row.
5. `trace.json` in Perfetto for the slowest frames: which thread waited for which.
