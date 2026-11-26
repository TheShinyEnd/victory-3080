//! Frame-time profiler: named scopes with atomic totals, aggregated per
//! frame and per log interval, optionally written as a per-frame CSV and a
//! Chrome-trace (Perfetto) file; plus a stall report, a long-block monitor,
//! drop tallies and "slowest" lists. See `docs/profiling.md`.
//!
//! A scope costs two clock reads, a thread-local touch and three relaxed
//! atomic adds when it closes (about 60 ns); nothing else runs until a
//! frame ends. No scope allocates or takes a lock on its hot path (a trace,
//! when asked for, pushes into the calling thread's own buffer). With
//! `VICTORY_PROFILE_OFF=1` spans do nothing at all (to measure that cost).
//!
//! * `VICTORY_PROFILE=<dir>`: writes `frames.csv` (a row per frame, a
//!   column per scope: ms, and `name#n` calls), `trace.json` (Chrome trace:
//!   open in `chrome://tracing` or ui.perfetto.dev) and `summary.txt`
//!   there. The files are written by the monitor thread, never the
//!   presenting one.
//! * `VICTORY_PROFILE_FRAMES=<a-b>`: the trace holds frames a to b (frame
//!   numbers count presents); `<n>` or unset: the n (default 4) slowest
//!   frames with more than 300 draws.
//! * `VICTORY_PROFILE_SUMMARY=1`: logs p50/p90/max per scope over the
//!   frames with more than 300 draws (the last block is the end-of-run
//!   table).
//! * `VICTORY_STALL_MS` (default 1000): a frame longer than this logs one
//!   `stall:` line naming its biggest scopes.
//! * `VICTORY_LONG_BLOCK_MS` (default 2000): a thread inside one outermost
//!   scope for longer is named in a `long block:` line while it is stuck.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

/// A named total. Declare with [`scope!`] / [`count!`], not by hand.
pub struct Scope {
    name: &'static str,
    time: bool,
    total: AtomicU64,
    calls: AtomicU64,
    max: AtomicU64,
    /// Column + 1 once registered.
    id: AtomicU32,
}

impl Scope {
    /// A scope measuring time (nanoseconds).
    pub const fn time(name: &'static str) -> Self {
        Self::new(name, true)
    }

    /// A scope summing a quantity (bytes, pages, calls).
    pub const fn counter(name: &'static str) -> Self {
        Self::new(name, false)
    }

    const fn new(name: &'static str, time: bool) -> Self {
        Scope {
            name,
            time,
            total: AtomicU64::new(0),
            calls: AtomicU64::new(0),
            max: AtomicU64::new(0),
            id: AtomicU32::new(0),
        }
    }

    /// Adds one call of `v` (nanoseconds for time scopes).
    #[inline]
    pub fn add(&'static self, v: u64) {
        if self.id.load(Relaxed) == 0 {
            register(self);
        }
        self.total.fetch_add(v, Relaxed);
        self.calls.fetch_add(1, Relaxed);
        self.max.fetch_max(v, Relaxed);
    }
}

/// A scope measuring time, until dropped: `let _p = prof::scope!("x.y");`.
#[macro_export]
macro_rules! prof_scope {
    ($name:literal) => {{
        static S: $crate::prof::Scope = $crate::prof::Scope::time($name);
        $crate::prof::Span::new(&S)
    }};
}

/// Adds a quantity to a counter scope: `prof::count!("x.bytes", n)`.
#[macro_export]
macro_rules! prof_count {
    ($name:literal, $n:expr) => {{
        static S: $crate::prof::Scope = $crate::prof::Scope::counter($name);
        S.add($n as u64);
    }};
}

/// As [`scope!`], but only while fine-grained scopes are on
/// (`VICTORY_PROFILE_FINE=1`, or `VICTORY_PROFILE=<dir>`): for calls made tens
/// of thousands of times a frame (every draw-state call), where the clock
/// reads would cost a few percent.
#[macro_export]
macro_rules! prof_fine {
    ($name:literal) => {{
        static S: $crate::prof::Scope = $crate::prof::Scope::time($name);
        $crate::prof::Span::fine(&S)
    }};
}

/// Counts one hit for `key` (a pointer or hash) in a named tally; `label`
/// (run once, when the key first appears in the interval) says what the key
/// is. Reported once per interval, most hits first:
/// `prof::tally!("drop.draw", key, || format!("..."))`.
#[macro_export]
macro_rules! prof_tally {
    ($name:literal, $key:expr, $label:expr) => {{
        static T: $crate::prof::Tally = $crate::prof::Tally::new($name);
        T.hit($key as usize, $label);
    }};
}

/// Offers one measured thing (`ns` nanoseconds) to a named "slowest" list;
/// `label` runs only if it makes the list:
/// `prof::top!("pipeline.make", ns, || hashes)`.
#[macro_export]
macro_rules! prof_top {
    ($name:literal, $ns:expr, $label:expr) => {{
        static T: $crate::prof::Top = $crate::prof::Top::new($name);
        T.note($ns as u64, $label);
    }};
}

pub use crate::{prof_count as count, prof_fine as fine, prof_scope as scope, prof_tally as tally, prof_top as top};

fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PROFILE_OFF").is_none())
}

fn epoch() -> Instant {
    static E: OnceLock<Instant> = OnceLock::new();
    *E.get_or_init(Instant::now)
}

fn env_ms(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default)
}

/// Whether fine-grained scopes are on.
fn fine_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        enabled()
            && (std::env::var_os("VICTORY_PROFILE").is_some_and(|v| !v.is_empty())
                || std::env::var("VICTORY_PROFILE_FINE").is_ok_and(|v| v != "0" && !v.is_empty()))
    })
}

static TRACING: AtomicBool = AtomicBool::new(false);
/// Events shorter than this are not traced (nanoseconds).
const TRACE_MIN_NS: u64 = 10_000;

#[derive(Clone, Copy)]
struct Event {
    col: u32,
    start_ns: u64,
    dur_ns: u64,
}

struct Thread {
    name: &'static str,
    /// Nanoseconds in outermost scopes since the last frame end.
    busy: AtomicU64,
    events: Mutex<Vec<Event>>,
    /// The thread's column of busy time.
    col: u32,
    /// The outermost scope the thread is in now (a `*const Scope`, 0 if
    /// none) and when it began (ns since the epoch): the long-block monitor.
    cur: AtomicUsize,
    cur_start: AtomicU64,
    /// The innermost scope open now (0 if none).
    inner: AtomicUsize,
    warned: AtomicU64,
}

thread_local! {
    static DEPTH: Cell<u32> = const { Cell::new(0) };
    static THREAD: OnceLock<Arc<Thread>> = const { OnceLock::new() };
}

fn init_thread() -> Arc<Thread> {
    let cur = std::thread::current();
    let name: &'static str = Box::leak(
        match cur.name() {
            Some(n) => format!("thread.{n}#{:?}", cur.id()),
            None => format!("thread.#{:?}", cur.id()),
        }
        .into_boxed_str(),
    );
    let mut reg = REG.lock().unwrap_or_else(|e| e.into_inner());
    let col = reg.column(name, true);
    let t = Arc::new(Thread {
        name,
        busy: AtomicU64::new(0),
        events: Mutex::new(Vec::new()),
        col: col as u32,
        cur: AtomicUsize::new(0),
        cur_start: AtomicU64::new(0),
        inner: AtomicUsize::new(0),
        warned: AtomicU64::new(0),
    });
    reg.threads.push(t.clone());
    t
}

fn with_thread<R>(f: impl FnOnce(&Arc<Thread>) -> R) -> Option<R> {
    THREAD.try_with(|t| f(t.get_or_init(init_thread))).ok()
}

fn scope_name(p: usize) -> &'static str {
    if p == 0 {
        return "";
    }
    // SAFETY: a pointer to a `static` scope stored by `Span::new`.
    unsafe { (*(p as *const Scope)).name }
}

/// What the monitor and the stall report do each second or so: names
/// threads stuck in one scope call, and writes the files.
fn start_monitor() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let _ = std::thread::Builder::new().name("victory-prof-monitor".into()).spawn(|| {
            let limit_ns = env_ms("VICTORY_LONG_BLOCK_MS", 2000).max(1) * 1_000_000;
            loop {
                std::thread::sleep(std::time::Duration::from_millis(250));
                let now = epoch().elapsed().as_nanos() as u64;
                let threads = REG.lock().unwrap_or_else(|e| e.into_inner()).threads.clone();
                for t in threads {
                    let (p, start) = (t.cur.load(Relaxed), t.cur_start.load(Relaxed));
                    if p == 0 || now.saturating_sub(start) < limit_ns || t.warned.load(Relaxed) == start {
                        continue;
                    }
                    if t.cur.load(Relaxed) != p {
                        continue;
                    }
                    t.warned.store(start, Relaxed);
                    let (outer, inner) = (scope_name(p), scope_name(t.inner.load(Relaxed)));
                    tracing::warn!(
                        "long block: thread {} inside scope `{outer}`{} for {:.1} s and counting",
                        t.name,
                        if inner == outer || inner.is_empty() { String::new() } else { format!(" (now in `{inner}`)") },
                        now.saturating_sub(start) as f64 / 1e9
                    );
                }
                if FLUSH.swap(false, Relaxed) {
                    flush();
                }
            }
        });
    });
}

/// Set by [`report`]: the monitor thread writes the files and the summary.
static FLUSH: AtomicBool = AtomicBool::new(false);

static NONE: Scope = Scope::time("none");

/// The scope of `name`: one per distinct text.
#[cold]
fn named_scope(name: &'static str) -> &'static Scope {
    static MADE: Mutex<BTreeMap<&'static str, &'static Scope>> = Mutex::new(BTreeMap::new());
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    made.entry(name).or_insert_with(|| Box::leak(Box::new(Scope::time(name))))
}

/// Times its life as one call of a scope.
pub struct Span {
    scope: &'static Scope,
    start: Option<Instant>,
    /// The scope this one was opened inside (0: none).
    prev: usize,
}

impl Span {
    /// A span that does nothing unless fine scopes are on ([`prof_fine!`]).
    #[inline]
    pub fn fine(scope: &'static Scope) -> Span {
        if fine_on() { Span::new(scope) } else { Span { scope, start: None, prev: 0 } }
    }

    /// A span of the scope named `name` (a string kept for the whole run),
    /// made on first use: for callers whose names are not literals at the
    /// call. A thread-local table finds it again without a lock.
    #[inline]
    pub fn named(name: &'static str) -> Span {
        if !enabled() {
            return Span { scope: &NONE, start: None, prev: 0 };
        }
        thread_local! {
            static BY_NAME: std::cell::RefCell<std::collections::HashMap<(usize, usize), &'static Scope>> =
                std::cell::RefCell::new(std::collections::HashMap::new());
        }
        let key = (name.as_ptr() as usize, name.len());
        let found = BY_NAME
            .try_with(|m| *m.borrow_mut().entry(key).or_insert_with(|| named_scope(name)))
            .unwrap_or(&NONE);
        Span::new(found)
    }

    #[inline]
    pub fn new(scope: &'static Scope) -> Span {
        if !enabled() {
            return Span { scope, start: None, prev: 0 };
        }
        let depth = DEPTH.with(|d| {
            d.set(d.get() + 1);
            d.get()
        });
        let start = Instant::now();
        let me = std::ptr::from_ref(scope) as usize;
        let prev = with_thread(|t| {
            if depth == 1 {
                t.cur_start.store(start.duration_since(epoch()).as_nanos() as u64, Relaxed);
                t.cur.store(me, Relaxed);
            }
            t.inner.swap(me, Relaxed)
        })
        .unwrap_or(0);
        Span { scope, start: Some(start), prev }
    }
}

impl Drop for Span {
    #[inline]
    fn drop(&mut self) {
        let Some(start) = self.start else { return };
        let ns = start.elapsed().as_nanos() as u64;
        self.scope.add(ns);
        let outer = DEPTH
            .try_with(|d| {
                d.set(d.get().saturating_sub(1));
                d.get() == 0
            })
            .unwrap_or(false);
        let tracing = TRACING.load(Relaxed) && ns >= TRACE_MIN_NS;
        with_thread(|t| {
            t.inner.store(self.prev, Relaxed);
            if outer {
                t.busy.fetch_add(ns, Relaxed);
                t.cur.store(0, Relaxed);
            }
            if tracing {
                let col = self.scope.id.load(Relaxed) - 1;
                let start_ns = start.duration_since(epoch()).as_nanos() as u64;
                let mut ev = t.events.lock().unwrap_or_else(|e| e.into_inner());
                if ev.len() < 400_000 {
                    ev.push(Event { col, start_ns, dur_ns: ns });
                }
            }
        });
    }
}

/// A mutex guard that counted its wait and its hold into two scopes.
pub struct Locked<'a, T> {
    guard: std::sync::MutexGuard<'a, T>,
    hold: &'static Scope,
    since: Option<Instant>,
}

/// Locks `m`, timing the wait into `wait` and the hold into `hold`.
#[inline]
pub fn lock<'a, T>(m: &'a Mutex<T>, wait: &'static Scope, hold: &'static Scope) -> Locked<'a, T> {
    if !enabled() {
        return Locked { guard: m.lock().unwrap_or_else(|e| e.into_inner()), hold, since: None };
    }
    let t = Instant::now();
    let guard = m.lock().unwrap_or_else(|e| e.into_inner());
    let since = Instant::now();
    wait.add(since.duration_since(t).as_nanos() as u64);
    Locked { guard, hold, since: Some(since) }
}

/// Locks `m`; when another thread holds it, the wait goes into `wait`. An
/// uncontended lock reads no clock.
#[inline]
pub fn lock_waiting<'a, T>(m: &'a Mutex<T>, wait: &'static Scope) -> std::sync::MutexGuard<'a, T> {
    match m.try_lock() {
        Ok(g) => g,
        Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
        Err(std::sync::TryLockError::WouldBlock) => {
            let _p = Span::new(wait);
            m.lock().unwrap_or_else(|e| e.into_inner())
        }
    }
}

/// `prof::waits!(MUTEX, "x.wait")`: [`lock_waiting`] with its scope.
#[macro_export]
macro_rules! prof_waits {
    ($m:expr, $wait:literal) => {{
        static W: $crate::prof::Scope = $crate::prof::Scope::time($wait);
        $crate::prof::lock_waiting(&$m, &W)
    }};
}
pub use crate::prof_waits as waits;

/// `prof::locked!(MUTEX, "x.wait", "x.hold")`: [`lock`] with its two scopes.
#[macro_export]
macro_rules! prof_locked {
    ($m:expr, $wait:literal, $hold:literal) => {{
        static W: $crate::prof::Scope = $crate::prof::Scope::time($wait);
        static H: $crate::prof::Scope = $crate::prof::Scope::time($hold);
        $crate::prof::lock(&$m, &W, &H)
    }};
}
pub use crate::prof_locked as locked;

impl<T> std::ops::Deref for Locked<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.guard
    }
}

impl<T> std::ops::DerefMut for Locked<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.guard
    }
}

impl<T> Drop for Locked<'_, T> {
    fn drop(&mut self) {
        if let Some(s) = self.since {
            self.hold.add(s.elapsed().as_nanos() as u64);
        }
    }
}

const TALLY_SLOTS: usize = 256;
const TALLY_PROBES: usize = 16;

/// Hits per key (a pointer or hash) with a label per key: a fixed table of
/// atomics, so a hit allocates nothing and takes no lock. Keys beyond the
/// table's room are counted as "other".
pub struct Tally {
    name: &'static str,
    registered: AtomicBool,
    keys: [AtomicUsize; TALLY_SLOTS],
    counts: [AtomicU64; TALLY_SLOTS],
    other: AtomicU64,
    labels: Mutex<Vec<(usize, String)>>,
}

impl Tally {
    pub const fn new(name: &'static str) -> Self {
        Tally {
            name,
            registered: AtomicBool::new(false),
            keys: [const { AtomicUsize::new(0) }; TALLY_SLOTS],
            counts: [const { AtomicU64::new(0) }; TALLY_SLOTS],
            other: AtomicU64::new(0),
            labels: Mutex::new(Vec::new()),
        }
    }

    /// One hit for `key`; `label` runs when the key first appears.
    #[inline]
    pub fn hit(&'static self, key: usize, label: impl FnOnce() -> String) {
        if !enabled() {
            return;
        }
        if !self.registered.load(Relaxed) && !self.registered.swap(true, Relaxed) {
            REG.lock().unwrap_or_else(|e| e.into_inner()).tallies.push(self);
        }
        let key = key.max(1);
        let mut i = key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 7;
        for _ in 0..TALLY_PROBES {
            i %= TALLY_SLOTS;
            let k = self.keys[i].load(Relaxed);
            if k == key {
                self.counts[i].fetch_add(1, Relaxed);
                return;
            }
            if k == 0 {
                match self.keys[i].compare_exchange(0, key, Relaxed, Relaxed) {
                    Ok(_) => {
                        let text = label();
                        self.labels.lock().unwrap_or_else(|e| e.into_inner()).push((key, text));
                        self.counts[i].fetch_add(1, Relaxed);
                        return;
                    }
                    Err(now) if now == key => {
                        self.counts[i].fetch_add(1, Relaxed);
                        return;
                    }
                    Err(_) => {}
                }
            }
            i += 1;
        }
        self.other.fetch_add(1, Relaxed);
    }

    /// The hits since the last call, most first, with the count of those
    /// that found no room.
    pub fn take(&self) -> (Vec<(String, u64)>, u64) {
        let labels = std::mem::take(&mut *self.labels.lock().unwrap_or_else(|e| e.into_inner()));
        let mut out = Vec::new();
        for i in 0..TALLY_SLOTS {
            let n = self.counts[i].swap(0, Relaxed);
            let key = self.keys[i].swap(0, Relaxed);
            if key == 0 || n == 0 {
                continue;
            }
            let text = labels
                .iter()
                .rev()
                .find(|(k, _)| *k == key)
                .map_or_else(|| format!("{key:#x}"), |(_, t)| t.clone());
            out.push((text, n));
        }
        // The same label for several keys (a pipeline made again) adds up.
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out.dedup_by(|b, a| {
            let same = a.0 == b.0;
            if same {
                a.1 += b.1;
            }
            same
        });
        out.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        (out, self.other.swap(0, Relaxed))
    }
}

const TOP_KEPT: usize = 8;

/// The slowest few of a kind per interval (pipeline makes, say), with
/// labels. A note takes a lock only when it makes the list.
pub struct Top {
    name: &'static str,
    registered: AtomicBool,
    /// The least of a full list (0 while it has room).
    floor: AtomicU64,
    list: Mutex<Vec<(u64, String)>>,
}

impl Top {
    pub const fn new(name: &'static str) -> Self {
        Top { name, registered: AtomicBool::new(false), floor: AtomicU64::new(0), list: Mutex::new(Vec::new()) }
    }

    #[inline]
    pub fn note(&'static self, ns: u64, label: impl FnOnce() -> String) {
        if !enabled() || ns <= self.floor.load(Relaxed) {
            return;
        }
        if !self.registered.load(Relaxed) && !self.registered.swap(true, Relaxed) {
            REG.lock().unwrap_or_else(|e| e.into_inner()).tops.push(self);
        }
        let mut list = self.list.lock().unwrap_or_else(|e| e.into_inner());
        list.push((ns, label()));
        list.sort_by_key(|x| std::cmp::Reverse(x.0));
        list.truncate(TOP_KEPT);
        if list.len() == TOP_KEPT {
            self.floor.store(list[TOP_KEPT - 1].0, Relaxed);
        }
    }

    pub fn take(&self) -> Vec<(u64, String)> {
        self.floor.store(0, Relaxed);
        std::mem::take(&mut *self.list.lock().unwrap_or_else(|e| e.into_inner()))
    }
}

struct Col {
    name: &'static str,
    time: bool,
}

struct Reg {
    cols: Vec<Col>,
    by_name: BTreeMap<&'static str, usize>,
    scopes: Vec<(usize, &'static Scope)>,
    threads: Vec<Arc<Thread>>,
    tallies: Vec<&'static Tally>,
    tops: Vec<&'static Top>,
}

impl Reg {
    fn column(&mut self, name: &'static str, time: bool) -> usize {
        if let Some(&c) = self.by_name.get(name) {
            return c;
        }
        self.cols.push(Col { name, time });
        self.by_name.insert(name, self.cols.len() - 1);
        self.cols.len() - 1
    }
}

static REG: Mutex<Reg> = Mutex::new(Reg {
    cols: Vec::new(),
    by_name: BTreeMap::new(),
    scopes: Vec::new(),
    threads: Vec::new(),
    tallies: Vec::new(),
    tops: Vec::new(),
});

#[cold]
fn register(s: &'static Scope) {
    let mut reg = REG.lock().unwrap_or_else(|e| e.into_inner());
    if s.id.load(Relaxed) != 0 {
        return;
    }
    let col = reg.column(s.name, s.time);
    reg.scopes.push((col, s));
    s.id.store(col as u32 + 1, Relaxed);
}

/// A value measured elsewhere (counters kept by other modules), handed in
/// each frame as the amount since the last frame.
pub struct Extra {
    pub name: &'static str,
    pub time: bool,
    /// Nanoseconds for time, else the quantity.
    pub total: u64,
    pub calls: u64,
}

#[derive(Clone, Copy, Default)]
struct Cell3 {
    total: u64,
    calls: u64,
    max: u64,
}

/// One frame's row, only the columns that were not zero.
type Row = Vec<(u32, Cell3)>;

struct Kept {
    frame: u64,
    wall_ns: u64,
    end_ns: u64,
    events: Vec<(u32, Vec<Event>)>,
}

struct FrameRow {
    frame: u64,
    world: bool,
    wall_ns: u64,
    row: Row,
}

struct State {
    last: Option<Instant>,
    frame: u64,
    /// Since the last report.
    acc: Vec<Cell3>,
    acc_frames: u64,
    acc_wall_ns: u64,
    rows: Vec<FrameRow>,
    kept: Vec<Kept>,
    dirty: bool,
    full_said: bool,
}

static STATE: Mutex<State> = Mutex::new(State {
    last: None,
    frame: 0,
    acc: Vec::new(),
    acc_frames: 0,
    acc_wall_ns: 0,
    rows: Vec::new(),
    kept: Vec::new(),
    dirty: false,
    full_said: false,
});

const MAX_ROWS: usize = 20_000;

struct Config {
    dir: Option<std::path::PathBuf>,
    range: Option<(u64, u64)>,
    slowest: usize,
    summary: bool,
    stall_ns: u64,
}

fn config() -> &'static Config {
    static C: OnceLock<Config> = OnceLock::new();
    C.get_or_init(|| {
        let dir = std::env::var_os("VICTORY_PROFILE").filter(|v| !v.is_empty()).map(std::path::PathBuf::from);
        let frames = std::env::var("VICTORY_PROFILE_FRAMES").unwrap_or_default();
        let (mut range, mut slowest) = (None, 4);
        if let Some((a, b)) = frames.split_once('-') {
            if let (Ok(a), Ok(b)) = (a.trim().parse(), b.trim().parse()) {
                range = Some((a, b));
            }
        } else if let Ok(n) = frames.trim().parse() {
            slowest = n;
        }
        let summary = std::env::var("VICTORY_PROFILE_SUMMARY").is_ok_and(|v| v != "0" && !v.is_empty());
        if dir.is_some() && enabled() {
            TRACING.store(true, Relaxed);
        }
        Config { dir, range, slowest, summary, stall_ns: env_ms("VICTORY_STALL_MS", 1000).max(1) * 1_000_000 }
    })
}

/// Text about the world's state for the stall line (pipelines being made,
/// textures uploaded, video memory): set once by the code that knows.
static STALL_NOTE: OnceLock<fn() -> String> = OnceLock::new();

/// Registers the function whose text ends each `stall:` line.
pub fn set_stall_note(f: fn() -> String) {
    let _ = STALL_NOTE.set(f);
}

/// Call once at startup (or any time): reads the environment.
pub fn init() {
    let _ = epoch();
    let _ = config();
    if enabled() {
        start_monitor();
    }
}

/// The biggest `n` time scopes of a row, as `name 12.3 ms`; threads' busy
/// columns are left out.
fn top_scopes(row: &Row, cols: &[Col], n: usize) -> Vec<(&'static str, f64)> {
    let mut v: Vec<(&'static str, u64)> = row
        .iter()
        .filter_map(|(i, c)| {
            let col = cols.get(*i as usize)?;
            (col.time && !col.name.starts_with("thread.") && c.total > 0).then_some((col.name, c.total))
        })
        .collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    v.into_iter().take(n).map(|(name, ns)| (name, ms(ns as f64))).collect()
}

fn stall_line(frame: u64, wall_ns: u64, top: &[(&str, f64)], note: &str) -> String {
    let mut s = format!("stall: frame {frame} took {:.0} ms: ", ms(wall_ns as f64));
    if top.is_empty() {
        s.push_str("no scope was open");
    }
    for (i, (name, t)) in top.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        s.push_str(&format!("{name} {t:.1} ms"));
    }
    if !note.is_empty() {
        s.push_str(", ");
        s.push_str(note);
    }
    s
}

/// A frame (a present) ended: folds every scope's totals since the last
/// frame into the interval and the frame's row. `world` marks frames of
/// the game proper (many draws); `extras` are other counters' amounts
/// since the last frame (gauges too: they average over the interval).
pub fn frame_end(world: bool, extras: &[Extra]) {
    if !enabled() {
        return;
    }
    let cfg = config();
    let now = Instant::now();
    let mut st = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let wall = st.last.map_or(0, |l| now.duration_since(l).as_nanos() as u64);
    st.last = Some(now);
    st.frame += 1;
    let frame = st.frame;
    let mut dense = Vec::new();
    let threads = {
        let mut reg = REG.lock().unwrap_or_else(|e| e.into_inner());
        for e in extras {
            reg.column(e.name, e.time);
        }
        dense.resize(reg.cols.len(), Cell3::default());
        for &(col, s) in &reg.scopes {
            let c = &mut dense[col];
            c.total += s.total.swap(0, Relaxed);
            c.calls += s.calls.swap(0, Relaxed);
            c.max = c.max.max(s.max.swap(0, Relaxed));
        }
        for e in extras {
            let c = &mut dense[reg.by_name[e.name]];
            c.total += e.total;
            c.calls += e.calls;
            c.max = c.max.max(e.total);
        }
        reg.threads.clone()
    };
    for t in &threads {
        let busy = t.busy.swap(0, Relaxed);
        if let Some(c) = dense.get_mut(t.col as usize) {
            c.total += busy;
            c.calls += 1;
            c.max = c.max.max(busy);
        }
    }
    if st.acc.len() < dense.len() {
        st.acc.resize(dense.len(), Cell3::default());
    }
    for (a, c) in st.acc.iter_mut().zip(&dense) {
        a.total += c.total;
        a.calls += c.calls;
        a.max = a.max.max(c.max);
    }
    st.acc_frames += 1;
    st.acc_wall_ns += wall;
    let row: Row = dense
        .iter()
        .enumerate()
        .filter(|(_, c)| c.total != 0 || c.calls != 0)
        .map(|(i, c)| (i as u32, *c))
        .collect();
    if wall >= cfg.stall_ns {
        let cols: Vec<Col> = {
            let reg = REG.lock().unwrap_or_else(|e| e.into_inner());
            reg.cols.iter().map(|c| Col { name: c.name, time: c.time }).collect()
        };
        let note = STALL_NOTE.get().map(|f| f()).unwrap_or_default();
        tracing::warn!("{}", stall_line(frame, wall, &top_scopes(&row, &cols, 6), &note));
    }
    if cfg.dir.is_some() || cfg.summary {
        if st.rows.len() < MAX_ROWS {
            st.rows.push(FrameRow { frame, world, wall_ns: wall, row });
        } else if !st.full_said {
            st.full_said = true;
            tracing::warn!("profiler: {MAX_ROWS} frames recorded, no more rows are kept");
        }
    }
    // Trace events of this frame.
    if TRACING.load(Relaxed) {
        let mut events = Vec::new();
        let mut count = 0;
        for t in &threads {
            let mut ev = t.events.lock().unwrap_or_else(|e| e.into_inner());
            count += ev.len();
            events.push((t.col, std::mem::take(&mut *ev)));
        }
        let end_ns = now.duration_since(epoch()).as_nanos() as u64;
        let keep = match cfg.range {
            Some((a, b)) => (a..=b).contains(&frame),
            None => {
                world && count > 0 && {
                    st.kept.len() < cfg.slowest || st.kept.iter().map(|k| k.wall_ns).min().is_some_and(|m| wall > m)
                }
            }
        };
        if keep {
            let k = Kept { frame, wall_ns: wall, end_ns, events };
            if cfg.range.is_none() && st.kept.len() >= cfg.slowest {
                if let Some(i) = st.kept.iter().enumerate().min_by_key(|(_, k)| k.wall_ns).map(|(i, _)| i) {
                    st.kept.swap_remove(i);
                }
            }
            st.kept.push(k);
            st.dirty = true;
        }
    }
}

fn ms(ns: f64) -> f64 {
    ns / 1e6
}

/// The lines of the interval's `frame profile` block, every scope sorted
/// by time per frame (counters after, by size), then each thread's
/// unattributed time, the `slowest` lists and the drop tallies; clears the
/// interval and asks the monitor thread to write the files.
pub fn report() -> Vec<String> {
    if !enabled() {
        return Vec::new();
    }
    let cfg = config();
    let mut st = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let frames = st.acc_frames.max(1) as f64;
    let wall = st.acc_wall_ns as f64;
    let acc = std::mem::take(&mut st.acc);
    let acc_frames = st.acc_frames;
    st.acc_frames = 0;
    st.acc_wall_ns = 0;
    drop(st);
    let (cols, threads, tallies, tops) = {
        let reg = REG.lock().unwrap_or_else(|e| e.into_inner());
        (
            reg.cols.iter().map(|c| Col { name: c.name, time: c.time }).collect::<Vec<_>>(),
            reg.threads.clone(),
            reg.tallies.clone(),
            reg.tops.clone(),
        )
    };
    let mut out = interval_lines(&acc, &cols, acc_frames, frames, wall);
    for t in &threads {
        let busy = acc.get(t.col as usize).map_or(0, |c| c.total) as f64;
        if busy > 0.0 {
            out.push(format!(
                "frame profile | {:<34} in scopes {:>9.2} ms of {:.2} ms wall: unattributed {:.2} ms",
                t.name,
                ms(busy / frames),
                ms(wall / frames),
                ms((wall - busy).max(0.0) / frames)
            ));
        }
    }
    for t in tops {
        let list = t.take();
        if list.is_empty() {
            continue;
        }
        let items: Vec<String> = list.iter().map(|(ns, l)| format!("{l} {:.1} ms", ms(*ns as f64))).collect();
        out.push(format!("frame profile | slowest {}: {}", t.name, items.join("; ")));
    }
    for t in tallies {
        let (list, other) = t.take();
        let total: u64 = list.iter().map(|x| x.1).sum::<u64>() + other;
        if total == 0 {
            continue;
        }
        out.push(format!(
            "drop report | {}: {total} in {acc_frames} frames ({:.1}/frame), top {}",
            t.name,
            total as f64 / frames,
            list.len().min(10)
        ));
        for (label, n) in list.iter().take(10) {
            out.push(format!("drop report |   {n:>7} ({:>7.1}/frame) {label}", *n as f64 / frames));
        }
        if other > 0 {
            out.push(format!("drop report |   {other:>7} ({:>7.1}/frame) other (table full)", other as f64 / frames));
        }
    }
    if cfg.dir.is_some() || cfg.summary {
        FLUSH.store(true, Relaxed);
    }
    out
}

fn interval_lines(acc: &[Cell3], cols: &[Col], acc_frames: u64, frames: f64, wall: f64) -> Vec<String> {
    let mut times: Vec<(usize, &Cell3)> = Vec::new();
    let mut counts: Vec<(usize, &Cell3)> = Vec::new();
    for (i, c) in acc.iter().enumerate() {
        if (c.total == 0 && c.calls == 0) || cols[i].name.starts_with("thread.") {
            continue;
        }
        if cols[i].time { times.push((i, c)) } else { counts.push((i, c)) }
    }
    times.sort_by_key(|(_, c)| std::cmp::Reverse(c.total));
    counts.sort_by_key(|(_, c)| std::cmp::Reverse(c.total));
    let mut out = vec![format!(
        "frame profile | {acc_frames} frames, {:.1} ms/frame wall; scope: ms/frame, calls/frame, longest single frame total",
        ms(wall / frames)
    )];
    for (i, c) in times.iter().filter(|(_, c)| c.total as f64 / frames >= 5_000.0) {
        out.push(format!(
            "frame profile | {:<34} {:>9.2} ms {:>9.1} calls  max {:>8.2} ms",
            cols[*i].name,
            ms(c.total as f64 / frames),
            c.calls as f64 / frames,
            ms(c.max as f64)
        ));
    }
    for (i, c) in &counts {
        out.push(format!(
            "frame profile | {:<34} {:>12.1} per frame {:>9.1} calls  max {:>10.0}",
            cols[*i].name,
            c.total as f64 / frames,
            c.calls as f64 / frames,
            c.max as f64
        ));
    }
    out
}

fn percentile(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    v[(((v.len() - 1) as f64) * p).round() as usize]
}

/// What the files and the summary are made from: copied out of the state,
/// so the presenting thread does not wait for the formatting.
struct Snapshot {
    cols: Vec<Col>,
    rows: Vec<FrameRow>,
    threads: Vec<(u32, &'static str)>,
    kept: Option<Vec<Kept>>,
}

fn snapshot(with_trace: bool) -> Snapshot {
    let (cols, threads) = {
        let reg = REG.lock().unwrap_or_else(|e| e.into_inner());
        (
            reg.cols.iter().map(|c| Col { name: c.name, time: c.time }).collect(),
            reg.threads.iter().map(|t| (t.col, t.name)).collect(),
        )
    };
    let mut st = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let rows = st
        .rows
        .iter()
        .map(|r| FrameRow { frame: r.frame, world: r.world, wall_ns: r.wall_ns, row: r.row.clone() })
        .collect();
    let kept = (with_trace && st.dirty).then(|| {
        st.dirty = false;
        st.kept
            .iter()
            .map(|k| Kept { frame: k.frame, wall_ns: k.wall_ns, end_ns: k.end_ns, events: k.events.clone() })
            .collect()
    });
    Snapshot { cols, rows, threads, kept }
}

fn value(col: &Col, c: &Cell3) -> f64 {
    if col.time { ms(c.total as f64) } else { c.total as f64 }
}

/// `scope: p50 / p90 / max` per scope over the world frames, biggest p50 first.
fn summary_lines(snap: &Snapshot) -> Vec<String> {
    let world: Vec<&FrameRow> = snap.rows.iter().filter(|r| r.world).collect();
    let mut out = vec![format!(
        "frame profile summary | {} world frames (>300 draws): scope p50 / p90 / max ms (or quantity) per frame",
        world.len()
    )];
    if world.is_empty() {
        return out;
    }
    let mut per: Vec<Vec<f64>> = vec![vec![0.0; world.len()]; snap.cols.len()];
    for (w, r) in world.iter().enumerate() {
        for (i, c) in &r.row {
            if let Some(col) = snap.cols.get(*i as usize) {
                per[*i as usize][w] = value(col, c);
            }
        }
    }
    let mut lines: Vec<(f64, String)> = Vec::new();
    let mut wall: Vec<f64> = world.iter().map(|r| ms(r.wall_ns as f64)).collect();
    let (p50, p90, max) = (percentile(&mut wall, 0.5), percentile(&mut wall, 0.9), percentile(&mut wall, 1.0));
    lines.push((f64::MAX, format!("frame profile summary | {:<34} {p50:>9.1} / {p90:>9.1} / {max:>9.1} ms", "frame wall")));
    for (col, v) in snap.cols.iter().zip(per.iter_mut()) {
        let p50 = percentile(v, 0.5);
        let p90 = percentile(v, 0.9);
        let max = percentile(v, 1.0);
        if max <= 0.0 || (col.time && max < 0.01) || (col.name.starts_with("thread.") && p50 < 1.0) {
            continue;
        }
        lines.push((p50, format!("frame profile summary | {:<34} {p50:>9.2} / {p90:>9.2} / {max:>9.2}{}", col.name, if col.time { " ms" } else { "" })));
    }
    lines.sort_by(|a, b| b.0.total_cmp(&a.0));
    out.extend(lines.into_iter().map(|l| l.1));
    out
}

fn csv(snap: &Snapshot) -> String {
    use std::fmt::Write;
    let mut csv = String::from("frame,world,wall_ms");
    // (Scope names may hold commas and quotes: they are not in a header.)
    let plain = |n: &str| n.replace([',', '"', '\n'], ";");
    for c in &snap.cols {
        let _ = write!(csv, ",{}", plain(c.name));
    }
    for c in &snap.cols {
        let _ = write!(csv, ",{}#n", plain(c.name));
    }
    csv.push('\n');
    let n = snap.cols.len();
    let mut dense = vec![Cell3::default(); n];
    for r in &snap.rows {
        let _ = write!(csv, "{},{},{:.3}", r.frame, u8::from(r.world), ms(r.wall_ns as f64));
        dense.fill(Cell3::default());
        for (i, c) in &r.row {
            if let Some(d) = dense.get_mut(*i as usize) {
                *d = *c;
            }
        }
        for (col, c) in snap.cols.iter().zip(&dense) {
            let _ = write!(csv, ",{:.3}", value(col, c));
        }
        for c in &dense {
            let _ = write!(csv, ",{}", c.calls);
        }
        csv.push('\n');
    }
    csv
}

/// Chrome trace: complete events (`"ph":"X"`: a begin and an end in one).
fn trace_json(snap: &Snapshot, kept: &[Kept]) -> String {
    use std::fmt::Write;
    let mut j = String::from("{\"traceEvents\":[\n");
    j.push_str("{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":1,\"args\":{\"name\":\"victory\"}}");
    for (n, (_, name)) in snap.threads.iter().enumerate() {
        let _ = write!(j, ",\n{{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":{},\"args\":{{\"name\":\"{name}\"}}}}", n + 1);
    }
    j.push_str(",\n{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":0,\"args\":{\"name\":\"frames\"}}");
    let mut kept: Vec<&Kept> = kept.iter().collect();
    kept.sort_by_key(|k| k.frame);
    for k in kept {
        let _ = write!(
            j,
            ",\n{{\"name\":\"frame {}\",\"ph\":\"X\",\"pid\":1,\"tid\":0,\"ts\":{:.1},\"dur\":{:.1}}}",
            k.frame,
            (k.end_ns.saturating_sub(k.wall_ns)) as f64 / 1e3,
            k.wall_ns as f64 / 1e3
        );
        for (tcol, events) in &k.events {
            let tid = snap.threads.iter().position(|t| t.0 == *tcol).map_or(0, |p| p + 1);
            for e in events {
                let _ = write!(
                    j,
                    ",\n{{\"name\":\"{}\",\"ph\":\"X\",\"pid\":1,\"tid\":{tid},\"ts\":{:.1},\"dur\":{:.1}}}",
                    snap.cols[e.col as usize].name.replace('\\', "/").replace('"', "'"),
                    e.start_ns as f64 / 1e3,
                    e.dur_ns as f64 / 1e3
                );
            }
        }
    }
    j.push_str("\n]}\n");
    j
}

/// Writes the files and logs the summary now (the monitor thread does it
/// after each [`report`]).
pub fn flush() {
    if !enabled() {
        return;
    }
    let cfg = config();
    let snap = snapshot(cfg.dir.is_some());
    let summary = summary_lines(&snap);
    if cfg.summary {
        for l in &summary {
            tracing::info!("{l}");
        }
    }
    let Some(dir) = &cfg.dir else { return };
    let _ = std::fs::create_dir_all(dir);
    let _ = std::fs::write(dir.join("frames.csv"), csv(&snap));
    let _ = std::fs::write(dir.join("summary.txt"), summary.join("\n") + "\n");
    if let Some(kept) = &snap.kept {
        let _ = std::fs::write(dir.join("trace.json"), trace_json(&snap, kept));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `report` empties every tally: the tests that use them go one at a time.
    static SERIAL: Mutex<()> = Mutex::new(());

    #[test]
    fn scopes_sum_per_frame_and_the_report_lists_them_sorted() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        std::thread::spawn(|| {
            {
                let _p = scope!("test.slow");
                std::thread::sleep(std::time::Duration::from_millis(12));
            }
            for _ in 0..3 {
                let _p = scope!("test.fast");
            }
            count!("test.bytes", 100);
            count!("test.bytes", 50);
            frame_end(true, &[Extra { name: "test.extra", time: true, total: 1_000_000, calls: 2 }]);
            let lines = report();
            let slow = lines.iter().position(|l| l.contains("test.slow")).expect("slow");
            if let Some(extra) = lines.iter().position(|l| l.contains("test.extra")) {
                assert!(slow < extra, "{lines:?}");
            }
            assert!(lines.iter().any(|l| l.contains("test.bytes") && l.contains("150.0")), "{lines:?}");
            // Another report with nothing since: the scopes are zero.
            frame_end(true, &[]);
            let lines = report();
            assert!(!lines.iter().any(|l| l.contains("test.slow")), "{lines:?}");
        })
        .join()
        .unwrap();
    }

    #[test]
    fn the_stall_line_names_the_top_scopes_and_the_note() {
        let cols = vec![
            Col { name: "a.small", time: true },
            Col { name: "b.big", time: true },
            Col { name: "thread.x#1", time: true },
            Col { name: "c.bytes", time: false },
            Col { name: "d.mid", time: true },
        ];
        let c = |total| Cell3 { total, calls: 1, max: total };
        let row: Row = vec![(0, c(1_000_000)), (1, c(900_000_000)), (2, c(2_000_000_000)), (3, c(5_000_000_000)), (4, c(50_000_000))];
        let top = top_scopes(&row, &cols, 6);
        assert_eq!(top.iter().map(|t| t.0).collect::<Vec<_>>(), ["b.big", "d.mid", "a.small"]);
        let line = stall_line(42, 1_234_000_000, &top, "pipelines being made 3->9");
        assert_eq!(
            line,
            "stall: frame 42 took 1234 ms: b.big 900.0 ms, d.mid 50.0 ms, a.small 1.0 ms, pipelines being made 3->9"
        );
    }

    #[test]
    fn a_tally_counts_per_key_labels_once_and_sorts() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        static T: Tally = Tally::new("test.tally");
        let mut labelled = 0;
        for _ in 0..5 {
            T.hit(0x10, || {
                labelled += 1;
                "five".into()
            });
        }
        for _ in 0..9 {
            T.hit(0x20, || "nine".into());
        }
        T.hit(0, || "zero key".into());
        assert_eq!(labelled, 1);
        let (list, other) = T.take();
        assert_eq!(other, 0);
        assert_eq!(list[0], ("nine".to_owned(), 9));
        assert_eq!(list[1], ("five".to_owned(), 5));
        assert_eq!(list[2], ("zero key".to_owned(), 1));
        // Cleared: the next interval starts empty and labels again.
        T.hit(0x10, || "five again".into());
        assert_eq!(T.take().0, vec![("five again".to_owned(), 1)]);
    }

    #[test]
    fn a_full_tally_counts_the_rest_as_other() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        static T: Tally = Tally::new("test.full");
        // Many more keys than slots: some find no room within their probes.
        for k in 1..2000usize {
            T.hit(k * 4096, || format!("k{k}"));
        }
        let (list, other) = T.take();
        assert!(other > 0);
        assert_eq!(list.iter().map(|x| x.1).sum::<u64>() + other, 1999);
    }

    #[test]
    fn the_slowest_list_keeps_the_biggest_few() {
        static T: Top = Top::new("test.top");
        for i in 1..=20u64 {
            T.note(i * 1000, || format!("n{i}"));
        }
        let list = T.take();
        assert_eq!(list.len(), TOP_KEPT);
        assert_eq!(list[0].1, "n20");
        assert_eq!(list[TOP_KEPT - 1].1, "n13");
        // After a take the list fills again from nothing.
        T.note(1, || "tiny".into());
        assert_eq!(T.take(), vec![(1, "tiny".to_owned())]);
    }

    #[test]
    fn spans_named_at_run_time_share_one_scope_per_text() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        std::thread::spawn(|| {
            for _ in 0..2 {
                let _p = Span::named("test.named span");
            }
            frame_end(false, &[]);
            let lines = report();
            assert!(
                lines.iter().any(|l| l.contains("test.named span") && l.contains("2.0 calls")),
                "{lines:?}"
            );
        })
        .join()
        .unwrap();
    }

    #[test]
    fn percentiles_pick_from_the_sorted_values() {
        let mut v = vec![5.0, 1.0, 3.0, 2.0, 4.0];
        assert_eq!(percentile(&mut v, 0.0), 1.0);
        assert_eq!(percentile(&mut v, 0.5), 3.0);
        assert_eq!(percentile(&mut v, 1.0), 5.0);
        assert_eq!(percentile(&mut [], 0.5), 0.0);
    }

    #[test]
    fn csv_and_summary_cover_world_frames_only() {
        let cols = vec![Col { name: "x.time", time: true }, Col { name: "x.bytes", time: false }];
        let cell = |total, calls| Cell3 { total, calls, max: total };
        let mut rows = Vec::new();
        for (frame, world, t) in [(1, true, 10_000_000u64), (2, false, 99_000_000), (3, true, 30_000_000), (4, true, 20_000_000)] {
            rows.push(FrameRow { frame, world, wall_ns: t, row: vec![(0, cell(t, 2)), (1, cell(100, 1))] });
        }
        let snap = Snapshot { cols, rows, threads: vec![], kept: None };
        let text = csv(&snap);
        let mut lines = text.lines();
        assert_eq!(lines.next().unwrap(), "frame,world,wall_ms,x.time,x.bytes,x.time#n,x.bytes#n");
        assert_eq!(lines.next().unwrap(), "1,1,10.000,10.000,100.000,2,1");
        assert_eq!(text.lines().count(), 5);
        let s = summary_lines(&snap);
        assert!(s[0].contains("3 world frames"), "{s:?}");
        // The wall row: p50 20, p90 30 (nearest rank), max 30.
        assert!(s[1].contains("20.0 /      30.0 /      30.0 ms"), "{s:?}");
        assert!(s.iter().any(|l| l.contains("x.time") && l.contains("20.00 / ")), "{s:?}");
        assert!(s.iter().any(|l| l.contains("x.bytes") && l.contains("100.00")), "{s:?}");
    }
}
