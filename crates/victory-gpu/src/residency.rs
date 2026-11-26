//! Host-memory admission for speculative texture work and reusable upload storage.
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;
static PRESSURED: AtomicBool = AtomicBool::new(false);
static HEADROOM: AtomicU64 = AtomicU64::new(u64::MAX);

fn memory_status() -> Option<MEMORYSTATUSEX> {
    // SAFETY: the API initializes the fields of a correctly sized structure.
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    (unsafe { GlobalMemoryStatusEx(&mut status) } != 0).then_some(status)
}

// The texture sweep may run for seconds inside a single submitted frame.
static NEXT_COMMIT_SAMPLE: AtomicU64 = AtomicU64::new(0);
static AVAILABLE_COMMIT: AtomicU64 = AtomicU64::new(u64::MAX);
static RETRY_UPLOAD_AT: AtomicU64 = AtomicU64::new(0);

pub(crate) fn allocation_error(error: &windows::core::Error) {
    if error.code() == windows::Win32::Foundation::E_OUTOFMEMORY {
        RETRY_UPLOAD_AT.fetch_max(crate::host::coarse_ms().saturating_add(500), Ordering::Relaxed);
    }
}

thread_local! {
    static STAGING_DEFERRED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(crate) fn note_staging_deferred() {
    STAGING_DEFERRED.set(true);
}

pub(crate) fn take_staging_deferred() -> bool {
    STAGING_DEFERRED.replace(false)
}

pub(crate) fn upload_refused() -> bool {
    crate::host::coarse_ms() < RETRY_UPLOAD_AT.load(Ordering::Relaxed)
}

fn sample_commit(now: u64) -> u64 {
    let next = NEXT_COMMIT_SAMPLE.load(Ordering::Relaxed);
    if now >= next
        && NEXT_COMMIT_SAMPLE
            .compare_exchange(next, now.saturating_add(32), Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        && let Some(status) = memory_status()
    {
        AVAILABLE_COMMIT.store(status.ullAvailPageFile, Ordering::Relaxed);
    }
    AVAILABLE_COMMIT.load(Ordering::Relaxed)
}

fn has_upload_reserve(commit: u64, bake: bool) -> bool {
    // Keep a protective floor against Rust/game OOM aborts without starving
    // bindless world textures or one-shot ped decoration bakes when steady-state
    // commit headroom sits in the 1-4 GiB range on 32 GiB hosts.
    commit >= if bake { 256 * MIB } else { 512 * MIB }
}

fn has_growth_reserve(commit: u64, bake: bool) -> bool {
    // Growing a fresh 64 MiB upload block for speculative bindless prefetch
    // requires more headroom than reusing slices already committed inside the
    // staging pool.
    commit >= if bake { 256 * MIB } else { 1024 * MIB }
}

/// Leave commit for game allocations and submissions already in flight while
/// still letting bindless textures and one-shot material bakes upload.
pub(crate) fn admit_texture_pass(bake: bool) -> bool {
    let now = crate::host::coarse_ms();
    if now < RETRY_UPLOAD_AT.load(Ordering::Relaxed) {
        return false;
    }
    has_upload_reserve(sample_commit(now), bake)
}

/// Whether the staging allocator may commit a brand-new OS upload buffer
/// (reusing existing free ranges in already-committed pool blocks stays open
/// whenever `admit_texture_pass` is true).
pub(crate) fn allow_staging_growth(bake: bool) -> bool {
    let now = crate::host::coarse_ms();
    if now < RETRY_UPLOAD_AT.load(Ordering::Relaxed) {
        return false;
    }
    has_growth_reserve(sample_commit(now), bake)
}

pub(crate) fn limited_host_ram() -> bool {
    static LIMITED: OnceLock<bool> = OnceLock::new();
    *LIMITED.get_or_init(|| memory_status().is_some_and(|s| s.ullTotalPhys <= 32 * GIB))
}

fn next_pressure(was: bool, physical: u64, commit: u64) -> bool {
    if was {
        physical < 5 * GIB || commit < 6 * GIB
    } else {
        physical <= 3 * GIB || commit <= 4 * GIB
    }
}

/// Called by the once-per-second submission sweep. A failed sample keeps the last good state.
pub(crate) fn refresh() {
    let Some(status) = memory_status() else { return };
    let (physical, commit) = (status.ullAvailPhys, status.ullAvailPageFile);
    HEADROOM.store(physical.min(commit), Ordering::Relaxed);
    let was = pressured();
    let pressure = next_pressure(was, physical, commit);
    if pressure != was {
        PRESSURED.store(pressure, Ordering::Relaxed);
        if pressure {
            tracing::warn!(
                pid = std::process::id(),
                frame = crate::frame::current(),
                physical_free_mib = physical >> 20,
                available_commit_mib = commit >> 20,
                buffer_mirror_span_mib = crate::shadow::live_span_bytes() >> 20,
                texture_staging_mib = crate::texture::live_staging_bytes() >> 20,
                spare_scratch_mib = crate::host::ALIVE[4].load(Ordering::Relaxed) >> 20,
                game_filled_gpu_texture_mib = crate::resource::KINDS[1].load(Ordering::Relaxed) >> 20,
                "host memory pressure: limiting texture prefetch and idle upload storage"
            );
        } else {
            tracing::info!(physical_free_mib = physical >> 20, available_commit_mib = commit >> 20, "host memory pressure cleared");
        }
    }
}

pub(crate) fn headroom() -> Option<u64> {
    let bytes = HEADROOM.load(Ordering::Relaxed);
    (bytes != u64::MAX).then_some(bytes)
}

pub(crate) fn pressured() -> bool {
    PRESSURED.load(Ordering::Relaxed)
}

/// True when OS commit headroom is critically low (< 1.5 GiB) or an allocation
/// just failed with `E_OUTOFMEMORY`, requiring emergency throttling.
pub(crate) fn critical_commit() -> bool {
    let now = crate::host::coarse_ms();
    now < RETRY_UPLOAD_AT.load(Ordering::Relaxed) || sample_commit(now) < 1536 * MIB
}

/// A soft per-frame admission limit. A single texture can cross it. The
/// queue retains unserved slots so they can be retried on later frames.
pub(crate) fn prefetch_limit(bytes: u64, pressure: bool, critical: bool, limited: bool, staging: u64) -> u64 {
    if critical {
        bytes.min(16 << 20)
    } else if pressure || (limited && staging >= GIB) {
        bytes.min(64 << 20)
    } else {
        bytes
    }
}

// ---------------------------------------------------------------------------
// Card (video memory) residency of title textures.
//
// The base derived its limits from constants written for a 32 GB card (a 2 GiB
// "floor" of host copies that is always admitted, a 3 GiB reserve for render
// targets, nine tenths of the budget as the admission line) and refused every
// host copy over the line outright: a refused slot of a texture picked by
// index shows nothing (null) until the card has room, which on a 10 GB card is
// never in a busy scene (the main character's head).  This policy is derived
// from the budget the system reports (`QueryVideoMemoryInfo`, local group):
//
//   admit_line  budget - reserve         new host copies are made below it
//   hard_line   budget - reserve / 2     nothing is admitted past it (no spill)
//   evict_hi    admit_line - budget/32   eviction of idle copies starts here
//   evict_lo    admit_line - budget/12   ... and stops here (hysteresis)
//   floor       min(budget/8, 1 GiB)     copies always admitted (below hard_line)
//   small       8 MiB                    copies admitted past admit_line
//
// `VICTORY_RESIDENCY=legacy` restores the base rules. Switches (auto mode):
//   VICTORY_RESIDENCY_RESERVE_MB  reserve below the budget (max(512, budget/16))
//   VICTORY_RESIDENCY_FLOOR_MB    always-admitted copies (min(budget/8, 1024))
//   VICTORY_RESIDENCY_SMALL_MB    "small" texture size (8)
//   VICTORY_RESIDENCY_EVICT_FRAMES frames a copy must be unused to be evicted (30)
//   VICTORY_RESIDENCY_STAGE_MB / _STAGE_MS  upload bytes and milliseconds per frame
//   VICTORY_TEXTURE_CAP_MB        top mips of a bigger host copy are dropped (0: none)
//   VICTORY_RESIDENCY_FIRST       log the first N refused textures (16)
//   VICTORY_RESIDENCY_TRACE       list the N largest non-resident textures at each line
// ---------------------------------------------------------------------------

/// `VICTORY_RESIDENCY=legacy` (also `base`, `off`, `0`): the base's rules.
pub(crate) fn legacy() -> bool {
    static LEGACY: OnceLock<bool> = OnceLock::new();
    *LEGACY.get_or_init(|| {
        matches!(std::env::var("VICTORY_RESIDENCY").as_deref().map(str::trim), Ok("legacy" | "base" | "off" | "0"))
    })
}

fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok()?.trim().parse::<u64>().ok()
}

fn env_mb(name: &str) -> Option<u64> {
    env_u64(name).map(|mb| mb.min(1 << 20) << 20)
}

/// The lines of the card policy for one budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Policy {
    pub budget: u64,
    pub admit_line: u64,
    pub hard_line: u64,
    pub evict_hi: u64,
    pub evict_lo: u64,
    pub floor: u64,
    pub small: u64,
}

/// How a request for a new host copy was answered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Admit {
    /// Under the admission line.
    Yes,
    /// Over it, but the host copies of title textures are under the floor.
    Floor,
    /// Over it, but a small texture.
    Small,
    /// Refused: over the line, or the copy would pass the hard line.
    No,
}

impl Policy {
    pub(crate) fn derive(budget: u64, reserve: Option<u64>, floor: Option<u64>, small: Option<u64>) -> Policy {
        let reserve = reserve.unwrap_or_else(|| (budget / 16).max(512 * MIB)).min(budget / 4);
        let admit_line = budget - reserve;
        Policy {
            budget,
            admit_line,
            hard_line: budget - reserve / 2,
            evict_hi: admit_line - budget / 32,
            evict_lo: admit_line - budget / 12,
            floor: floor.unwrap_or_else(|| (budget / 8).min(GIB)),
            small: small.unwrap_or(8 * MIB).min(budget / 64),
        }
    }

    /// `used`: the card's memory in use counted as the admission counts it;
    /// `filled`: bytes of host copies of title-filled textures; `bytes`: what
    /// the new copy takes.
    pub(crate) fn admit(&self, used: u64, filled: u64, bytes: u64) -> Admit {
        if used.saturating_add(bytes) > self.hard_line {
            Admit::No
        } else if used < self.admit_line {
            Admit::Yes
        } else if filled < self.floor {
            Admit::Floor
        } else if bytes > 0 && bytes <= self.small {
            Admit::Small
        } else {
            Admit::No
        }
    }

    /// Eviction switches on at `evict_hi` (or when admissions starve) and
    /// off again below `evict_lo`.
    pub(crate) fn evicting(&self, was: bool, used: u64, starved: bool) -> bool {
        if was { used > self.evict_lo || starved } else { used >= self.evict_hi || starved }
    }

    /// What may be let go while evicting: (idle milliseconds, idle frames,
    /// most bytes in one sweep). Not evicting: nothing for ten minutes.
    pub(crate) fn evict_plan(&self, evicting: bool, used: u64, starved: bool, frames: u64) -> (u64, u64, u64) {
        if !evicting {
            return (600_000, 4, u64::MAX);
        }
        let urgent = used >= self.admit_line || starved;
        let most = used.saturating_sub(self.evict_lo).max(64 * MIB);
        if urgent { (500, (frames / 3).max(4), most) } else { (1_500, frames.max(4), most) }
    }
}

/// The policy for the budget the system reports (with the env overrides).
pub(crate) fn policy(budget: u64) -> Policy {
    static OVERRIDES: OnceLock<[Option<u64>; 3]> = OnceLock::new();
    let [reserve, floor, small] = *OVERRIDES.get_or_init(|| {
        [
            env_mb("VICTORY_RESIDENCY_RESERVE_MB"),
            env_mb("VICTORY_RESIDENCY_FLOOR_MB"),
            env_mb("VICTORY_RESIDENCY_SMALL_MB"),
        ]
    });
    Policy::derive(budget, reserve, floor, small)
}

/// Frames a host copy must be unused to be evicted under pressure.
pub(crate) fn evict_frames() -> u64 {
    static N: OnceLock<u64> = OnceLock::new();
    *N.get_or_init(|| env_u64("VICTORY_RESIDENCY_EVICT_FRAMES").unwrap_or(30).clamp(4, 100_000))
}

static EVICTING: AtomicBool = AtomicBool::new(false);
static STARVED_AT: AtomicU64 = AtomicU64::new(0);

/// A host copy was refused for the budget: eviction runs harder for a while.
pub(crate) fn note_starved() {
    STARVED_AT.store(crate::host::coarse_ms().saturating_add(1), Ordering::Relaxed);
}

pub(crate) fn starved() -> bool {
    let at = STARVED_AT.load(Ordering::Relaxed);
    at != 0 && crate::host::coarse_ms() < at.saturating_add(3_000)
}

/// For the sweep of idle host copies: (idle ms, idle frames, most bytes).
pub(crate) fn eviction(budget: u64, used: u64) -> (u64, u64, u64) {
    let p = policy(budget);
    let starved = starved();
    let was = EVICTING.load(Ordering::Relaxed);
    let now = p.evicting(was, used, starved);
    if now != was {
        EVICTING.store(now, Ordering::Relaxed);
        tracing::info!(
            evicting = now,
            used_mb = used >> 20,
            budget_mb = budget >> 20,
            evict_hi_mb = p.evict_hi >> 20,
            evict_lo_mb = p.evict_lo >> 20,
            starved,
            "residency: eviction of idle texture copies {}",
            if now { "begins" } else { "ends" }
        );
    }
    p.evict_plan(now, used, starved, evict_frames())
}

/// Upload bytes and milliseconds one frame's pass over textures picked by
/// index may spend (a backlog doubles both). Spread over frames, a new
/// area's textures arrive in a second or two instead of one long stall.
pub(crate) fn upload_budget(budget: Option<u64>, backlog: bool) -> (u64, u64) {
    let bytes = env_mb("VICTORY_RESIDENCY_STAGE_MB").unwrap_or_else(|| budget.map_or(96 * MIB, |b| (b / 96).clamp(48 * MIB, 192 * MIB)));
    let ms = env_u64("VICTORY_RESIDENCY_STAGE_MS").unwrap_or(60);
    if backlog { (bytes * 2, ms * 2) } else { (bytes, ms) }
}

/// Biggest host copy (bytes from its first level) of a placed texture
/// before its top levels are dropped; `u64::MAX`: no cap. Fixed once.
pub(crate) fn mip_cap_bytes() -> u64 {
    static CAP: OnceLock<u64> = OnceLock::new();
    if let Some(cap) = CAP.get() {
        return *cap;
    }
    let cap = match env_u64("VICTORY_TEXTURE_CAP_MB") {
        Some(0) => u64::MAX,
        Some(mb) => mb.min(1 << 20) << 20,
        // Off unless asked for: starting a copy below its top level changed
        // the picture (colour banding over the whole frame, 2026-11-26: a
        // big 2D texture read at its top level, noise or a table, was shown
        // from an averaged level). `VICTORY_TEXTURE_CAP_MB=auto` sizes the
        // cap from the card's budget as before.
        None if legacy() || std::env::var("VICTORY_TEXTURE_CAP_MB").as_deref().map(str::trim) != Ok("auto") => {
            return *CAP.get_or_init(|| u64::MAX);
        }
        None => match crate::host::host().and_then(|h| h.video_memory()) {
            Some([budget, ..]) if budget > 0 => cap_for_budget(budget),
            // No device yet: no copies either; asked again.
            _ => return u64::MAX,
        },
    };
    *CAP.get_or_init(|| cap)
}

pub(crate) fn cap_for_budget(budget: u64) -> u64 {
    if budget >= 20 * GIB { u64::MAX } else { (budget / 512).clamp(8 * MIB, 64 * MIB) }
}

/// The first level to make a host copy from: `from` or, while its bytes
/// (`size_from(level)`) are over `cap`, the next, down to the last level.
pub(crate) fn cap_base(size_from: impl Fn(u32) -> u64, from: u32, levels: u32, cap: u64) -> u32 {
    let mut base = from;
    while base + 1 < levels && size_from(base) > cap {
        base += 1;
    }
    base
}

// ---- probes ----------------------------------------------------------------

/// Why a texture picked by index has no host copy now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Why {
    /// The card's budget (the admission line) refused it.
    Budget,
    /// Host memory (commit) or a failed upload allocation deferred it.
    Commit,
}

/// What a probe line says of a texture.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Shape {
    pub slot: u32,
    pub bytes: u64,
    pub width: u64,
    pub height: u32,
    pub mips: u32,
    pub format: u32,
    pub first: u32,
}

static REFUSED: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
static REFUSED_BYTES: AtomicU64 = AtomicU64::new(0);
static ADMITTED: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
static FIRST_LOGGED: AtomicU64 = AtomicU64::new(0);
static NULLS: std::sync::Mutex<std::collections::BTreeMap<u32, Shape>> = std::sync::Mutex::new(std::collections::BTreeMap::new());

fn first_limit() -> u64 {
    static N: OnceLock<u64> = OnceLock::new();
    *N.get_or_init(|| env_u64("VICTORY_RESIDENCY_FIRST").unwrap_or(16))
}

fn trace_limit() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| env_u64("VICTORY_RESIDENCY_TRACE").unwrap_or(0).min(1000) as usize)
}

/// A request for a new host copy over the line was let through (`Floor`, `Small`).
pub(crate) fn note_admitted(how: Admit) {
    match how {
        Admit::Floor => ADMITTED[0].fetch_add(1, Ordering::Relaxed),
        Admit::Small => ADMITTED[1].fetch_add(1, Ordering::Relaxed),
        _ => 0,
    };
}

/// Slot `shape.slot` was left without a host copy.
pub(crate) fn note_refused(shape: Shape, why: Why) {
    match why {
        Why::Budget => {
            REFUSED[0].fetch_add(1, Ordering::Relaxed);
            REFUSED_BYTES.fetch_add(shape.bytes, Ordering::Relaxed);
            note_starved();
            let new = NULLS.lock().unwrap_or_else(|e| e.into_inner()).insert(shape.slot, shape).is_none();
            if new && FIRST_LOGGED.fetch_add(1, Ordering::Relaxed) < first_limit() {
                tracing::warn!(
                    slot = shape.slot,
                    bytes = shape.bytes,
                    width = shape.width,
                    height = shape.height,
                    mips = shape.mips,
                    format = shape.format,
                    first_level = shape.first,
                    "residency: texture refused, over the card's budget line: it shows nothing until the card has room"
                );
            }
        }
        Why::Commit => {
            if REFUSED[1].fetch_add(1, Ordering::Relaxed) < first_limit() {
                tracing::warn!(slot = shape.slot, bytes = shape.bytes, format = shape.format, "residency: texture deferred, host memory or upload buffer refused");
            }
        }
    }
}

/// Slot `slot` has its host copy now.
pub(crate) fn note_resident(slot: u32) {
    let mut nulls = NULLS.lock().unwrap_or_else(|e| e.into_inner());
    if !nulls.is_empty() {
        nulls.remove(&slot);
    }
}

/// The compact line, at most every five seconds (called by the pass over
/// textures picked by index). `memory`: the card's budget and use.
pub(crate) fn report(memory: Option<[u64; 4]>) {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    static BEFORE: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
    static LOGGED_POLICY: AtomicBool = AtomicBool::new(false);
    let now = crate::host::coarse_ms();
    let next = NEXT.load(Ordering::Relaxed);
    if now < next || NEXT.compare_exchange(next, now + 5_000, Ordering::Relaxed, Ordering::Relaxed).is_err() {
        return;
    }
    use crate::draw::{HEAP_TEXTURES_AWAY, HEAP_TEXTURES_QUEUE};
    let (copies, filled) = (crate::resource::KINDS[0].load(Ordering::Relaxed), crate::resource::KINDS[1].load(Ordering::Relaxed));
    let reduced = [crate::resource::HELD[0].load(Ordering::Relaxed), crate::resource::HELD[1].load(Ordering::Relaxed)];
    let queue: Vec<u64> = HEAP_TEXTURES_QUEUE.iter().map(|q| q.load(Ordering::Relaxed)).collect();
    let (budget, used, shared) = memory.map_or((0, 0, 0), |m| (m[0] >> 20, m[1] >> 20, m[3] >> 20));
    let nulls = NULLS.lock().unwrap_or_else(|e| e.into_inner());
    let (null_n, null_bytes) = (nulls.len(), nulls.values().map(|s| s.bytes).sum::<u64>());
    let now_counts = [
        REFUSED[0].load(Ordering::Relaxed),
        REFUSED[1].load(Ordering::Relaxed),
        ADMITTED[0].load(Ordering::Relaxed),
        ADMITTED[1].load(Ordering::Relaxed),
        HEAP_TEXTURES_AWAY.load(Ordering::Relaxed),
    ];
    let d: Vec<u64> = now_counts.iter().zip(&BEFORE).map(|(n, b)| n - b.swap(*n, Ordering::Relaxed).min(*n)).collect();
    if let Some([budget_bytes, ..]) = memory
        && budget_bytes > 0
        && !LOGGED_POLICY.swap(true, Ordering::Relaxed)
    {
        let p = policy(budget_bytes);
        let (stage, span) = upload_budget(Some(budget_bytes), false);
        let cap = mip_cap_bytes();
        tracing::info!(
            "residency: mode={} budget {} MB -> admit line {} MB, hard line {} MB, evict {}..{} MB after {} idle frames, floor {} MB, small {} MB, per frame {} MB / {} ms, mip cap {}",
            if legacy() { "legacy" } else { "auto" },
            p.budget >> 20,
            p.admit_line >> 20,
            p.hard_line >> 20,
            p.evict_hi >> 20,
            p.evict_lo >> 20,
            evict_frames(),
            p.floor >> 20,
            p.small >> 20,
            stage >> 20,
            span,
            if cap == u64::MAX { "none".to_string() } else { format!("{} MB", cap >> 20) },
        );
    }
    tracing::info!(
        "residency: card {used}/{budget} MB (shared {shared} MB); resident full {copies} copies {} MB (of them reduced-mip made so far {} {} MB); \
         waiting {} (queue {} due {} held for memory {} partly {}); refused budget +{} (now null {null_n} {} MB, {} events so far), \
         refused other +{} (commit {}), passed over unmapped +{}, admitted over the line: floor +{} small +{}; evicting={}",
        filled >> 20,
        reduced[0],
        reduced[1] >> 20,
        queue[2],
        queue[0],
        queue[1],
        queue[2],
        queue[3],
        d[0],
        null_bytes >> 20,
        now_counts[0],
        d[1] + d[4],
        now_counts[1],
        d[4],
        d[2],
        d[3],
        EVICTING.load(Ordering::Relaxed),
    );
    let n = trace_limit();
    if n > 0 && !nulls.is_empty() {
        let mut all: Vec<&Shape> = nulls.values().collect();
        all.sort_by_key(|s| std::cmp::Reverse(s.bytes));
        for s in all.into_iter().take(n) {
            tracing::info!(
                slot = s.slot,
                bytes = s.bytes,
                width = s.width,
                height = s.height,
                mips = s.mips,
                format = s.format,
                first_level = s.first,
                "residency trace: non-resident texture picked by index (refused for the budget)"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BUDGET: u64 = 8643 * MIB;

    #[test]
    fn policy_lines_follow_the_reported_budget() {
        let p = Policy::derive(BUDGET, None, None, None);
        assert_eq!(p.admit_line, BUDGET - BUDGET / 16);
        assert!(p.hard_line > p.admit_line && p.hard_line < BUDGET);
        assert!(p.evict_lo < p.evict_hi && p.evict_hi < p.admit_line);
        assert_eq!(p.floor, GIB);
        assert_eq!(p.small, 8 * MIB);
        // A big card keeps a proportional reserve; a tiny one never goes negative.
        let big = Policy::derive(32 * GIB, None, None, None);
        assert_eq!(big.admit_line, 32 * GIB - 2 * GIB);
        let tiny = Policy::derive(GIB, None, None, None);
        assert!(tiny.admit_line >= GIB * 3 / 4);
        let zero = Policy::derive(0, None, None, None);
        assert_eq!(zero.admit(0, 0, 1), Admit::No);
    }

    #[test]
    fn small_textures_are_admitted_past_the_line_but_never_past_the_hard_line() {
        let p = Policy::derive(BUDGET, None, None, None);
        let filled = 3 * GIB;
        assert_eq!(p.admit(p.admit_line - 1, filled, 64 * MIB), Admit::Yes);
        assert_eq!(p.admit(p.admit_line + MIB, filled, 2 * MIB), Admit::Small);
        assert_eq!(p.admit(p.admit_line + MIB, filled, 64 * MIB), Admit::No);
        assert_eq!(p.admit(p.hard_line, filled, MIB), Admit::No);
        // Under the floor anything fits the hard line.
        assert_eq!(p.admit(p.admit_line + MIB, p.floor - 1, 64 * MIB), Admit::Floor);
        assert_eq!(p.admit(p.hard_line - MIB, p.floor - 1, 64 * MIB), Admit::No);
    }

    #[test]
    fn eviction_has_hysteresis_and_hurries_when_starved() {
        let p = Policy::derive(BUDGET, None, None, None);
        assert!(!p.evicting(false, p.evict_hi - 1, false));
        assert!(p.evicting(false, p.evict_hi, false));
        assert!(p.evicting(true, p.evict_lo + 1, false));
        assert!(!p.evicting(true, p.evict_lo, false));
        assert!(p.evicting(false, 0, true));
        assert_eq!(p.evict_plan(false, p.evict_hi, false, 30), (600_000, 4, u64::MAX));
        let (ms, frames, most) = p.evict_plan(true, p.evict_hi, false, 30);
        assert_eq!((ms, frames), (1_500, 30));
        assert_eq!(most, (p.evict_hi - p.evict_lo).max(64 * MIB));
        let (ms, frames, _) = p.evict_plan(true, p.admit_line, false, 30);
        assert_eq!((ms, frames), (500, 10));
        assert_eq!(p.evict_plan(true, 0, true, 30).2, 64 * MIB);
    }

    #[test]
    fn mip_cap_drops_levels_until_the_copy_fits() {
        // 4096^2 block-compressed at one byte a texel with a full chain.
        let size = |level: u32| ((4096u64 >> level) * (4096u64 >> level)) * 4 / 3;
        let cap = cap_for_budget(BUDGET);
        assert_eq!(cap, BUDGET / 512);
        assert_eq!(cap_base(size, 0, 13, cap), 1);
        assert_eq!(cap_base(size, 0, 13, u64::MAX), 0);
        assert_eq!(cap_base(size, 2, 13, cap), 2);
        // Never past the last level.
        assert_eq!(cap_base(size, 0, 3, 1), 2);
        assert_eq!(cap_for_budget(24 * GIB), u64::MAX);
        assert_eq!(cap_for_budget(2 * GIB), 8 * MIB);
    }

    #[test]
    fn upload_is_spread_over_frames() {
        let (bytes, ms) = upload_budget(Some(BUDGET), false);
        assert_eq!(bytes, BUDGET / 96);
        assert_eq!(ms, 60);
        assert_eq!(upload_budget(Some(BUDGET), true), (bytes * 2, 120));
        assert_eq!(upload_budget(Some(2 * GIB), false).0, 48 * MIB);
        assert_eq!(upload_budget(Some(32 * GIB), false).0, 192 * MIB);
    }

    #[test]
    fn bake_and_lookahead_leave_room_for_live_game_allocations() {
        for (commit, bake, lookahead, growth) in [
            (128 * MIB, false, false, false),
            (256 * MIB, true, false, false),
            (512 * MIB, true, true, false),
            (768 * MIB, true, true, false),
            (1024 * MIB, true, true, true),
        ] {
            assert_eq!(has_upload_reserve(commit, true), bake);
            assert_eq!(has_upload_reserve(commit, false), lookahead);
            assert_eq!(has_growth_reserve(commit, false), growth);
        }
    }

    #[test]
    fn either_constraint_enters_pressure_and_both_must_recover() {
        assert!(next_pressure(false, 12 * GIB, 4 * GIB));
        assert!(next_pressure(false, 3 * GIB, 16 * GIB));
        assert!(next_pressure(true, 4 * GIB, 7 * GIB));
        assert!(next_pressure(true, 6 * GIB, 5 * GIB));
        assert!(!next_pressure(true, 5 * GIB, 6 * GIB));
        assert!(!next_pressure(false, 4 * GIB, 5 * GIB));
    }

    #[test]
    fn prefetch_respects_existing_budgets_and_resumes_after_recovery() {
        assert_eq!(prefetch_limit(512 << 20, false, false, true, GIB), 64 << 20);
        assert_eq!(prefetch_limit(512 << 20, false, false, true, GIB - 1), 512 << 20);
        assert_eq!(prefetch_limit(512 << 20, false, false, false, 2 * GIB), 512 << 20);
        assert_eq!(prefetch_limit(0, true, true, false, 0), 0);
        assert_eq!(prefetch_limit(8 << 20, true, true, false, 0), 8 << 20);
        assert_eq!(prefetch_limit(512 << 20, true, false, false, 0), 64 << 20);
        assert_eq!(prefetch_limit(512 << 20, true, true, false, 0), 16 << 20);
        assert_eq!(prefetch_limit(192 << 20, false, false, false, 0), 192 << 20);
    }
}
