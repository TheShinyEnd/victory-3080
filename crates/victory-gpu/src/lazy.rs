//! Lazy copy-back (`VICTORY_BACK_ON_READ=1`, off by default): what shaders
//! wrote to a block of title memory that the title does not read is fetched
//! from the card but NOT merged into title memory until something touches it.
//!
//! WHY. The landing merged 1,500 pieces / 49 MB a frame into title memory
//! (23.6 ms of a 33 ms copy-back on 4 threads) though the title's CPU read
//! none of the sampled pages (access probe, fps3). The fetch itself (card to
//! a read-back buffer) is cheap (3.6 ms, the card is nearly idle); the merge
//! is what costs.
//!
//! WHAT (fps5 design). A piece in a 64 KiB block of title memory that only
//! this card copy holds is DEFERRED: it is fetched into a slot of a
//! persistent read-back "stash" (`crate::host::lazyfetch`) instead of the
//! landing's buffer, the block (every title view of its physical page) is
//! made `PAGE_NOACCESS`, and nothing is merged. Title memory and the copy's
//! CPU side keep their old bytes, equal to each other, so a refresh finds
//! nothing to send and the card keeps its newer bytes. A later landing that
//! flags the block again fetches into a second slot, swapped in when the
//! fetch is done (a fault meanwhile uses the older slot: the title has not
//! been told of the newer flight yet, as with a landing still under way).
//!
//! RESOLVING is lock-free and on the faulting thread: no D3D12 call, no
//! heap allocation, no lock of any other subsystem (the stall of fps3-h4 was
//! a resolve that needed the journal's/guards' locks a faulting thread of
//! ours held). Per group an atomic state, DEFERRED -> RESOLVING (CAS-owned)
//! -> PRESENT: the owner merges the slot's bytes into title memory through
//! our own view of the pages (view memory) or through the title's address
//! opened for that moment (private memory, `VICTORY_BACK_ON_READ_PRIVATE=1`),
//! gives the guards' protection back and marks it PRESENT; others that
//! fault meanwhile spin/yield until then. The notes a landing makes (journal,
//! "landed" pages, placed textures) are made later by the relay thread
//! ([`split`]'s reclaim), which takes those locks anyway.
//!
//! WRITES of the title into a deferred block fault too and resolve first,
//! then the write is made again (and meets the guards as before): resolve-
//! then-write. Our own threads' reads (refresh compares) fault and resolve
//! the same way; they then read the landed bytes, which the copy's CPU side
//! also holds, so nothing is taken for a title write.
//!
//! Not deferred: copies in the PC's memory, in-place copies, windows, blocks
//! another copy (or a placed texture) covers, blocks with bytes waiting to go
//! up, ranges over two blocks, blocks faulted lately (`hot`, 120 frames,
//! doubling).

use std::collections::HashMap;
use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use crate::shadow::Shadow;

const BLOCK: usize = 64 << 10;
const PAGE: usize = 4096;
/// Groups at most (each is one 64 KiB block and its aliases).
const GROUPS: usize = 16384;
const ALIASES: usize = 8;

/// `VICTORY_BACK_ON_READ=1` (and the landing on its threads, and the guards).
///
/// With `VICTORY_AB=<s>,VICTORY_BACK_ON_READ` (diagnostics) it alternates in
/// phases of the run; the blocks deferred when it goes off are resolved by
/// [`drain`].
pub(crate) fn on() -> bool {
    static ASKED: OnceLock<Result<bool, u32>> = OnceLock::new();
    let asked = ASKED.get_or_init(|| {
        let base = crate::land::on();
        #[cfg(feature = "diagnostics")]
        if let Some(bit) = crate::ab::bit("VICTORY_BACK_ON_READ") {
            return if base { Err(bit) } else { Ok(false) };
        }
        Ok(base && std::env::var_os("VICTORY_BACK_ON_READ").is_some_and(|v| v == "1"))
    });
    match *asked {
        Ok(on) => on,
        #[cfg(feature = "diagnostics")]
        Err(bit) => crate::ab::on(bit),
        #[cfg(not(feature = "diagnostics"))]
        Err(_) => false,
    }
}

static WORLD_FRAMES: AtomicU64 = AtomicU64::new(0);

/// A frame was finished: whether it was one of the game proper (many draws).
pub(crate) fn world(heavy: bool) {
    if heavy {
        WORLD_FRAMES.fetch_add(1, Ordering::Relaxed);
    }
}

/// Deferring starts only after this many frames of the game proper
/// (`VICTORY_BACK_ON_READ_WORLD`, 120).
pub(crate) fn active() -> bool {
    static AFTER: OnceLock<u64> = OnceLock::new();
    let after = *AFTER.get_or_init(|| std::env::var("VICTORY_BACK_ON_READ_WORLD").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(120));
    on() && WORLD_FRAMES.load(Ordering::Relaxed) >= after
}

/// Groups deferred right now (not PRESENT, not FREE).
static ANY: AtomicUsize = AtomicUsize::new(0);

/// Whether anything is deferred (one load).
pub(crate) fn any() -> bool {
    ANY.load(Ordering::Acquire) != 0
}

/// Why a deferral ended.
#[derive(Clone, Copy)]
pub(crate) enum Why {
    ReadFault = 0,
    WriteFault = 1,
    Guards = 2,
    Mapping = 3,
    Landing = 4,
    Kernel = 5,
    Resend = 6,
    Audit = 7,
    Another = 8,
}

const WHYS: usize = 9;
const WHY_NAMES: [&str; WHYS] = ["title read", "title write", "guards", "mapping", "landing", "kernel write", "resend", "audit", "other"];

// ---------------------------------------------------------------- states

const FREE: u32 = 0;
const DEFERRED: u32 = 1;
const RESOLVING: u32 = 2;
/// Owned for a moment by the relay (slot swap) or the guards (wanted).
const SWAPPING: u32 = 3;
const PRESENT: u32 = 4;

/// A group: one 64 KiB block of title memory (and the other title addresses
/// of its physical page) whose newest bytes of one copy are in a stash slot.
/// All atomics: read by faulting threads without any lock. Fields other than
/// `state`/`wanted` change only while the relay owns the group (FREE, or
/// SWAPPING by the relay).
struct Group {
    state: AtomicU32,
    shadow: AtomicPtr<Shadow>,
    /// Range of the slot's bytes, offsets in the copy.
    lo: AtomicUsize,
    hi: AtomicUsize,
    /// Address of the slot's byte for offset `lo` (0: none yet).
    data: AtomicUsize,
    /// The copy's CPU side, its title address (offset 0).
    held: AtomicUsize,
    title: AtomicUsize,
    /// Our own view of the primary block (0: private memory).
    beside: AtomicUsize,
    nblocks: AtomicUsize,
    blocks: [AtomicUsize; ALIASES],
    wanted: [AtomicU32; ALIASES],
    /// Set by the resolve: 4 KiB pages (bits, of the primary block) that got
    /// new bytes; why; the frame.
    changed: AtomicU32,
    why: AtomicU32,
    present_at: AtomicU64,
}

impl Group {
    const fn new() -> Group {
        Group {
            state: AtomicU32::new(FREE),
            shadow: AtomicPtr::new(std::ptr::null_mut()),
            lo: AtomicUsize::new(0),
            hi: AtomicUsize::new(0),
            data: AtomicUsize::new(0),
            held: AtomicUsize::new(0),
            title: AtomicUsize::new(0),
            beside: AtomicUsize::new(0),
            nblocks: AtomicUsize::new(0),
            blocks: [const { AtomicUsize::new(0) }; ALIASES],
            wanted: [const { AtomicU32::new(0) }; ALIASES],
            changed: AtomicU32::new(0),
            why: AtomicU32::new(0),
            present_at: AtomicU64::new(0),
        }
    }
}

static G: [Group; GROUPS] = [const { Group::new() }; GROUPS];

/// Block number -> group + 1 (0: none), in pieces of 65,536 blocks (4 GiB of
/// address space) made when first needed (by the relay) and never freed.
struct Table([AtomicPtr<[AtomicU32; Table::PER]>; Table::PIECES]);

impl Table {
    const PER: usize = 1 << 16;
    /// 2^31 blocks of 64 KiB: 128 TiB.
    const PIECES: usize = 1 << 15;

    fn get(&self, block: usize) -> Option<usize> {
        let piece = self.0.get(block / Table::PER)?.load(Ordering::Acquire);
        // SAFETY: null or a piece, never freed.
        let piece = unsafe { piece.as_ref() }?;
        let g = piece[block % Table::PER].load(Ordering::Acquire);
        (g != 0).then(|| g as usize - 1)
    }

    /// On the relay thread only (allocates).
    fn set(&self, block: usize, g: Option<usize>) {
        let Some(place) = self.0.get(block / Table::PER) else {
            return;
        };
        let mut piece = place.load(Ordering::Acquire);
        if piece.is_null() {
            if g.is_none() {
                return;
            }
            let made: Box<[AtomicU32; Table::PER]> = Box::new([const { AtomicU32::new(0) }; Table::PER]);
            piece = Box::into_raw(made);
            place.store(piece, Ordering::Release);
        }
        // SAFETY: a piece, never freed.
        let piece: &[AtomicU32; Table::PER] = unsafe { &*piece };
        piece[block % Table::PER].store(g.map_or(0, |g| g as u32 + 1), Ordering::Release);
    }
}

static TABLE: Table = Table([const { AtomicPtr::new(std::ptr::null_mut()) }; Table::PIECES]);

// ---------------------------------------------------------------- counters

mod c {
    pub const FLAGGED: usize = 0;
    pub const FETCHED_NOW: usize = 1;
    pub const STASHED: usize = 2;
    pub const NEW_GROUPS: usize = 3;
    pub const REFLAGS: usize = 4;
    pub const RESOLVED: usize = 5; // + why
    pub const RESOLVED_BYTES: usize = 5 + super::WHYS;
    pub const RESOLVE_NS: usize = 5 + 2 * super::WHYS;
    pub const RESOLVE_MAX_NS: usize = RESOLVE_NS + 1;
    pub const WAITED: usize = RESOLVE_NS + 2;
    pub const WAIT_NS: usize = RESOLVE_NS + 3;
    pub const NO_CARD: usize = RESOLVE_NS + 4;
    pub const NO_SPAN: usize = NO_CARD + 1;
    pub const NO_PENDING: usize = NO_CARD + 2;
    pub const NO_VIEW: usize = NO_CARD + 3;
    pub const NO_CLEAN: usize = NO_CARD + 4;
    pub const NO_PROTECT: usize = NO_CARD + 5;
    pub const NO_SHARED: usize = NO_CARD + 6;
    pub const NO_REMAP: usize = NO_CARD + 7;
    pub const NO_ROOM: usize = NO_CARD + 8;
    pub const HOT: usize = NO_CARD + 9;
    pub const OTHER_COPY: usize = NO_CARD + 10;
    pub const PROTECT_CALLS: usize = NO_CARD + 11;
    pub const PROTECT_NS: usize = NO_CARD + 12;
    pub const LATE: usize = NO_CARD + 13;
    pub const BREACH: usize = NO_CARD + 14;
    pub const DIRTY_SKIP: usize = NO_CARD + 15;
    pub const SEND_FAIL: usize = NO_CARD + 16;
    pub const SPLIT_NS: usize = NO_CARD + 17;
    pub const FETCHED_NS: usize = NO_CARD + 18;
    pub const RECLAIM_NS: usize = NO_CARD + 19;
    pub const MERGED_BYTES: usize = NO_CARD + 20;
    pub const NO_TEXTURE: usize = NO_CARD + 21;
    pub const CACHED: usize = NO_CARD + 22;
    pub const COUNT: usize = NO_CARD + 23;
}

static STATS: [AtomicU64; c::COUNT] = [const { AtomicU64::new(0) }; c::COUNT];

fn count(which: usize, n: u64) {
    STATS[which].fetch_add(n, Ordering::Relaxed);
}

/// Of a landing: how many bytes were flagged, how many are merged now.
pub(crate) fn account(flagged: u64, fetched: u64) {
    count(c::FLAGGED, flagged);
    count(c::FETCHED_NOW, fetched);
}

static SKIP: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// A resolve went on without a copy's lock (see `Shadow::land_card`).
pub(crate) fn lock_given_up() {
    SKIP[3].fetch_add(1, Ordering::Relaxed);
}

/// A landed piece of `bytes`, of which `kept_out` were not put into title memory
/// because the title's own bytes there are still to go up or just went.
pub(crate) fn skipped(bytes: usize, kept_out: usize) {
    SKIP[0].fetch_add(bytes as u64, Ordering::Relaxed);
    if kept_out != 0 {
        SKIP[1].fetch_add(kept_out as u64, Ordering::Relaxed);
        SKIP[2].fetch_add(1, Ordering::Relaxed);
    }
}

/// Kept for the old callers' sake (the stage of a resolve; nothing now).
pub(crate) fn stage(_what: &'static str) {}

/// The frame line.
pub(crate) fn log(frames: f64) {
    {
        let k: Vec<f64> = SKIP.iter().map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames).collect();
        if k[0] > 0.0 {
            tracing::info!(
                "title-wins census per frame: {:.1} MB of landed pieces, of which {:.3} MB in {:.1} pieces not put into title memory because the title's bytes there were still to go up or had just gone",
                k[0] / 1e6,
                k[1] / 1e6,
                k[2],
            );
        }
    }
    let max_ns = STATS[c::RESOLVE_MAX_NS].swap(0, Ordering::Relaxed);
    let s: Vec<f64> = STATS.iter().map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    if !on() && s.iter().all(|v| *v == 0.0) {
        return;
    }
    let mb = |v: f64| v / (1 << 20) as f64;
    let live = G.iter().filter(|g| matches!(g.state.load(Ordering::Relaxed), DEFERRED | RESOLVING | SWAPPING)).count();
    let resolves: f64 = (0..WHYS).map(|i| s[c::RESOLVED + i]).sum();
    let why: Vec<String> = (0..WHYS)
        .filter(|&i| s[c::RESOLVED + i] > 0.0)
        .map(|i| format!("{} {:.2} ({:.2} MB)", WHY_NAMES[i], s[c::RESOLVED + i], mb(s[c::RESOLVED_BYTES + i])))
        .collect();
    tracing::info!(
        "lazy copy-back per frame (VICTORY_BACK_ON_READ={}): flagged {:.1} MB, merged now {:.1} MB, fetched to the stash unmerged {:.1} MB; \
         groups new {:.1}, flagged again {:.1}, deferred now {}; resolves {:.2} [{}], merged by them {:.2} MB, {:.3} ms each on average, \
         longest {:.3} ms since the last line, {:.3} ms summed; threads that waited for another's resolve {:.2} ({:.3} ms); \
         not deferred: not a card copy {:.1}, spans blocks {:.1}, bytes waiting to go up {:.1}, no view of ours {:.1}, \
         not clean {:.1}, protection {:.1}, another copy {:.1}, a placed texture {:.1}, remapping {:.1}, no stash room {:.1}, hot {:.1}, \
         another copy's group {:.1}, refused lately (not asked again) {:.1}; protection calls {:.1} in {:.2} ms; flagged after a resolve (landed late) {:.2}; \
         audit breaches {:.2}; pages kept out (bytes waiting) {:.2}; stash fetch failures {:.2}; relay: split {:.2} ms, \
         fetched {:.2} ms, reclaim {:.2} ms",
        u8::from(on()),
        mb(s[c::FLAGGED]),
        mb(s[c::FETCHED_NOW]),
        mb(s[c::STASHED]),
        s[c::NEW_GROUPS],
        s[c::REFLAGS],
        live,
        resolves,
        why.join("; "),
        mb(s[c::MERGED_BYTES]),
        if resolves > 0.0 { s[c::RESOLVE_NS] / resolves / 1e6 } else { 0.0 },
        max_ns as f64 / 1e6,
        s[c::RESOLVE_NS] / 1e6,
        s[c::WAITED],
        s[c::WAIT_NS] / 1e6,
        s[c::NO_CARD],
        s[c::NO_SPAN],
        s[c::NO_PENDING],
        s[c::NO_VIEW],
        s[c::NO_CLEAN],
        s[c::NO_PROTECT],
        s[c::NO_SHARED],
        s[c::NO_TEXTURE],
        s[c::NO_REMAP],
        s[c::NO_ROOM],
        s[c::HOT],
        s[c::OTHER_COPY],
        s[c::CACHED],
        s[c::PROTECT_CALLS],
        s[c::PROTECT_NS] / 1e6,
        s[c::LATE],
        s[c::BREACH],
        s[c::DIRTY_SKIP],
        s[c::SEND_FAIL],
        s[c::SPLIT_NS] / 1e6,
        s[c::FETCHED_NS] / 1e6,
        s[c::RECLAIM_NS] / 1e6,
    );
}

// ---------------------------------------------------------------- helpers

fn query_any(at: usize) -> Option<(u32, usize, bool, bool)> {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery};
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    if n == 0 {
        return None;
    }
    let end = (info.BaseAddress as usize).saturating_add(info.RegionSize);
    Some((info.Protect, end, info.Type == MEM_MAPPED, info.State == MEM_COMMIT))
}

/// `VICTORY_BACK_ON_READ_LOOSE=1`: a block need not be known clean by the
/// write watch's bits to be deferred.
fn loose() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_BACK_ON_READ_LOOSE").is_some_and(|v| v == "1"))
}

/// `VICTORY_BACK_ON_READ_PRIVATE=1`: memory with no view is deferred too.
fn private_ok() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_BACK_ON_READ_PRIVATE").is_some_and(|v| v == "1"))
}

/// Frames a block that was faulted in is merged eagerly (doubling with each
/// fault, at most 8x): `VICTORY_BACK_ON_READ_HOT`, 120.
fn hot_frames() -> u64 {
    static N: OnceLock<u64> = OnceLock::new();
    *N.get_or_init(|| std::env::var("VICTORY_BACK_ON_READ_HOT").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(120))
}

fn protect(block: usize, prot: u32) -> Option<u32> {
    use windows_sys::Win32::System::Memory::VirtualProtect;
    let started = std::time::Instant::now();
    let mut old = 0;
    // SAFETY: title memory; only its protection changes.
    let done = unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, prot, &mut old) } != 0;
    count(c::PROTECT_CALLS, 1);
    count(c::PROTECT_NS, started.elapsed().as_nanos() as u64);
    done.then_some(old)
}

/// Block `b` made `PAGE_NOACCESS` page by page: the protection its pages
/// had, if all had the same one and it was read-only or read-write; else
/// every page has its own back and `None`. (Sixteen calls of a few
/// microseconds; no `VirtualQuery`, which walks a big region to its end.)
fn close_block(b: usize) -> Option<u32> {
    use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE, VirtualProtect};
    let started = std::time::Instant::now();
    let mut olds = [0u32; BLOCK / PAGE];
    let mut done = 0;
    let mut fit = true;
    for k in 0..olds.len() {
        let mut old = 0;
        // SAFETY: title memory; only its protection changes.
        if unsafe { VirtualProtect((b * BLOCK + k * PAGE) as *const _, PAGE, PAGE_NOACCESS, &mut old) } == 0 {
            fit = false;
            break;
        }
        olds[k] = old;
        done += 1;
        if old != olds[0] || (old != PAGE_READONLY && old != PAGE_READWRITE) {
            fit = false;
            break;
        }
    }
    if !fit {
        for (k, &old) in olds.iter().enumerate().take(done) {
            let mut was = 0;
            // SAFETY: as above, the protection the page had.
            unsafe { VirtualProtect((b * BLOCK + k * PAGE) as *const _, PAGE, old, &mut was) };
        }
    }
    count(c::PROTECT_CALLS, done as u64 * if fit { 1 } else { 2 });
    count(c::PROTECT_NS, started.elapsed().as_nanos() as u64);
    fit.then_some(olds[0])
}

thread_local! {
    /// This thread is merging a deferral (a fault of its own is not one).
    static RESOLVING_HERE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether this thread is in a resolve.
pub(crate) fn resolving() -> bool {
    RESOLVING_HERE.with(|r| r.get())
}

/// Waits (spinning, then yielding, then sleeping briefly) while group `g`
/// is in state `busy`. No lock, no allocation.
fn wait_while(g: usize, busy: impl Fn(u32) -> bool) -> u32 {
    let began = std::time::Instant::now();
    let mut tries = 0u32;
    loop {
        let s = G[g].state.load(Ordering::Acquire);
        if !busy(s) {
            if tries > 0 {
                count(c::WAITED, 1);
                count(c::WAIT_NS, began.elapsed().as_nanos() as u64);
            }
            return s;
        }
        tries += 1;
        if tries < 64 {
            std::hint::spin_loop();
        } else if tries < 4096 {
            std::thread::yield_now();
        } else {
            std::thread::sleep(std::time::Duration::from_micros(50));
        }
    }
}

/// Group `g` resolved (by this thread, or by another while this waits).
/// No lock, no allocation, no call into D3D12.
fn resolve(g: usize, why: Why) {
    loop {
        let s = G[g].state.load(Ordering::Acquire);
        match s {
            DEFERRED => {
                if G[g].state.compare_exchange(DEFERRED, RESOLVING, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                    merge_group(g, why);
                    return;
                }
            }
            RESOLVING | SWAPPING => {
                if resolving() {
                    // (Our own merge faulted in another group: not waited for.)
                    return;
                }
                wait_while(g, |s| s == RESOLVING || s == SWAPPING);
            }
            _ => return,
        }
    }
}

/// The owner of group `g` (RESOLVING) merges its slot into title memory,
/// gives the protection back and marks it PRESENT.
fn merge_group(g: usize, why: Why) {
    use windows_sys::Win32::System::Memory::PAGE_READWRITE;
    let began = std::time::Instant::now();
    RESOLVING_HERE.with(|r| r.set(true));
    let grp = &G[g];
    let (lo, hi, data) = (grp.lo.load(Ordering::Acquire), grp.hi.load(Ordering::Acquire), grp.data.load(Ordering::Acquire));
    let (held, title, beside) = (grp.held.load(Ordering::Acquire), grp.title.load(Ordering::Acquire), grp.beside.load(Ordering::Acquire));
    let nblocks = grp.nblocks.load(Ordering::Acquire).min(ALIASES);
    let shadow = grp.shadow.load(Ordering::Acquire);
    let mut changed = 0u32;
    let mut merged = 0u64;
    if data != 0 && hi > lo && !shadow.is_null() {
        let t0 = title + lo;
        let into = if beside != 0 {
            beside + t0 % BLOCK
        } else {
            // Private memory: the title's own address, opened for the merge
            // (a window of microseconds in which another title thread may
            // read part-merged bytes without faulting: an approximation).
            protect(t0 / BLOCK, PAGE_READWRITE);
            t0
        };
        // SAFETY: the group holds its copy (an `Arc` given up only at reclaim).
        let copy = unsafe { &*shadow };
        let mut at = 0;
        let n_all = hi - lo;
        while at < n_all {
            let n = (PAGE - (t0 + at) % PAGE).min(n_all - at);
            // Bytes of the title's waiting to go up are not overwritten (not
            // expected: the block was closed with none, and the title cannot
            // write it without resolving it first). Asked without waiting.
            if copy.lazy_dirty_overlaps(lo + at, lo + at + n) {
                count(c::DIRTY_SKIP, 1);
            } else {
                // SAFETY: `into` shows the block's title memory (our view of the
                // same pages, or the title's address opened just now); `data`
                // is the slot's bytes, which the relay does not touch while the
                // group is RESOLVING; `held` the copy's CPU side, which nothing
                // else writes in a deferred range.
                let done = unsafe {
                    crate::safecopy::merge((into + at) as *mut u8, (data + at) as *const u8, (held + lo + at) as *mut u8, n, true)
                };
                if done.new > 0 {
                    changed |= 1 << (((t0 + at) % BLOCK) / PAGE);
                    merged += done.dwords * 4;
                }
            }
            at += n;
        }
    }
    for i in 0..nblocks {
        let b = grp.blocks[i].load(Ordering::Relaxed);
        protect(b, grp.wanted[i].load(Ordering::Acquire));
    }
    grp.changed.store(changed, Ordering::Relaxed);
    grp.why.store(why as u32, Ordering::Relaxed);
    grp.present_at.store(crate::frame::current(), Ordering::Relaxed);
    grp.state.store(PRESENT, Ordering::Release);
    ANY.fetch_sub(1, Ordering::AcqRel);
    RESOLVING_HERE.with(|r| r.set(false));
    let ns = began.elapsed().as_nanos() as u64;
    count(c::RESOLVED + why as usize, 1);
    count(c::RESOLVED_BYTES + why as usize, (hi.saturating_sub(lo)) as u64);
    count(c::MERGED_BYTES, merged);
    count(c::RESOLVE_NS, ns);
    STATS[c::RESOLVE_MAX_NS].fetch_max(ns, Ordering::Relaxed);
}

/// The block at `at` faulted (`write`: a write). Whether the access is to be
/// made again: the block was deferred and is resolved now (or another did
/// it just before). Lock-free and allocation-free: called first in the
/// access-violation handler, on any thread.
pub(crate) fn fault(at: usize, write: bool) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, PAGE_READWRITE};
    if resolving() {
        return false;
    }
    let Some(g) = TABLE.get(at / BLOCK) else {
        return false;
    };
    let s = G[g].state.load(Ordering::Acquire);
    if matches!(s, DEFERRED | RESOLVING | SWAPPING) {
        resolve(g, if write { Why::WriteFault } else { Why::ReadFault });
    }
    // Made again only if it can be now (a write to a read-only guard goes on
    // to the guards' handler; a block given back is not ours). The
    // protection given back is known while the group is PRESENT (no
    // `VirtualQuery`, a millisecond in big private regions).
    let grp = &G[g];
    if grp.state.load(Ordering::Acquire) == PRESENT {
        let n = grp.nblocks.load(Ordering::Relaxed).min(ALIASES);
        if let Some(i) = (0..n).find(|&i| grp.blocks[i].load(Ordering::Relaxed) == at / BLOCK) {
            let p = grp.wanted[i].load(Ordering::Acquire);
            if grp.state.load(Ordering::Acquire) == PRESENT {
                return p == PAGE_READWRITE || (!write && p == PAGE_READONLY);
            }
        }
    }
    match query_any(at) {
        Some((p, _, _, true)) => p == PAGE_READWRITE || (!write && p == PAGE_READONLY),
        _ => false,
    }
}

/// Everything deferred in title memory `[at, at + len)` is resolved.
pub(crate) fn resolve_range(at: usize, len: usize, why: Why) {
    // (The access probe lets go of pages the kernel or a remap is to touch.)
    crate::accessprobe::release(at, len);
    if !any() || len == 0 {
        return;
    }
    let (first, last) = (at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
    if last - first > GROUPS {
        for g in 0..GROUPS {
            if !matches!(G[g].state.load(Ordering::Acquire), DEFERRED | RESOLVING | SWAPPING) {
                continue;
            }
            let n = G[g].nblocks.load(Ordering::Relaxed).min(ALIASES);
            if (0..n).any(|i| (first..last).contains(&G[g].blocks[i].load(Ordering::Relaxed))) {
                resolve(g, why);
            }
        }
        return;
    }
    for b in first..last {
        if let Some(g) = TABLE.get(b) {
            resolve(g, why);
        }
    }
}

/// The copy's range `[from, to)`: resolved if deferred.
pub(crate) fn resolve_shadow(shadow: &Shadow, from: usize, to: usize, why: Why) {
    if any() {
        resolve_range(shadow.span().0 as usize + from, to.saturating_sub(from), why);
    }
}

/// For `crate::watch`'s change of protection of `blocks` blocks from block
/// `at` to `prot`: `real(from, n)` makes it for a run of blocks. A deferred
/// block stays closed, with the protection noted for when it opens; one
/// being resolved is waited for (its resolver takes no lock) and then
/// changed as asked.
pub(crate) fn guard_call(at: usize, blocks: usize, prot: u32, real: &dyn Fn(usize, usize) -> (bool, u32)) -> (bool, u32) {
    use windows_sys::Win32::System::Memory::PAGE_READONLY;
    if !any() || resolving() {
        return real(at, blocks);
    }
    // Blocks kept closed: noted for the resolve instead.
    let kept = |b: usize| -> bool {
        let Some(g) = TABLE.get(b) else {
            return false;
        };
        loop {
            let s = wait_while(g, |s| s == RESOLVING || s == SWAPPING);
            if s != DEFERRED {
                return false;
            }
            if G[g].state.compare_exchange(DEFERRED, SWAPPING, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                let n = G[g].nblocks.load(Ordering::Relaxed).min(ALIASES);
                if let Some(i) = (0..n).find(|&i| G[g].blocks[i].load(Ordering::Relaxed) == b) {
                    G[g].wanted[i].store(prot, Ordering::Release);
                }
                G[g].state.store(DEFERRED, Ordering::Release);
                return true;
            }
        }
    };
    let (mut done, mut old) = (true, PAGE_READONLY);
    let mut first = true;
    let mut from = at;
    while from < at + blocks {
        if kept(from) {
            from += 1;
            continue;
        }
        let mut to = from + 1;
        while to < at + blocks && TABLE.get(to).is_none() {
            to += 1;
        }
        let (d, o) = real(from, to - from);
        done &= d;
        if first {
            old = o;
            first = false;
        }
        from = to;
    }
    (done, old)
}

// ---------------------------------------------------------------- the relay's side

/// What only the relay thread (and [`drain`]) keeps, under its own lock,
/// which no fault path takes.
#[derive(Default)]
struct Relay {
    free: Vec<usize>,
    made: bool,
    /// Per group: its copy (owned), its slot.
    owned: HashMap<usize, (Arc<Shadow>, Option<u32>)>,
    heat: HashMap<usize, (u8, u64)>,
    /// Blocks refused for what does not change from frame to frame (no card
    /// copy, no view, another copy over it): not asked again until then.
    nope: HashMap<usize, u64>,
    /// Fetches of this landing into the stash: group, range, slot, data.
    pending: Vec<Pending>,
    audit_at: usize,
}

struct Pending {
    g: usize,
    lo: usize,
    hi: usize,
    slot: u32,
    data: usize,
    behind: u64,
}

static RELAY: Mutex<Option<Relay>> = Mutex::new(None);

fn relay() -> MutexGuard<'static, Option<Relay>> {
    RELAY.lock().unwrap_or_else(|e| e.into_inner())
}

impl Relay {
    fn group(&mut self) -> Option<usize> {
        if !self.made {
            self.made = true;
            self.free = (0..GROUPS).rev().collect();
        }
        self.free.pop()
    }

    /// Groups resolved two frames ago or more: their notes made (the journal,
    /// the landed pages, the copy's landing bits), hot, given back.
    fn reclaim(&mut self, all: bool) {
        let began = std::time::Instant::now();
        let frame = crate::frame::current();
        let ids: Vec<usize> = self.owned.keys().copied().collect();
        let mut notes = crate::land::Notes::default();
        // (The notes hold the copies' addresses: they live until the flush,
        // one for all: a flush per group was 2 s for 900 groups in fps5-e.)
        let mut keep: Vec<Arc<Shadow>> = Vec::new();
        for g in ids {
            let grp = &G[g];
            if grp.state.load(Ordering::Acquire) != PRESENT || (!all && grp.present_at.load(Ordering::Relaxed) + 2 > frame) {
                continue;
            }
            let Some((shadow, slot)) = self.owned.remove(&g) else {
                continue;
            };
            let n = grp.nblocks.load(Ordering::Relaxed).min(ALIASES);
            for i in 0..n {
                TABLE.set(grp.blocks[i].load(Ordering::Relaxed), None);
            }
            let changed = grp.changed.load(Ordering::Relaxed);
            if changed != 0 {
                let (lo, hi) = (grp.lo.load(Ordering::Relaxed), grp.hi.load(Ordering::Relaxed));
                shadow.lazy_landed(lo, hi);
                notes.resolved((grp.title.load(Ordering::Relaxed) + lo) / BLOCK, changed, Arc::as_ptr(&shadow));
            }
            let why = grp.why.load(Ordering::Relaxed);
            if why == Why::ReadFault as u32 || why == Why::WriteFault as u32 {
                for i in 0..n {
                    let h = self.heat.entry(grp.blocks[i].load(Ordering::Relaxed)).or_default();
                    h.0 = (h.0 + 1).min(4);
                    h.1 = frame + (hot_frames() << (h.0 - 1));
                }
            }
            if let Some(slot) = slot {
                crate::host::lazy_stash_free(slot);
            }
            let raw = grp.shadow.swap(std::ptr::null_mut(), Ordering::AcqRel);
            if !raw.is_null() {
                // SAFETY: the reference given to the group when it was made.
                drop(unsafe { Arc::from_raw(raw) });
            }
            grp.data.store(0, Ordering::Relaxed);
            grp.state.store(FREE, Ordering::Release);
            self.free.push(g);
            keep.push(shadow);
        }
        notes.flush();
        drop(keep);
        if self.heat.len() > 100_000 {
            self.heat.retain(|_, h| h.1 > frame);
        }
        if self.nope.len() > 100_000 {
            self.nope.retain(|_, until| *until > frame);
        }
        count(c::RECLAIM_NS, began.elapsed().as_nanos() as u64);
    }
}

/// The lazy mode is off now and blocks are still deferred: all resolved.
pub(crate) fn drain() {
    let mut r = relay();
    let Some(r) = r.as_mut() else {
        return;
    };
    if r.owned.is_empty() {
        return;
    }
    // At most 256 a landing: the rest stay deferred (and resolvable by a
    // fault) until the next ones.
    let ids: Vec<usize> = r.owned.keys().copied().filter(|&g| G[g].state.load(Ordering::Acquire) != PRESENT).take(256).collect();
    for g in ids {
        resolve(g, Why::Another);
    }
    r.reclaim(true);
}

/// Protections someone else changed are found and those blocks resolved.
pub(crate) fn audit() {
    use windows_sys::Win32::System::Memory::PAGE_NOACCESS;
    if !any() {
        return;
    }
    let picked: Vec<usize> = {
        let mut r = relay();
        let Some(r) = r.as_mut() else {
            return;
        };
        let all: Vec<usize> = r.owned.keys().copied().collect();
        if all.is_empty() {
            return;
        }
        let from = r.audit_at % all.len();
        r.audit_at = r.audit_at.wrapping_add(8);
        all.iter().cycle().skip(from).take(8.min(all.len())).copied().collect()
    };
    for g in picked {
        if G[g].state.load(Ordering::Acquire) != DEFERRED {
            continue;
        }
        let n = G[g].nblocks.load(Ordering::Relaxed).min(ALIASES);
        let open = (0..n).any(|i| query_any(G[g].blocks[i].load(Ordering::Relaxed) * BLOCK).is_none_or(|q| q.0 != PAGE_NOACCESS));
        if open && G[g].state.load(Ordering::Acquire) == DEFERRED {
            count(c::BREACH, 1);
            resolve(g, Why::Audit);
        }
    }
}

enum Verdict {
    Deferred,
    Eager,
}

/// The pieces (copy, offsets, offset in the fetch buffer) of a landing, less
/// those that are deferred: those are fetched into the stash (sent here, on
/// the resolvers' own list and queue) and published by [`fetched`]; the
/// rest are fetched and merged now. `behind` is the send number a fetch of
/// this landing runs behind.
pub(crate) fn split(pieces: Vec<(Arc<Shadow>, usize, usize, u64)>, behind: u64) -> Vec<(Arc<Shadow>, usize, usize, u64)> {
    use windows_sys::Win32::System::Memory::PAGE_READWRITE;
    let began = std::time::Instant::now();
    let frame = crate::frame::current();
    let mut guard = relay();
    let r = guard.get_or_insert_with(Relay::default);
    r.reclaim(false);
    let mut out = Vec::with_capacity(pieces.len());
    let mut sends: Vec<(Arc<Shadow>, usize, usize, u32, usize)> = Vec::new();
    let start_pending = r.pending.len();
    for piece in pieces {
        let (shadow, lo, hi) = (&piece.0, piece.1, piece.2);
        let t0 = shadow.span().0 as usize + lo;
        let block = t0 / BLOCK;
        let verdict = (|| {
            if hi <= lo || (t0 + (hi - lo) - 1) / BLOCK != block {
                count(c::NO_SPAN, 1);
                return Verdict::Eager;
            }
            if let Some(g) = TABLE.get(block) {
                let grp = &G[g];
                let s = wait_while(g, |s| s == SWAPPING);
                let mine = std::ptr::eq(grp.shadow.load(Ordering::Acquire), Arc::as_ptr(shadow))
                    && grp.blocks[0].load(Ordering::Relaxed) == block;
                if s == DEFERRED && mine && !r.pending[start_pending..].iter().any(|p| p.g == g) {
                    // Flagged again before anything read it: the hull of both
                    // fetched into a second slot.
                    let (clo, chi) = (grp.lo.load(Ordering::Relaxed), grp.hi.load(Ordering::Relaxed));
                    let (nlo, nhi) = if grp.data.load(Ordering::Relaxed) != 0 { (clo.min(lo), chi.max(hi)) } else { (lo, hi) };
                    if !shadow.lazy_nothing_waiting(nlo, nhi, behind) {
                        count(c::NO_PENDING, 1);
                        resolve(g, Why::Landing);
                        return Verdict::Eager;
                    }
                    let Some((slot, base)) = crate::host::lazy_stash_slot() else {
                        count(c::NO_ROOM, 1);
                        resolve(g, Why::Landing);
                        return Verdict::Eager;
                    };
                    let data = base + (shadow.span().0 as usize + nlo) % BLOCK;
                    r.pending.push(Pending { g, lo: nlo, hi: nhi, slot, data, behind });
                    sends.push((shadow.clone(), nlo, nhi, slot, (shadow.span().0 as usize + nlo) % BLOCK));
                    count(c::REFLAGS, 1);
                    return Verdict::Deferred;
                }
                if matches!(s, DEFERRED | RESOLVING) {
                    // Another copy's (or alias's) group: its bytes go in first.
                    count(c::OTHER_COPY, 1);
                    resolve(g, Why::Landing);
                } else if s == PRESENT {
                    count(c::LATE, 1);
                }
                return Verdict::Eager;
            }
            if r.heat.get(&block).is_some_and(|h| h.1 > frame) {
                count(c::HOT, 1);
                return Verdict::Eager;
            }
            if r.nope.get(&block).is_some_and(|&until| until > frame) {
                count(c::CACHED, 1);
                return Verdict::Eager;
            }
            if !shadow.lazy_card() {
                count(c::NO_CARD, 1);
                r.nope.insert(block, frame + 30);
                return Verdict::Eager;
            }
            if !shadow.lazy_nothing_waiting(lo, hi, behind) {
                count(c::NO_PENDING, 1);
                return Verdict::Eager;
            }
            let page = block * BLOCK;
            let mut addrs = crate::inplace::pages_of(page);
            let private = addrs.is_empty() && private_ok();
            if private {
                addrs.push(page);
            }
            let beside = if private { None } else { crate::inplace::beside(page, BLOCK) };
            if addrs.is_empty() || addrs.len() > ALIASES || !addrs.contains(&page) || (!private && beside.is_none()) {
                count(c::NO_VIEW, 1);
                r.nope.insert(block, frame + 30);
                return Verdict::Eager;
            }
            // (The primary block first.)
            addrs.retain(|&a| a != page);
            addrs.insert(0, page);
            let Some(_reading) = crate::watch::title_memory_reading_now() else {
                count(c::NO_REMAP, 1);
                return Verdict::Eager;
            };
            let blocks: Vec<usize> = addrs.iter().map(|a| a / BLOCK).collect();
            if blocks.iter().any(|&b| TABLE.get(b).is_some()) {
                count(c::OTHER_COPY, 1);
                return Verdict::Eager;
            }
            if blocks.iter().any(|&b| !(loose() || crate::watch::block_published(b))) {
                count(c::NO_CLEAN, 1);
                return Verdict::Eager;
            }
            // Only this copy holds these bytes: another copy, or a placed
            // texture, would go on showing the old ones.
            if let Some(why) = blocks.iter().map(|&b| shadow.lazy_alone((b * BLOCK) as u64, page as u64)).find(|&w| w != 0) {
                count(if why == 1 { c::NO_SHARED } else { c::NO_TEXTURE }, 1);
                r.nope.insert(block, frame + 30);
                return Verdict::Eager;
            }
            // (No `VirtualQuery` here: in the title's big private regions it
            // walks the page tables to the region's end, a millisecond each,
            // 650 ms in the first second of a phase in fps5-b. The protection
            // there is what closing the block returns.)
            let Some((slot, base)) = crate::host::lazy_stash_slot() else {
                count(c::NO_ROOM, 1);
                return Verdict::Eager;
            };
            let Some(g) = r.group() else {
                crate::host::lazy_stash_free(slot);
                count(c::NO_ROOM, 1);
                return Verdict::Eager;
            };
            let grp = &G[g];
            grp.shadow.store(Arc::into_raw(shadow.clone()).cast_mut(), Ordering::Release);
            grp.lo.store(lo, Ordering::Relaxed);
            grp.hi.store(hi, Ordering::Relaxed);
            grp.data.store(0, Ordering::Relaxed);
            grp.held.store(shadow.card_data() as usize, Ordering::Relaxed);
            grp.title.store(shadow.span().0 as usize, Ordering::Relaxed);
            grp.beside.store(beside.unwrap_or(0), Ordering::Relaxed);
            grp.nblocks.store(blocks.len(), Ordering::Relaxed);
            for (i, &b) in blocks.iter().enumerate() {
                grp.blocks[i].store(b, Ordering::Relaxed);
                grp.wanted[i].store(PAGE_READWRITE, Ordering::Relaxed);
            }
            grp.changed.store(0, Ordering::Relaxed);
            // SWAPPING (ours) while the blocks are closed: a fault meanwhile
            // waits a few microseconds for the protections to be noted.
            ANY.fetch_add(1, Ordering::AcqRel);
            grp.state.store(SWAPPING, Ordering::Release);
            for &b in &blocks {
                TABLE.set(b, Some(g));
            }
            r.owned.insert(g, (shadow.clone(), None));
            let mut done = 0;
            let mut fit = true;
            for (i, &b) in blocks.iter().enumerate() {
                // Page by page: a block whose pages are not all alike (fps5-c:
                // its first page's protection given back to all sixteen made
                // a title write fault for good) is not deferred.
                let Some(old) = close_block(b) else {
                    fit = false;
                    break;
                };
                grp.wanted[i].store(old, Ordering::Release);
                done += 1;
            }
            // Not all closed, or a write of the title's that came before the
            // block was closed: the deferral ends at once (nothing merged;
            // the protections given back), the piece is eager.
            if !fit || done != blocks.len() || blocks.iter().any(|&b| !(loose() || crate::watch::block_published(b))) {
                count(c::NO_PROTECT, 1);
                for i in 0..done {
                    protect(blocks[i], grp.wanted[i].load(Ordering::Acquire));
                }
                crate::host::lazy_stash_free(slot);
                grp.present_at.store(frame, Ordering::Relaxed);
                grp.why.store(Why::Another as u32, Ordering::Relaxed);
                grp.state.store(PRESENT, Ordering::Release);
                ANY.fetch_sub(1, Ordering::AcqRel);
                return Verdict::Eager;
            }
            // DEFERRED with no data yet: a fault before the fetch is in only
            // opens the block (the title was not told of this flight yet), and
            // `fetched` then merges what came as a landing does.
            grp.state.store(DEFERRED, Ordering::Release);
            r.pending.push(Pending { g, lo, hi, slot, data: base + t0 % BLOCK, behind });
            sends.push((shadow.clone(), lo, hi, slot, t0 % BLOCK));
            count(c::NEW_GROUPS, 1);
            Verdict::Deferred
        })();
        match verdict {
            Verdict::Deferred => {}
            Verdict::Eager => out.push(piece),
        }
    }
    if !sends.is_empty() {
        let bytes: u64 = sends.iter().map(|s| (s.2 - s.1) as u64).sum();
        if crate::host::lazy_stash_send(&sends) {
            count(c::STASHED, bytes);
        } else {
            // Nothing came: the deferrals of this landing end (resolved with
            // what they held) and their pieces are fetched and merged now.
            count(c::SEND_FAIL, 1);
            let mine: Vec<Pending> = r.pending.drain(start_pending..).collect();
            for p in mine {
                crate::host::lazy_stash_free(p.slot);
                resolve(p.g, Why::Another);
                let shadow = r.owned.get(&p.g).map(|o| o.0.clone());
                if let Some(shadow) = shadow {
                    out.push((shadow, p.lo, p.hi, 0));
                }
            }
        }
    }
    drop(guard);
    count(c::SPLIT_NS, began.elapsed().as_nanos() as u64);
    out
}

/// After the landing's fetches (before the title is told of the flights):
/// the stash's fetches waited for, and each group given its new slot. A
/// group resolved meanwhile (with older bytes, or none) gets the new ones
/// merged now as a landing would.
pub(crate) fn fetched() {
    let began = std::time::Instant::now();
    let mut guard = relay();
    let Some(r) = guard.as_mut() else {
        return;
    };
    if r.pending.is_empty() {
        return;
    }
    crate::host::lazy_stash_wait();
    let pending = std::mem::take(&mut r.pending);
    let mut notes = crate::land::Notes::default();
    for p in pending {
        let grp = &G[p.g];
        let swapped = loop {
            match grp.state.compare_exchange(DEFERRED, SWAPPING, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => break true,
                Err(SWAPPING) | Err(RESOLVING) => {
                    wait_while(p.g, |s| s == SWAPPING || s == RESOLVING);
                }
                Err(_) => break false,
            }
        };
        let Some(owned) = r.owned.get_mut(&p.g) else {
            crate::host::lazy_stash_free(p.slot);
            continue;
        };
        if swapped {
            grp.lo.store(p.lo, Ordering::Relaxed);
            grp.hi.store(p.hi, Ordering::Relaxed);
            grp.data.store(p.data, Ordering::Relaxed);
            grp.state.store(DEFERRED, Ordering::Release);
            if let Some(old) = owned.1.replace(p.slot) {
                crate::host::lazy_stash_free(old);
            }
        } else {
            // Resolved meanwhile: these bytes are landed now, as before.
            count(c::LATE, 1);
            let shadow = owned.0.clone();
            // SAFETY: the slot's bytes, fetched and waited for; nothing else
            // writes the slot until it is freed below.
            unsafe { shadow.land_card(&[(p.lo, p.hi, p.data)], p.behind, &mut notes) };
            notes.flush();
            crate::host::lazy_stash_free(p.slot);
        }
    }
    drop(guard);
    count(c::FETCHED_NS, began.elapsed().as_nanos() as u64);
}
