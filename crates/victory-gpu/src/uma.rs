//! Zero-copy title memory (`VICTORY_UMA=1`, default off).
//!
//! The console has one memory, which CPU and GPU both address. The host
//! keeps copies of title memory ([`crate::shadow`]) and carries bytes in and
//! back. Where the title's memory is committed memory the process owns, D3D12
//! can open it as a heap (`OpenExistingHeapFromAddress*`) and a buffer placed
//! in that heap IS the title's memory, for the GPU too: nothing is copied in,
//! nothing copied back, nothing guarded or compared. `crate::inplace` (the
//! lineage's `VICTORY_INPLACE`) has the shadow side of that: a [`Placed`]
//! buffer makes a [`crate::shadow::Shadow`] that does nothing when asked to
//! copy. What this file adds, for `VICTORY_UMA=1` (which replaces
//! `VICTORY_INPLACE` while on):
//!
//! - heaps shared by many buffers: the lineage opened one heap per buffer
//!   (a `VirtualQuery` of the title's big allocations each time: hundreds of
//!   ms). Here a heap covers a chunk of memory (`VICTORY_UMA_CHUNK_MB`, 64) or
//!   the buffer if bigger, and each shadow's buffer is placed in it.
//! - no `VirtualQuery` ever. Private memory is what the runtime's own record
//!   says ([`title_allocation`], from `XMemVirtualAlloc`; freed through
//!   [`freed`], from the title's `VirtualFree`). Mapped memory is the
//!   runtime's table of views ([`crate::inplace::run_around`]): the title's
//!   views of physical pages are not opened (the title could not unmap them
//!   while a heap stood on them) but our own view of the whole section is,
//!   at the offsets the title's views show, so the title may map and unmap
//!   as it likes; a shadow over mapped memory is dropped when the runtime
//!   tells that its view changed (`inplace::title_memory_freeing`).
//! - what shaders write is told to the other trackers ([`wrote`], after the
//!   submission ran) and declared as before to the flights' rules, which keep
//!   the host from reading title memory a flight writes before it has run.
//! - the probes: a line per interval ([`log`]) and one self-check at the
//!   first use ([`mode`]) with the cost of each step.
//!
//! Facts it rests on are measured on the RTX 3080 (umaexp, exp1/exp2 of the
//! first prototype): heaps open over committed `VirtualAlloc` memory and
//! views of pagefile sections; placed buffers there need
//! `ALLOW_CROSS_ADAPTER`; CPU writes, GPU read/write and CPU reads are
//! coherent with no copy; a swizzled texture cannot be placed (textures stay
//! copied). `OpenExistingHeapFromAddress1` (`ID3D12Device13`) is needed for
//! parts of memory and for sections; without it only whole `VirtualAlloc`
//! allocations of under 4 GiB open (the self-check says which).
//!
//! shortcut: the title's `VirtualProtect` is not seen (a protection change
//! under an open heap fails there with error 87, and a change of what is
//! committed is not seen unless it goes through `XMemVirtualAlloc` or the
//! title's `VirtualFree`); a free of memory under a heap fails until the
//! last list that holds the heap is done ([`freed`] counts it).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use crate::inplace::{Cover, From as Open, Placed};

const K64: u64 = 64 << 10;
/// Heaps over 4 GiB - 64 KiB are refused by D3D12.
const MAX_HEAP: u64 = (4096 << 20) - (64 << 10);
/// `MEM_WRITE_WATCH` (not in windows-sys' memory module).
const MEM_WRITE_WATCH: u32 = 0x20_0000;

/// Whether `VICTORY_UMA` is on.
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = crate::switch::on("VICTORY_UMA");
        if on {
            tracing::info!(
                "zero-copy title memory (VICTORY_UMA=1): title buffers and windows are opened as host heaps where the runtime knows the memory is committed (chunk {} MB, at most {} MB open; VICTORY_UMA_CHUNK_MB, VICTORY_UMA_MB; VICTORY_UMA_PRIVATE={}, VICTORY_UMA_MAPPED={}, VICTORY_UMA_BIG={})",
                chunk() >> 20,
                limit() >> 20,
                u8::from(flag("VICTORY_UMA_PRIVATE")),
                u8::from(flag("VICTORY_UMA_MAPPED")),
                u8::from(flag("VICTORY_UMA_BIG")),
            );
        }
        on
    })
}

/// A switch that is on unless its variable is `0`.
fn flag(name: &str) -> bool {
    std::env::var_os(name).is_none_or(|v| v != "0")
}

fn number(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default)
}

/// Bytes one heap covers at least (`VICTORY_UMA_CHUNK_MB`, 64). Opening pins
/// every page of the heap into the PC's memory (8 ms for 64 MB measured):
/// the more is opened at once, the fewer the opens and the more pinned that
/// nothing asked for.
fn chunk() -> u64 {
    static C: OnceLock<u64> = OnceLock::new();
    *C.get_or_init(|| (number("VICTORY_UMA_CHUNK_MB", 64).max(1) << 20).next_multiple_of(K64))
}

/// The most bytes of heaps open at once (`VICTORY_UMA_MB`, 8192).
fn limit() -> u64 {
    static C: OnceLock<u64> = OnceLock::new();
    *C.get_or_init(|| number("VICTORY_UMA_MB", 8192) << 20)
}

/// Whether the big title buffers that `VICTORY_CARD_BUFFERS` keeps in the
/// card's own memory are opened in place too (`VICTORY_UMA_BIG`, on): the
/// card then reads them over the bus (19 GB/s measured against 689 GB/s),
/// and the copy-back, the compare and the sends of them are gone.
pub(crate) fn big() -> bool {
    static C: OnceLock<bool> = OnceLock::new();
    enabled() && *C.get_or_init(|| flag("VICTORY_UMA_BIG"))
}

/// Counters, per interval (swapped out by [`log`]).
#[derive(Clone, Copy)]
#[repr(usize)]
enum C {
    /// Shadows served in place, by memory: buffers and windows, private and
    /// mapped; their bytes.
    ServedPrivate,
    ServedPrivateBytes,
    ServedMapped,
    ServedMappedBytes,
    ServedWindows,
    /// Asks left to the copy path, and their bytes.
    Copied,
    CopiedBytes,
    /// Refreshes of in-place shadows that were not copied (declared bytes).
    Refreshes,
    RefreshBytes,
    /// Declared shader-write ranges of in-place shadows (copy-backs, snapshots
    /// and compares not made), and their bytes.
    Backs,
    BackBytes,
    /// Pages told to the other trackers, and the microseconds that took.
    Told,
    ToldUs,
    /// Heaps opened, their bytes, the microseconds of the opens.
    Opens,
    OpenBytes,
    OpenUs,
    /// Buffers placed and the microseconds.
    Placements,
    PlaceUs,
    /// Heaps dropped because their memory was freed or its section went.
    Dropped,
    /// Frees that hit a region with an open heap.
    FreesUnderHeap,
    N,
}

static COUNTS: [AtomicU64; C::N as usize] = [const { AtomicU64::new(0) }; C::N as usize];

fn add(c: C, n: u64) {
    COUNTS[c as usize].fetch_add(n, Relaxed);
}

/// Since the start: heaps open now (private, mapped) and their bytes; heaps
/// opened and refused.
static OPEN_NOW: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
static OPENED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Why asks were left to the copy path, since the start: (asks, bytes).
static REFUSED: Mutex<BTreeMap<&'static str, (u64, u64)>> = Mutex::new(BTreeMap::new());

fn refuse(why: &'static str, len: u64) -> Option<Placed> {
    add(C::Copied, 1);
    add(C::CopiedBytes, len);
    let mut r = REFUSED.lock().unwrap_or_else(|e| e.into_inner());
    let e = r.entry(why).or_default();
    e.0 += 1;
    e.1 += len;
    if e.0 == 1 {
        tracing::info!(why, len, "zero copy: first ask left to the copy path for this reason");
    }
    None
}

/// A refresh of `len` declared bytes of an in-place shadow was not copied.
pub(crate) fn note_refresh(len: usize) {
    add(C::Refreshes, 1);
    add(C::RefreshBytes, len as u64);
}

// ---- the runtime's record of private allocations ----

/// An allocation `XMemVirtualAlloc` made: its length and the read-write
/// committed ranges of it (sorted, apart).
struct Alloc {
    len: usize,
    committed: Vec<(usize, usize)>,
}

static ALLOCS: Mutex<BTreeMap<usize, Alloc>> = Mutex::new(BTreeMap::new());

fn add_range(ranges: &mut Vec<(usize, usize)>, from: usize, to: usize) {
    ranges.push((from, to));
    ranges.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for &(a, b) in ranges.iter() {
        match out.last_mut() {
            Some(last) if a <= last.1 => last.1 = last.1.max(b),
            _ => out.push((a, b)),
        }
    }
    *ranges = out;
}

fn cut_range(ranges: &mut Vec<(usize, usize)>, from: usize, to: usize) {
    let mut out = Vec::with_capacity(ranges.len() + 1);
    for &(a, b) in ranges.iter() {
        if b <= from || a >= to {
            out.push((a, b));
            continue;
        }
        if a < from {
            out.push((a, from));
        }
        if b > to {
            out.push((to, b));
        }
    }
    *ranges = out;
}

/// The runtime gave the title `[at, at + size)` (`XMemVirtualAlloc`): `fresh`
/// if it is a new allocation (the runtime chose the address), else memory
/// reserved before; `committed` if it is committed read-write.
pub fn title_allocation(at: usize, size: usize, fresh: bool, committed: bool) {
    if size == 0 || !enabled() {
        return;
    }
    let mut allocs = ALLOCS.lock().unwrap_or_else(|e| e.into_inner());
    let end = at.saturating_add(size);
    if fresh {
        // (An allocation the title freed without our being told: its record
        // goes with what is new there.)
        let stale: Vec<usize> = allocs.range(..end).filter(|(b, a)| **b + a.len > at).map(|(b, _)| *b).collect();
        for b in stale {
            allocs.remove(&b);
        }
        let mut alloc = Alloc { len: size, committed: Vec::new() };
        if committed {
            alloc.committed.push((at, end));
        }
        allocs.insert(at, alloc);
        return;
    }
    let Some((&base, alloc)) = allocs.range_mut(..=at).next_back() else {
        return;
    };
    if at >= base + alloc.len {
        return;
    }
    let end = end.min(base + alloc.len);
    if committed {
        add_range(&mut alloc.committed, at, end);
    } else {
        // A new protection: not what a heap may be opened over.
        cut_range(&mut alloc.committed, at, end);
    }
}

/// The title is about to free or decommit `[at, at + len)` (no length: the
/// allocation at `at`): the heaps over it go, and the shadows made in them.
/// Called by `watch::title_memory_freed` ahead of the free.
pub(crate) fn freed(at: usize, len: usize) {
    if !enabled() {
        return;
    }
    let range = {
        let mut allocs = ALLOCS.lock().unwrap_or_else(|e| e.into_inner());
        let Some((&base, alloc)) = allocs.range_mut(..=at).next_back().filter(|(b, a)| at < **b + a.len) else {
            return;
        };
        if len == 0 {
            let (from, to) = (base, base + alloc.len);
            allocs.remove(&base);
            (from, to)
        } else {
            let to = at.saturating_add(len);
            cut_range(&mut alloc.committed, at, to);
            (at, to)
        }
    };
    let gone: Vec<Arc<Region>> = {
        let mut private = PRIVATE.lock().unwrap_or_else(|e| e.into_inner());
        let keys: Vec<u64> = private.range(..range.1 as u64).filter(|(lo, r)| **lo + r.len > range.0 as u64).map(|(lo, _)| *lo).collect();
        keys.into_iter().filter_map(|k| private.remove(&k)).collect()
    };
    if gone.is_empty() {
        return;
    }
    drop_regions(&gone, false);
    add(C::FreesUnderHeap, gone.len() as u64);
    static SAID: AtomicU64 = AtomicU64::new(0);
    if SAID.fetch_add(1, Relaxed) < 8 {
        tracing::warn!(
            at = format_args!("{:#x}", range.0),
            len = range.1 - range.0,
            heaps = gone.len(),
            "zero copy: the title frees memory that a heap is open over; its free fails until the last list holding the heap is done"
        );
    }
    // The shadows made over it, out of the tables.
    crate::inplace::title_memory_freeing(range.0, range.1 - range.0);
}

/// The runtime lets go of `section`: its heaps go (our view of the whole
/// section, which [`crate::inplace`] unmaps, cannot be unmapped while one
/// stands: that unmapping fails and the view stays, a leak of address space).
pub(crate) fn section_gone(section: usize) {
    if !enabled() {
        return;
    }
    let gone: Vec<Arc<Region>> = {
        let mut mapped = MAPPED.lock().unwrap_or_else(|e| e.into_inner());
        let keys: Vec<(usize, u64)> = mapped.range((section, 0)..=(section, u64::MAX)).map(|(k, _)| *k).collect();
        keys.into_iter().filter_map(|k| mapped.remove(&k)).collect()
    };
    drop_regions(&gone, true);
}

fn drop_regions(gone: &[Arc<Region>], mapped: bool) {
    add(C::Dropped, gone.len() as u64);
    for r in gone {
        OPEN_NOW[usize::from(mapped) * 2].fetch_sub(1, Relaxed);
        OPEN_NOW[usize::from(mapped) * 2 + 1].fetch_sub(r.len, Relaxed);
    }
}

// ---- regions ----

/// A heap over a stretch of memory, and where the stretch is.
struct Region {
    heap: ID3D12Heap,
    /// The stretch: a title address (private) or an offset in a section
    /// (mapped), and its bytes.
    lo: u64,
    len: u64,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Region {}
// SAFETY: as above.
unsafe impl Sync for Region {}

/// By first address.
static PRIVATE: Mutex<BTreeMap<u64, Arc<Region>>> = Mutex::new(BTreeMap::new());
/// By section and first offset.
static MAPPED: Mutex<BTreeMap<(usize, u64), Arc<Region>>> = Mutex::new(BTreeMap::new());

/// Several heaps that were refused: the memory kind is given up.
static REFUSALS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
static GIVEN_UP: [AtomicBool; 2] = [const { AtomicBool::new(false) }; 2];

fn open_heap(device: &ID3D12Device, from: Open, bytes: u64, mapped: bool) -> Option<ID3D12Heap> {
    if OPEN_NOW[1].load(Relaxed) + OPEN_NOW[3].load(Relaxed) + bytes > limit() {
        refuse("over VICTORY_UMA_MB of heaps open", bytes);
        return None;
    }
    let started = Instant::now();
    let opened = inplace_open(device, from);
    let us = started.elapsed().as_micros() as u64;
    add(C::OpenUs, us);
    match opened {
        Ok(heap) => {
            add(C::Opens, 1);
            add(C::OpenBytes, bytes);
            OPENED[0].fetch_add(1, Relaxed);
            OPEN_NOW[usize::from(mapped) * 2].fetch_add(1, Relaxed);
            OPEN_NOW[usize::from(mapped) * 2 + 1].fetch_add(bytes, Relaxed);
            if us > 20_000 {
                tracing::info!(mb = bytes >> 20, ms = us as f64 / 1e3, mapped, "zero copy: a heap opened (slow)");
            }
            Some(heap)
        }
        Err(e) => {
            OPENED[1].fetch_add(1, Relaxed);
            let n = REFUSALS[usize::from(mapped)].fetch_add(1, Relaxed) + 1;
            if n <= 8 {
                tracing::warn!(mb = bytes >> 20, mapped, error = %e, "zero copy: the host refused to open memory as a heap");
            }
            if n >= 8 {
                GIVEN_UP[usize::from(mapped)].store(true, Relaxed);
            }
            None
        }
    }
}

fn inplace_open(device: &ID3D12Device, from: Open) -> windows::core::Result<ID3D12Heap> {
    crate::inplace::open(device, from)
}

/// What can be opened: the self-check's verdict, less what was given up.
#[derive(Clone, Copy)]
struct Mode {
    private: bool,
    mapped: bool,
    /// `ID3D12Device13`: parts of memory open.
    parts: bool,
}

fn mode() -> Mode {
    static CHECK: OnceLock<Mode> = OnceLock::new();
    let m = *CHECK.get_or_init(|| {
        let m = self_check();
        Mode {
            private: m.private && flag("VICTORY_UMA_PRIVATE"),
            mapped: m.mapped && flag("VICTORY_UMA_MAPPED"),
            parts: m.parts,
        }
    });
    Mode {
        private: m.private && !GIVEN_UP[0].load(Relaxed),
        mapped: m.mapped && !GIVEN_UP[1].load(Relaxed),
        parts: m.parts,
    }
}

/// Title memory `[start, start + len)` in place: a buffer over a heap on
/// it, shaped as `crate::inplace::place_for` gives them; `None` leaves the
/// caller to make a copy, with the reason counted.
pub(crate) fn place(start: u64, len: u64, window: bool) -> Option<Placed> {
    let mode = mode();
    let Some(host) = crate::host::host() else {
        return refuse("no host device", len);
    };
    let at = start & !(K64 - 1);
    let Some(end) = start.checked_add(len.max(1)) else {
        return refuse("range past the address space", len);
    };
    let rounded = end.next_multiple_of(K64);
    let started = Instant::now();
    let found = match crate::inplace::cover(at as usize, (rounded - at) as usize) {
        Cover::Run(section, offset) => {
            if !mode.mapped {
                return refuse("mapped memory: not available (self-check, VICTORY_UMA_MAPPED, or given up)", len);
            }
            mapped_region(&host.device, mode, section, offset, at, rounded - at).map(|(r, off, size)| (r, off, size, true))
        }
        Cover::None => {
            if !mode.private {
                return refuse("private memory: not available (self-check, VICTORY_UMA_PRIVATE, or given up)", len);
            }
            private_region(&host.device, mode, at, end, rounded).map(|(r, off, size)| (r, off, size, false))
        }
        Cover::Holes => return refuse("mapped memory with a hole in it", len),
        Cover::Scattered => return refuse("mapped memory over scattered physical pages", len),
    };
    let (region, offset, size, mapped) = match found {
        Ok(f) => f,
        Err(why) => return refuse(why, len),
    };
    let placed = crate::inplace::place(&host.device, &region.heap, offset, size);
    add(C::PlaceUs, started.elapsed().as_micros() as u64);
    let buffer = match placed {
        Ok(buffer) => buffer,
        Err(e) => {
            static SAID: AtomicU64 = AtomicU64::new(0);
            if SAID.fetch_add(1, Relaxed) < 8 {
                tracing::warn!(start = format_args!("{start:#x}"), len, error = %e, "zero copy: no buffer could be placed in the heap");
            }
            return refuse("CreatePlacedResource refused", len);
        }
    };
    add(C::Placements, 1);
    if window {
        add(C::ServedWindows, 1);
    }
    let (n, bytes) = if mapped { (C::ServedMapped, C::ServedMappedBytes) } else { (C::ServedPrivate, C::ServedPrivateBytes) };
    add(n, 1);
    add(bytes, len);
    let bias = start - at;
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() } + bias;
    Some(Placed { buffer, heap: region.heap.clone(), gpu, bias })
}

/// The heap over private title memory `[at, rounded)` (`end` is where the
/// buffer ends), opened if need be, the buffer's offset in it and its size.
fn private_region(
    device: &ID3D12Device,
    mode: Mode,
    at: u64,
    end: u64,
    rounded: u64,
) -> Result<(Arc<Region>, u64, u64), &'static str> {
    let (base, alloc_len, committed) = {
        let allocs = ALLOCS.lock().unwrap_or_else(|e| e.into_inner());
        let Some((&base, alloc)) = allocs.range(..=at as usize).next_back().filter(|(b, a)| (at as usize) < **b + a.len) else {
            return Err("not memory the runtime allocated (its C heap, upload memory, other reservations)");
        };
        let Some(&(from, to)) = alloc.committed.iter().find(|r| r.0 <= at as usize && (at as usize) < r.1) else {
            return Err("not committed read-write when the runtime gave it");
        };
        (base as u64, alloc.len as u64, (from as u64, to as u64, alloc.committed.len()))
    };
    let (from, to, ranges) = committed;
    let hi = rounded.min(to);
    if hi < end {
        return Err("range reaches past the committed memory");
    }
    let need = hi - at;
    if need > MAX_HEAP {
        return Err("buffer of 4 GiB or more");
    }
    let mut private = PRIVATE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(r) = private.range(..=at).rev().take(16).map(|(_, r)| r).find(|r| r.lo + r.len >= hi) {
        return Ok((r.clone(), at - r.lo, need));
    }
    let c = chunk();
    let (a, b) = if mode.parts {
        let a = from.max(base + (at - base) / c * c).next_multiple_of(K64);
        let b = to.min(base + (hi - base).next_multiple_of(c)).max(hi);
        if b - a > MAX_HEAP { (at, hi) } else { (a.min(at), b) }
    } else if from == base && to == base + alloc_len && ranges == 1 && alloc_len <= MAX_HEAP {
        // No parts of memory: the whole allocation, which is what opens.
        (base, base + alloc_len)
    } else {
        return Err("no ID3D12Device13: only a whole committed allocation under 4 GiB opens");
    };
    let from_where = if mode.parts { Open::Part(a as usize, (b - a) as usize) } else { Open::Region(a as usize) };
    let Some(heap) = open_heap(device, from_where, b - a, false) else {
        return Err("the host refused to open it as a heap");
    };
    let region = Arc::new(Region { heap, lo: a, len: b - a });
    private.insert(a, region.clone());
    Ok((region, at - a, need))
}

/// As [`private_region`], for physical pages the title maps: the heap is
/// over our own view of the whole section, at the offsets the title's views
/// show (`offset` is that of `at`).
fn mapped_region(
    device: &ID3D12Device,
    mode: Mode,
    section: usize,
    offset: u64,
    at: u64,
    need: u64,
) -> Result<(Arc<Region>, u64, u64), &'static str> {
    if !mode.parts {
        return Err("no ID3D12Device13: a part of a section's view cannot open");
    }
    if need > MAX_HEAP {
        return Err("buffer of 4 GiB or more");
    }
    let c = chunk();
    // The run of views around: every page in it is committed (the runtime
    // commits pages before it shows them).
    let Some(run) = crate::inplace::run_around(at as usize, need as usize, c as usize) else {
        return Err("the views under it are not one run");
    };
    let need_hi = offset + need;
    let run_hi = run.offset + (run.last - run.first) as u64;
    let a = run.offset.max(offset / c * c);
    let b = run_hi.min(need_hi.next_multiple_of(c));
    let (a, b) = if b - a > MAX_HEAP { (offset, need_hi) } else { (a, b) };
    if a > offset || b < need_hi {
        return Err("the views under it are not one run");
    }
    let mut mapped = MAPPED.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(r) = mapped
        .range((section, 0)..=(section, offset))
        .rev()
        .take(16)
        .map(|(_, r)| r)
        .find(|r| r.lo + r.len >= need_hi)
    {
        return Ok((r.clone(), offset - r.lo, need));
    }
    let Some(window) = crate::inplace::window_base(section) else {
        return Err("no view of the whole section");
    };
    let Some(heap) = open_heap(device, Open::Part(window + a as usize, (b - a) as usize), b - a, true) else {
        return Err("the host refused to open it as a heap");
    };
    let region = Arc::new(Region { heap, lo: a, len: b - a });
    mapped.insert((section, a), region.clone());
    Ok((region, offset - a, need))
}

/// After a submission ran: its shaders wrote title memory `[at, at + len)` in
/// place (a declared range of an in-place shadow): copies the other trackers
/// hold of it (a texture's staging, a view's own copy) are stale.
pub(crate) fn wrote(at: usize, len: usize) {
    add(C::Backs, 1);
    add(C::BackBytes, len as u64);
    let started = Instant::now();
    let pages = crate::watch::gpu_wrote(at, len);
    add(C::Told, pages as u64);
    add(C::ToldUs, started.elapsed().as_micros() as u64);
}

// ---- the probes ----

/// The self-check: at the first use, one allocation and one section of the
/// host's own, opened, a buffer placed, the CPU's pointer to it compared with
/// the memory's address, released; each step timed, in one line.
fn self_check() -> Mode {
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Memory as m;
    use std::fmt::Write;
    fn note(line: &mut String, name: &str, started: Instant) {
        let _ = write!(line, " {name} {:.2} ms;", started.elapsed().as_secs_f64() * 1e3);
    }
    let mut line = String::new();
    let Some(host) = crate::host::host() else {
        tracing::warn!("zero copy self-check: no host device; every ask goes to the copy path");
        return Mode { private: false, mapped: false, parts: false };
    };
    let t = Instant::now();
    let parts = host.device.cast::<ID3D12Device13>().is_ok();
    let whole = host.device.cast::<ID3D12Device3>().is_ok();
    note(&mut line, "interfaces", t);
    const LEN: usize = 4 << 20;
    let identity = |buffer: &ID3D12Resource, expect: usize| -> bool {
        let mut p = std::ptr::null_mut();
        // SAFETY: a live buffer in a custom heap, mapped for a moment.
        let mapped = unsafe { buffer.Map(0, None, Some(&mut p)) };
        let same = mapped.is_ok() && p as usize == expect;
        // SAFETY: as above.
        unsafe { buffer.Unmap(0, None) };
        same
    };
    // private memory
    let mut private = false;
    {
        let t = Instant::now();
        // SAFETY: a fresh allocation, freed below.
        let p = unsafe { m::VirtualAlloc(std::ptr::null(), LEN, m::MEM_RESERVE | m::MEM_COMMIT | MEM_WRITE_WATCH, m::PAGE_READWRITE) } as usize;
        note(&mut line, "allocation", t);
        if p != 0 && (parts || whole) {
            // SAFETY: inside the allocation.
            unsafe { (p as *mut u32).write(0x600d) };
            let t = Instant::now();
            let from = if parts { Open::Part(p, LEN) } else { Open::Region(p) };
            match crate::inplace::open(&host.device, from) {
                Ok(heap) => {
                    note(&mut line, "open", t);
                    let t = Instant::now();
                    match crate::inplace::place(&host.device, &heap, 0, LEN as u64) {
                        Ok(buffer) => {
                            note(&mut line, "place", t);
                            private = true;
                            let _ = write!(line, " cpu pointer is the memory: {};", identity(&buffer, p));
                        }
                        Err(e) => {
                            let _ = std::fmt::Write::write_fmt(&mut line, format_args!(" place REFUSED {e};"));
                        }
                    }
                }
                Err(e) => {
                    let _ = std::fmt::Write::write_fmt(&mut line, format_args!(" open REFUSED {e};"));
                }
            }
            let t = Instant::now();
            // SAFETY: ours.
            let freed = unsafe { m::VirtualFree(p as *mut _, 0, m::MEM_RELEASE) } != 0;
            note(&mut line, if freed { "release" } else { "release REFUSED (heap still pinned)" }, t);
        }
    }
    // a section, as the runtime makes them (reserved, committed per page)
    let mut mapped = false;
    if parts {
        let t = Instant::now();
        // SAFETY: plain section creation.
        let handle = unsafe {
            m::CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(), m::PAGE_READWRITE | m::SEC_RESERVE, 0, LEN as u32, std::ptr::null())
        };
        // SAFETY: the whole of a section we made.
        let view = if handle.is_null() { 0 } else { unsafe { m::MapViewOfFile(handle, m::FILE_MAP_WRITE, 0, 0, LEN) }.Value as usize };
        // Committed through another view, as the runtime commits the pages
        // it shows (`xmem` `commit_shown`) and opened through ours.
        // SAFETY: the whole of the section again.
        let other = if handle.is_null() { 0 } else { unsafe { m::MapViewOfFile(handle, m::FILE_MAP_WRITE, 0, 0, LEN) }.Value as usize };
        // SAFETY: inside that view.
        let committed = other != 0 && !unsafe { m::VirtualAlloc(other as *const _, LEN, m::MEM_COMMIT, m::PAGE_READWRITE) }.is_null();
        note(&mut line, "section", t);
        if committed {
            // SAFETY: committed, writable.
            unsafe { (view as *mut u32).write(0x600d) };
            let t = Instant::now();
            match crate::inplace::open(&host.device, Open::Part(view, LEN)) {
                Ok(heap) => {
                    note(&mut line, "section open", t);
                    match crate::inplace::place(&host.device, &heap, 0, LEN as u64) {
                        Ok(buffer) => {
                            mapped = true;
                            let _ = write!(line, " section cpu pointer is the view: {};", identity(&buffer, view));
                        }
                        Err(e) => {
                            let _ = std::fmt::Write::write_fmt(&mut line, format_args!(" section place REFUSED {e};"));
                        }
                    }
                }
                Err(e) => {
                    let _ = std::fmt::Write::write_fmt(&mut line, format_args!(" section open REFUSED {e};"));
                }
            }
        } else {
            line.push_str(" section not committed (the reserved section's pages did not commit);");
        }
        let t = Instant::now();
        if view != 0 {
            // SAFETY: our view.
            let gone = unsafe { m::UnmapViewOfFile(m::MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) } != 0;
            note(&mut line, if gone { "unmap" } else { "unmap REFUSED (heap still pinned)" }, t);
        }
        if other != 0 {
            // SAFETY: our view.
            unsafe { m::UnmapViewOfFile(m::MEMORY_MAPPED_VIEW_ADDRESS { Value: other as *mut _ }) };
        }
        if !handle.is_null() {
            // SAFETY: our handle.
            unsafe { CloseHandle(handle) };
        }
    } else {
        line.push_str(" mapped memory needs ID3D12Device13 (OpenExistingHeapFromAddress1): not here;");
    }
    tracing::info!(
        "zero copy self-check: Device13 {parts}, Device3 {whole};{line} verdict: private memory {}, mapped memory {}",
        if private { "opens" } else { "DOES NOT OPEN" },
        if mapped { "opens" } else { "DOES NOT OPEN" },
    );
    Mode { private, mapped, parts }
}

/// The interval's lines: what was served in place and what copied, what was
/// not done because of it, what it cost, and why the rest was copied.
pub(crate) fn log(frames: f64) {
    if !enabled() || frames <= 0.0 {
        return;
    }
    let c = |k: C| COUNTS[k as usize].swap(0, Relaxed) as f64;
    let mb = |b: f64| b / 1e6 / frames;
    let per = |n: f64| n / frames;
    let (served_p, served_pb, served_m, served_mb, windows) =
        (c(C::ServedPrivate), c(C::ServedPrivateBytes), c(C::ServedMapped), c(C::ServedMappedBytes), c(C::ServedWindows));
    let (copied, copied_b) = (c(C::Copied), c(C::CopiedBytes));
    let (refreshes, refresh_b, backs, back_b) = (c(C::Refreshes), c(C::RefreshBytes), c(C::Backs), c(C::BackBytes));
    let (told, told_us) = (c(C::Told), c(C::ToldUs));
    let (opens, open_b, open_us) = (c(C::Opens), c(C::OpenBytes), c(C::OpenUs));
    let (placed, place_us) = (c(C::Placements), c(C::PlaceUs));
    let (dropped, frees) = (c(C::Dropped), c(C::FreesUnderHeap));
    tracing::info!(
        "zero copy, per frame: served in place {:.1} ranges {:.2} MB (private {:.1} / {:.2} MB, mapped {:.1} / {:.2} MB, {:.1} of them windows), \
         left to the copy path {:.1} ranges {:.2} MB; not copied: {:.0} refreshes of {:.1} MB declared, {:.0} declared shader writes of {:.1} MB (no snapshot, compare or copy-back); \
         told to other trackers {:.0} pages in {:.2} ms; heaps opened {:.2} ({:.1} MB, {:.2} ms), buffers placed {:.1} ({:.2} ms); heaps dropped {:.2}, frees under a heap {:.2}; \
         open now: {} private heaps {} MB, {} mapped heaps {} MB (of {} MB); opened {} refused {} since the start",
        per(served_p + served_m),
        mb(served_pb + served_mb),
        per(served_p),
        mb(served_pb),
        per(served_m),
        mb(served_mb),
        per(windows),
        per(copied),
        mb(copied_b),
        per(refreshes),
        mb(refresh_b),
        per(backs),
        mb(back_b),
        per(told),
        told_us / 1e3 / frames,
        per(opens),
        mb(open_b),
        open_us / 1e3 / frames,
        per(placed),
        place_us / 1e3 / frames,
        per(dropped),
        per(frees),
        OPEN_NOW[0].load(Relaxed),
        OPEN_NOW[1].load(Relaxed) >> 20,
        OPEN_NOW[2].load(Relaxed),
        OPEN_NOW[3].load(Relaxed) >> 20,
        limit() >> 20,
        OPENED[0].load(Relaxed),
        OPENED[1].load(Relaxed),
    );
    let r = REFUSED.lock().unwrap_or_else(|e| e.into_inner());
    if !r.is_empty() {
        let mut why: Vec<(&&str, &(u64, u64))> = r.iter().collect();
        why.sort_by_key(|(_, (_, bytes))| std::cmp::Reverse(*bytes));
        let text: Vec<String> = why.iter().take(8).map(|(why, (n, bytes))| format!("{why}: {n} asks {} MB", bytes >> 20)).collect();
        tracing::info!("zero copy, left to the copy path since the start: {}", text.join("; "));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn committed_ranges_join_and_cut() {
        let mut r = Vec::new();
        add_range(&mut r, 0x10000, 0x20000);
        add_range(&mut r, 0x30000, 0x40000);
        add_range(&mut r, 0x20000, 0x30000);
        assert_eq!(r, vec![(0x10000, 0x40000)]);
        cut_range(&mut r, 0x18000, 0x28000);
        assert_eq!(r, vec![(0x10000, 0x18000), (0x28000, 0x40000)]);
        cut_range(&mut r, 0, 0x100000);
        assert!(r.is_empty());
    }

    /// The views the runtime told of give the run a buffer lies in, and
    /// refuse one over a hole or another section.
    #[test]
    fn a_run_of_views_is_found_around_a_range() {
        let section = 0x7e57_0001usize;
        let k = K64 as usize;
        let base = 0x20_0000_0000usize;
        crate::inplace::title_memory_section(section, 8 * k as u64, false);
        for i in 0..4 {
            crate::inplace::title_memory_view(base + i * k, k, section, (i * k) as u64);
        }
        // a hole at page 4, then a page of another offset
        crate::inplace::title_memory_view(base + 5 * k, k, section, 6 * k as u64);
        let run = crate::inplace::run_around(base + k, k, 1 << 20).expect("one run");
        assert_eq!((run.first, run.last, run.offset), (base, base + 4 * k, 0));
        let small = crate::inplace::run_around(base + 2 * k, k, k).expect("one run");
        assert!(small.first <= base + k && small.last >= base + 3 * k);
        assert!(crate::inplace::run_around(base + 3 * k, 2 * k, k).is_none());
        assert!(crate::inplace::run_around(base + 5 * k, k, k).is_some());
        assert_eq!(crate::inplace::cover(base + 5 * k, k), Cover::Run(section, 6 * k as u64));
        crate::inplace::views_gone(base, 8 * k);
    }

    /// The runtime's record: a fresh allocation, a commit inside a
    /// reservation, a protection change, a decommit and a free.
    #[test]
    fn the_record_follows_the_title_calls() {
        // (The switch is read once; the record is plain data.)
        let mut allocs = BTreeMap::new();
        let mut alloc = Alloc { len: 0x100000, committed: Vec::new() };
        add_range(&mut alloc.committed, 0x7000_0000, 0x7001_0000);
        allocs.insert(0x7000_0000usize, alloc);
        let (&base, alloc) = allocs.range_mut(..=0x7000_8000usize).next_back().unwrap();
        assert_eq!(base, 0x7000_0000);
        cut_range(&mut alloc.committed, 0x7000_4000, 0x7000_8000);
        assert_eq!(alloc.committed, vec![(0x7000_0000, 0x7000_4000), (0x7000_8000, 0x7001_0000)]);
    }
}
