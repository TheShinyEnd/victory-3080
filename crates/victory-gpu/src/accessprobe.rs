//! A sampled probe of what the title's CPU does with the bytes shaders wrote
//! (`VICTORY_ACCESS_PROBE=<n>`: one 64 KiB page in `n` of those landed, 0 or
//! unset = off; a measuring aid, not for play).
//!
//! WHY. About 49 MB a frame are fetched from the card and merged into title
//! memory. Whether the title's CPU ever looks at them decides whether they
//! have to be landed every frame. Nothing tracked reads of title memory, so
//! this makes the question answerable: after a landing a sampled page is made
//! `PAGE_NOACCESS` (in every title view that shows it); the first access of
//! another thread than ours ends the probe (the handler gives the protection
//! back and the access goes on) and is counted as a read or a write of the
//! title; the next landing that comes to the page ends a probe nothing
//! touched (counted: not read since the last landing). Counted per copy in
//! the card's memory (its title range), which says where the bytes are.
//!
//! Soundness of the probe, not of play: a page whose protection was changed
//! by someone else meanwhile (the guards) is `lost`, not counted either way;
//! the kernel reading or writing a probed page (a file read into it) would
//! fail where it did not, so keep the sample small and only measure.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use crate::shadow::Shadow;

const PAGE: usize = 64 << 10;

/// One in this many pages (0: off).
pub(crate) fn every() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        let v = std::env::var("VICTORY_ACCESS_PROBE").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(0);
        // "1" would be every page: take it for the usual sample.
        if v == 1 { 16 } else { v }
    })
}

pub(crate) fn on() -> bool {
    every() != 0
}

static ARMED: AtomicUsize = AtomicUsize::new(0);

#[derive(Default, Clone)]
struct Stat {
    start: u64,
    len: u64,
    landed: u64,
    landings: u64,
    armed: u64,
    read: u64,
    written: u64,
    ours: u64,
    untouched: u64,
    lost: u64,
    unprobeable: u64,
    /// Frames from arming to the end of an untouched probe.
    exposed: u64,
}

struct Probe {
    /// Page address and the protection it had.
    pages: Vec<(usize, u32)>,
    copy: u64,
    frame: u64,
}

#[derive(Default)]
struct State {
    by_page: HashMap<usize, usize>,
    probes: HashMap<usize, Probe>,
    next: usize,
    stats: BTreeMap<u64, Stat>,
    window: HashMap<u64, Stat>,
    last_report: Option<std::time::Instant>,
    last_frame: u64,
}

fn state() -> MutexGuard<'static, State> {
    static S: OnceLock<Mutex<State>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(State::default())).lock().unwrap_or_else(|e| e.into_inner())
}

fn sampled(page: usize) -> bool {
    let n = every();
    n != 0 && (((page / PAGE) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as usize % n == 0
}

fn query(at: usize) -> Option<(u32, usize, bool)> {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    if n == 0 {
        return None;
    }
    let end = (info.BaseAddress as usize).saturating_add(info.RegionSize);
    Some((info.Protect, end, info.State == MEM_COMMIT))
}

fn protect(at: usize, prot: u32) -> bool {
    use windows_sys::Win32::System::Memory::VirtualProtect;
    let mut old = 0;
    // SAFETY: title memory; only its protection changes.
    unsafe { VirtualProtect(at as *const _, PAGE, prot, &mut old) != 0 }
}

/// Ends probe `id`: the protection of the pages back. How many pages were
/// found changed by someone else.
fn disarm(st: &mut State, id: usize) -> (Option<Probe>, usize) {
    use windows_sys::Win32::System::Memory::PAGE_NOACCESS;
    let Some(probe) = st.probes.remove(&id) else {
        return (None, 0);
    };
    let mut lost = 0;
    for &(at, prev) in &probe.pages {
        st.by_page.remove(&at);
        match query(at) {
            Some((p, _, true)) if p == PAGE_NOACCESS => {
                protect(at, prev);
            }
            _ => lost += 1,
        }
    }
    ARMED.fetch_sub(1, Ordering::Release);
    (Some(probe), lost)
}

fn bump(st: &mut State, copy: &Shadow, f: impl Fn(&mut Stat)) {
    let (start, len) = copy.span();
    let s = st.stats.entry(copy.id()).or_default();
    s.start = start;
    s.len = len;
    f(s);
    let w = st.window.entry(copy.id()).or_default();
    w.start = start;
    w.len = len;
    f(w);
}

/// A landing's pieces `(copy, from, to, _)` are about to be put into title
/// memory: they count as landed, and the probes of their pages end (nothing
/// of ours may fault on them while the merge writes).
pub(crate) fn landing(pieces: &[(Arc<Shadow>, usize, usize, u64)]) {
    if !on() {
        return;
    }
    let mut st = state();
    let frame = crate::frame::current();
    let mut seen: Vec<(usize, u64)> = Vec::new();
    for (copy, lo, hi, _) in pieces {
        if hi <= lo {
            continue;
        }
        let bytes = (hi - lo) as u64;
        bump(&mut st, copy, |s| {
            s.landed += bytes;
            s.landings += 1;
        });
        let t0 = copy.span().0 as usize + lo;
        let t1 = copy.span().0 as usize + hi - 1;
        for page in (t0 / PAGE..=t1 / PAGE).map(|p| p * PAGE) {
            let Some(&id) = st.by_page.get(&page) else {
                continue;
            };
            let (probe, lost) = disarm(&mut st, id);
            if let Some(probe) = probe {
                seen.push((id, probe.frame));
                let owner = probe.copy;
                let exposed = frame.saturating_sub(probe.frame);
                let s = st.stats.entry(owner).or_default();
                if lost == 0 {
                    s.untouched += 1;
                    s.exposed += exposed;
                } else {
                    s.lost += 1;
                }
                let w = st.window.entry(owner).or_default();
                if lost == 0 {
                    w.untouched += 1;
                    w.exposed += exposed;
                } else {
                    w.lost += 1;
                }
            }
        }
    }
}

/// The landing is done: sampled pages of it are probed.
pub(crate) fn landed(pieces: &[(Arc<Shadow>, usize, usize, u64)]) {
    use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE};
    if !on() {
        return;
    }
    let frame = crate::frame::current();
    let mut st = state();
    for (copy, lo, hi, _) in pieces {
        if hi <= lo {
            continue;
        }
        let (start, len) = copy.span();
        let (start, end) = (start as usize, (start + len) as usize);
        let t0 = start + lo;
        let t1 = start + hi - 1;
        for page in (t0 / PAGE..=t1 / PAGE).map(|p| p * PAGE) {
            if !sampled(page) || st.by_page.contains_key(&page) {
                continue;
            }
            if page < start || page + PAGE > end {
                bump(&mut st, copy, |s| s.unprobeable += 1);
                continue;
            }
            let mut addrs = crate::inplace::pages_of(page);
            if addrs.is_empty() {
                addrs.push(page);
            }
            if addrs.len() > 8 || !addrs.contains(&page) || addrs.iter().any(|a| st.by_page.contains_key(a)) {
                bump(&mut st, copy, |s| s.unprobeable += 1);
                continue;
            }
            let mut prevs = Vec::with_capacity(addrs.len());
            let ok = addrs.iter().all(|&a| match query(a) {
                Some((p, end, true)) if end >= a + PAGE && (p == PAGE_READONLY || p == PAGE_READWRITE) => {
                    prevs.push((a, p));
                    true
                }
                _ => false,
            });
            if !ok {
                bump(&mut st, copy, |s| s.unprobeable += 1);
                continue;
            }
            let mut done = 0;
            for &(a, _) in &prevs {
                if !protect(a, PAGE_NOACCESS) {
                    break;
                }
                done += 1;
            }
            if done != prevs.len() {
                for &(a, p) in &prevs[..done] {
                    protect(a, p);
                }
                bump(&mut st, copy, |s| s.unprobeable += 1);
                continue;
            }
            let id = st.next;
            st.next += 1;
            for &(a, _) in &prevs {
                st.by_page.insert(a, id);
            }
            st.probes.insert(id, Probe { pages: prevs, copy: copy.id(), frame });
            ARMED.fetch_add(1, Ordering::Release);
            bump(&mut st, copy, |s| s.armed += 1);
        }
    }
    drop(st);
    report();
}

/// Title memory `[at, at + len)` is about to be written by the kernel, mapped
/// or freed (the hooks of `crate::watch` that tell of it): the probes of its
/// pages end, so that nothing fails there that would not have.
pub(crate) fn release(at: usize, len: usize) {
    if ARMED.load(Ordering::Acquire) == 0 || len == 0 {
        return;
    }
    let mut st = state();
    let (first, last) = (at / PAGE, at.saturating_add(len).div_ceil(PAGE));
    let ids: Vec<usize> = if last - first > st.by_page.len() {
        st.by_page.iter().filter(|(p, _)| (first..last).contains(&(**p / PAGE))).map(|(_, id)| *id).collect()
    } else {
        (first..last).filter_map(|p| st.by_page.get(&(p * PAGE)).copied()).collect()
    };
    for id in ids {
        let _ = disarm(&mut st, id);
    }
}

/// First thing in the access-violation handler: an access to a probed page
/// ends the probe and goes on. Whether it was one.
pub(crate) fn fault(at: usize, write: bool) -> bool {
    if ARMED.load(Ordering::Acquire) == 0 {
        return false;
    }
    let page = at & !(PAGE - 1);
    let mut st = state();
    let Some(&id) = st.by_page.get(&page) else {
        return false;
    };
    let ours = std::thread::current().name().is_some_and(|n| n.starts_with("victory"));
    let owner = st.probes.get(&id).map(|p| p.copy);
    let _ = disarm(&mut st, id);
    if let Some(owner) = owner {
        let f = |s: &mut Stat| {
            if ours {
                s.ours += 1;
            } else if write {
                s.written += 1;
            } else {
                s.read += 1;
            }
        };
        f(st.stats.entry(owner).or_default());
        f(st.window.entry(owner).or_default());
    }
    true
}

fn report() {
    let mut st = state();
    let now = std::time::Instant::now();
    let Some(last) = st.last_report else {
        st.last_report = Some(now);
        st.last_frame = crate::frame::current();
        return;
    };
    if now.duration_since(last).as_secs() < 5 {
        return;
    }
    st.last_report = Some(now);
    let frames = crate::frame::current().saturating_sub(st.last_frame).max(1);
    st.last_frame = crate::frame::current();
    let mut rows: Vec<Stat> = st.window.values().cloned().collect();
    st.window.clear();
    rows.sort_by_key(|r| std::cmp::Reverse(r.landed));
    let total: u64 = rows.iter().map(|r| r.landed).sum();
    tracing::info!(
        "access probe (1 in {} pages of 64 KiB; {} frames): landed {:.1} MB a frame in {} copies; armed now {}",
        every(),
        frames,
        total as f64 / frames as f64 / 1e6,
        rows.len(),
        ARMED.load(Ordering::Relaxed)
    );
    for r in rows.iter().take(14) {
        tracing::info!(
            "access probe | copy {:#x}+{:.1} MB: landed {:.2} MB/frame ({} pieces); probes {}: read by title {}, written by title {}, by us {}, untouched until the next landing {} (mean {:.1} frames), lost {}, not probeable {}",
            r.start,
            r.len as f64 / 1e6,
            r.landed as f64 / frames as f64 / 1e6,
            r.landings,
            r.armed,
            r.read,
            r.written,
            r.ours,
            r.untouched,
            r.exposed as f64 / r.untouched.max(1) as f64,
            r.lost,
            r.unprobeable
        );
    }
    let (mut read, mut written, mut ours, mut untouched, mut lost) = (0, 0, 0, 0, 0);
    let mut mb = [0u64; 3];
    for r in &rows {
        read += r.read;
        written += r.written;
        ours += r.ours;
        untouched += r.untouched;
        lost += r.lost;
        let seen = r.read + r.written + r.untouched;
        // Landed bytes of the copies by what became of their probes.
        if seen == 0 {
            mb[2] += r.landed;
        } else if r.read + r.written == 0 {
            mb[0] += r.landed;
        } else {
            mb[1] += r.landed;
        }
    }
    tracing::info!(
        "access probe | all: read {read}, written {written}, by us {ours}, untouched {untouched}, lost {lost}; landed MB a frame in copies whose probes were never touched {:.1}, touched sometimes {:.1}, no probe finished {:.1}",
        mb[0] as f64 / frames as f64 / 1e6,
        mb[1] as f64 / frames as f64 / 1e6,
        mb[2] as f64 / frames as f64 / 1e6
    );
}
