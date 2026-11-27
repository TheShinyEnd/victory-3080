//! Arrival audit ("stream2" agent): the textures the pass over slots picked by
//! index refused for the card's budget ([`crate::residency::note_refused`]) and
//! what became of them.
//!
//! Measured (v7-m23t): ~17,000 slots (1,088 MB) counted "null" for four
//! minutes while the card had 300 to 550 MB free, and the queue held a few
//! thousand slots. Every five seconds this walks those slots and says how
//! many show a host copy now (stale entries: forgotten), show nothing of ours
//! (the title wrote another view: forgotten), wait in the queue, or are in
//! neither (dropped: null and nobody will look again). With
//! `VICTORY_NULL_RETRY=1` (default on) dropped slots are made due again once
//! the card has room, so every null texture whose bytes fit is tried.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

fn retry_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_NULL_RETRY").is_none_or(|v| v != "0"))
}

/// When (`coarse_ms` + 1) the pass over textures last found the card without room.
static FULL_AT: AtomicU64 = AtomicU64::new(0);

/// The pass found no room for textures now.
pub(crate) fn note_full() {
    let now = crate::host::coarse_ms();
    FULL_AT.store(now + 1, Relaxed);
    // Textures really missing while the card has no room: from when, and until when
    // the line is raised (`line_bonus_percent`).
    if crate::residency::null_bytes().0 == 0 {
        NULL_SINCE.store(0, Relaxed);
        return;
    }
    let since = NULL_SINCE.load(Relaxed);
    if since == 0 {
        NULL_SINCE.store(now + 1, Relaxed);
    } else if now.saturating_sub(since - 1) >= backlog_after_ms() {
        BONUS_UNTIL.store(now + 4_000, Relaxed);
    }
}

static NULL_SINCE: AtomicU64 = AtomicU64::new(0);
static BONUS_UNTIL: AtomicU64 = AtomicU64::new(0);

/// `VICTORY_BACKLOG_ROOM=<percent>` (off by default until the one test of it is repeated; 4 is the tested value): how many percent of the card's budget
/// the room line is raised by while textures wait for room.
fn backlog_percent() -> u64 {
    static N: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("VICTORY_BACKLOG_ROOM").ok().and_then(|v| v.trim().parse::<u64>().ok()).map_or(0, |p| p.min(6)))
}

/// `VICTORY_BACKLOG_AFTER_MS` (1500): how long textures wait for room before the line moves.
fn backlog_after_ms() -> u64 {
    static N: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("VICTORY_BACKLOG_AFTER_MS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(1_500))
}

/// The percent the room line is raised by now: while slots have been null for the
/// budget for 1.5 s with no room, for four seconds after the last refusal of that
/// kind. (Below the sweep's 95 % tier: the line stays at 94 % with the usual 90.)
pub(crate) fn line_bonus_percent() -> u64 {
    if crate::host::coarse_ms() < BONUS_UNTIL.load(Relaxed) { backlog_percent() } else { 0 }
}

/// `VICTORY_DEMAND_EVICT` (default on): the sweep is driven by the queue.
fn demand_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_DEMAND_EVICT").is_none_or(|v| v != "0"))
}

/// Bytes of the textures waiting for room now (0: the card had room within
/// the last two seconds, or nothing waits): what the sweep is to free at
/// once, least recently used first, besides its idle tiers.
pub(crate) fn demand_bytes() -> u64 {
    if !demand_on() {
        return 0;
    }
    let at = FULL_AT.load(Relaxed);
    if at == 0 || crate::host::coarse_ms().saturating_sub(at - 1) > 2_000 {
        return 0;
    }
    // At most what a pass can upload in a second or two: the rest is looked at again.
    crate::residency::null_bytes().1.min(768 << 20)
}

/// Counters since the start: audits, slots found shown (stale), found with
/// another view (stale), waiting, dropped, made due again.
pub(crate) static AUDIT: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

pub(crate) fn null_audit(
    heap: &crate::descriptor::Heap,
    host: &crate::descriptor::host::HostHeap,
    frame: u64,
    memory: Option<[u64; 4]>,
) {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    static LOG: AtomicU64 = AtomicU64::new(0);
    let now = crate::host::coarse_ms();
    if now < NEXT.load(Relaxed) {
        return;
    }
    // Looked at every second (the stale ones leave the count at once); said every five.
    NEXT.store(now + 1_000, Relaxed);
    let say = LOG.load(Relaxed) <= now;
    if say {
        LOG.store(now + 5_000, Relaxed);
    }
    let nulls = crate::residency::null_slots();
    if nulls.is_empty() {
        return;
    }
    let mb = |v: u64| v as f64 / (1 << 20) as f64;
    let (mut shown, mut other, mut waiting, mut dropped) = ((0u64, 0u64), (0u64, 0u64), (0u64, 0u64), (0u64, 0u64));
    let mut dropped_slots: Vec<u32> = Vec::new();
    for (slot, bytes) in nulls {
        if host.texture_slot_shown(slot) {
            shown.0 += 1;
            shown.1 += bytes;
            crate::residency::forget_null(slot);
            continue;
        }
        let is_texture = heap
            .handle(slot)
            .and_then(crate::descriptor::view)
            .is_some_and(|view| crate::descriptor::host::is_texture_view(&view));
        if !is_texture {
            other.0 += 1;
            other.1 += bytes;
            crate::residency::forget_null(slot);
            continue;
        }
        if host.texture_slot_queued(slot) {
            waiting.0 += 1;
            waiting.1 += bytes;
            continue;
        }
        dropped.0 += 1;
        dropped.1 += bytes;
        dropped_slots.push(slot);
    }
    let room = crate::resource::room(memory);
    let mut again = 0u64;
    if retry_on() && room {
        // (A bounded number a look: what the pass can serve in its budget.)
        for &slot in dropped_slots.iter().take(4096) {
            if host.texture_slot_due_now(slot, frame) {
                again += 1;
            }
        }
    }
    AUDIT[0].fetch_add(1, Relaxed);
    AUDIT[1].fetch_add(shown.0, Relaxed);
    AUDIT[2].fetch_add(other.0, Relaxed);
    AUDIT[3].fetch_add(waiting.0, Relaxed);
    AUDIT[4].fetch_add(dropped.0, Relaxed);
    AUDIT[5].fetch_add(again, Relaxed);
    if !say {
        return;
    }
    let oldest = crate::draw::HEAP_TEXTURES_QUEUE[4].load(Relaxed);
    let (left, left_bytes) = crate::residency::null_bytes();
    tracing::info!(
        "null audit (oldest due slot written {oldest} frames ago; {left} slots really null, {:.0} MB): of the slots refused for the budget since the last look,          {} ({:.0} MB) show a host copy now, {} ({:.0} MB) show another view or none (both forgotten),          {} ({:.0} MB) wait in the queue, {} ({:.0} MB) are dropped (null and in no queue); room now {room}, made due again {again} (VICTORY_NULL_RETRY, {})",
        mb(left_bytes),
        shown.0,
        mb(shown.1),
        other.0,
        mb(other.1),
        waiting.0,
        mb(waiting.1),
        dropped.0,
        mb(dropped.1),
        if retry_on() { "on" } else { "off" },
    );
}

/// One line a second ("scene second", stream3): what the title submitted that second
/// and what became of it, for the opening cutscene's arrival. Per frame: draws
/// run (complete); draws dropped because their pipeline is being made, because it
/// will never be made, for other reasons; then the pipelines still being made, those
/// made and loaded from the library this second, the texture queue (due, held for
/// memory, the oldest due slot's age in frames) and the MB uploaded.
pub(crate) fn second_line(frames: f64, fps: f64, ran: f64, ms: f64) {
    #[cfg(feature = "diagnostics")]
    {
        static BEFORE: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
        let kinds = crate::pipeline::take_drop_kinds().map(|n| n as f64 / frames);
        let lib = crate::pipeline::library_counts();
        let dl = |i: usize| {
            let was = BEFORE[i].swap(lib[i.min(1)], Relaxed);
            lib[i.min(1)].saturating_sub(was)
        };
        let (made, loaded) = (dl(0), dl(1));
        let up = {
            let now = crate::draw::HEAP_TEXTURE_BYTES.load(Relaxed);
            let was = BEFORE[2].swap(now, Relaxed);
            now.saturating_sub(was) as f64 / (1 << 20) as f64
        };
        let (null_n, null_b) = crate::residency::null_bytes();
        let (null_mb, bonus) = (null_b as f64 / (1 << 20) as f64, line_bonus_percent());
        let [queued, due, held, _, oldest] =
            [0, 1, 2, 3, 4].map(|i| crate::draw::HEAP_TEXTURES_QUEUE[i].load(Relaxed));
        tracing::info!(
            "scene second t={}s: {fps:.1} frames/s ({ms:.0} ms), per frame: {ran:.0} draws complete, {:.0} dropped (pipeline being made),              {:.0} dropped (pipeline never made), {:.0} dropped (other), {:.0} dropped (never requested: ray-tracing builder etc.); pipelines still being made {}, made this second {made},              loaded from the library {loaded}; texture queue {queued} slots, {due} due, {held} held for memory, oldest due {oldest} frames, {up:.0} MB uploaded; {null_n} slots really null ({null_mb:.0} MB), room line +{bonus}%",
            crate::host::now_ms() / 1000,
            kinds[0],
            kinds[1],
            kinds[2],
            kinds[3],
            crate::pipeline::making(),
        );
    }
    #[cfg(not(feature = "diagnostics"))]
    let _ = (frames, fps, ran, ms);
}
