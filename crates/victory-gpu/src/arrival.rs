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
    FULL_AT.store(crate::host::coarse_ms() + 1, Relaxed);
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
