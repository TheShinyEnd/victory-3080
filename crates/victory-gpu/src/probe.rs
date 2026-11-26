//! Probes for the frame profiler (`victory_core::prof`, `docs/profiling.md`):
//! the stall line's context and the reasons of dropped draws. Nothing here
//! changes what runs; every call is a few atomic operations.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// The pipelines being made and the texture bytes counted at the last frame
/// end: what a stall line compares with.
static LAST_MAKING: AtomicU64 = AtomicU64::new(0);
static LAST_TEXTURES: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Starts the profiler and tells it what to say beside a stalled frame.
pub(crate) fn init() {
    victory_core::prof::init();
    victory_core::prof::set_stall_note(stall_note);
}

/// Texture bytes counted so far: picked by index and uploaded, and hashed by
/// the submissions' refresh.
fn textures() -> [u64; 2] {
    [
        crate::draw::HEAP_TEXTURE_BYTES.load(Relaxed),
        crate::host::PREPARING[3].load(Relaxed),
    ]
}

/// Remembers the state the next stall line is measured from; called after
/// each frame's end.
pub(crate) fn frame_marked() {
    LAST_MAKING.store(crate::pipeline::making(), Relaxed);
    let now = textures();
    for i in 0..2 {
        LAST_TEXTURES[i].store(now[i], Relaxed);
    }
}

/// Pipelines made, textures and video memory over the stalled frame.
fn stall_note() -> String {
    let (before, now) = (LAST_MAKING.load(Relaxed), crate::pipeline::making());
    let tex = textures();
    // A counter cleared by its own log line restarts from zero.
    let gain = |i: usize| {
        let was = LAST_TEXTURES[i].load(Relaxed);
        if tex[i] >= was { tex[i] - was } else { tex[i] }
    };
    let video = match crate::host::host().and_then(|h| h.video_memory()) {
        Some([own_budget, own_used, shared_budget, shared_used]) => format!(
            "video memory local {}/{} MB, shared {}/{} MB",
            own_used >> 20,
            own_budget >> 20,
            shared_used >> 20,
            shared_budget >> 20
        ),
        None => "video memory unknown".to_owned(),
    };
    format!(
        "pipelines being made {before}->{now}, textures uploaded {:.1} MB (hashed {:.1} MB), {video}",
        gain(0) as f64 / 1e6,
        gain(1) as f64 / 1e6
    )
}

/// A draw, mesh dispatch or dispatch (`kind`) of `pipeline` was recorded but
/// not run: counted by pipeline for the interval's drop report with the
/// reason (shader hashes and what became of the pipeline).
#[inline]
pub(crate) fn dropped(pipeline: usize, kind: &'static str) {
    // Pointers are below 2^48: a hash of the kind rides in the top bits.
    let tag = kind.bytes().fold(7usize, |h, b| h.wrapping_mul(31).wrapping_add(usize::from(b))) & 0xFFFF;
    let key = pipeline | (tag << 48);
    victory_core::prof_tally!("drop.no_host_pipeline", key, || format!(
        "{kind}: {}",
        crate::pipeline::drop_reason(pipeline as crate::object::This)
    ));
}

/// An indirect execution was dropped for `why`, counted by `key`.
#[inline]
pub(crate) fn indirect_dropped(key: usize, why: impl FnOnce() -> String) {
    victory_core::prof_tally!("drop.indirect.why", key, why);
}
