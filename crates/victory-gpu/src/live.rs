//! Settings the in-game quality page (F8, `host/panel.rs`, diagnostics builds) changes while the title runs.
//!
//! Each knob is an override on top of what its `VICTORY_*` variable said at start: while no one has touched it the
//! variable (or its default) is read as before. Only values that the code reads again on every use are here:
//! `VICTORY_AMBIENT*` (root constants of the stand-in's pass), `VICTORY_PIPELINE_WAIT_MS` (read per frame) and
//! `VICTORY_TEXTURE_STAGE_MB` (read per texture pass). `VICTORY_NAN_SCRUB` and `VICTORY_BVH_MISS` are part of the
//! translated shaders (and of the shader cache's key), so they cannot change in a run.

// (Only the diagnostics build's panel calls the setters.)
#![cfg_attr(not(feature = "diagnostics"), allow(dead_code))]

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};

/// "No override": f32 knobs hold NaN's bits here, integer knobs this.
const NONE_F: u32 = u32::MAX;
const NONE_I: u64 = u64::MAX;

static AMBIENT: AtomicU32 = AtomicU32::new(NONE_F);
static AIR: AtomicU32 = AtomicU32::new(NONE_F);
static MOST: AtomicU32 = AtomicU32::new(NONE_F);
static WAIT_MS: AtomicU64 = AtomicU64::new(NONE_I);
static STAGE_MB: AtomicU64 = AtomicU64::new(NONE_I);

fn get_f(a: &AtomicU32) -> Option<f32> {
    match a.load(Relaxed) {
        NONE_F => None,
        bits => Some(f32::from_bits(bits)),
    }
}

fn get_i(a: &AtomicU64) -> Option<u64> {
    match a.load(Relaxed) {
        NONE_I => None,
        v => Some(v),
    }
}

/// The share of the light estimate (`VICTORY_AMBIENT=<number>`); 0 is off.
pub(crate) fn ambient() -> Option<f32> {
    get_f(&AMBIENT)
}
pub(crate) fn set_ambient(v: f32) {
    AMBIENT.store(v.max(0.0).to_bits(), Relaxed);
}

/// The air's bound (`VICTORY_AMBIENT_AIR`); 0 is the title's own.
pub(crate) fn air() -> Option<f32> {
    get_f(&AIR)
}
pub(crate) fn set_air(v: f32) {
    AIR.store(v.max(0.0).to_bits(), Relaxed);
}

/// What of the title's own light is kept (`VICTORY_AMBIENT_MOST`), as a multiple of the estimate.
pub(crate) fn most() -> Option<f32> {
    get_f(&MOST)
}
pub(crate) fn set_most(v: f32) {
    MOST.store(v.max(0.1).to_bits(), Relaxed);
}

/// Milliseconds a frame may wait for new pipelines (`VICTORY_PIPELINE_WAIT_MS`).
pub(crate) fn wait_ms() -> Option<u64> {
    get_i(&WAIT_MS)
}
pub(crate) fn set_wait_ms(v: u64) {
    WAIT_MS.store(v.min(120_000), Relaxed);
}

/// Megabytes of textures staged to the card per frame (`VICTORY_TEXTURE_STAGE_MB`); `Some(0)` is the automatic budget.
pub(crate) fn stage_mb() -> Option<u64> {
    get_i(&STAGE_MB)
}
pub(crate) fn set_stage_mb(v: u64) {
    STAGE_MB.store(v.min(4096), Relaxed);
}


/// The rows of the in-game page and the key that is down.
#[cfg(windows)]
#[derive(Default)]
pub(crate) struct Menu {
    pub(crate) sel: usize,
}

#[cfg(windows)]
const ROWS: usize = 7;
#[cfg(windows)]
const STAGES: [u64; 8] = [0, 32, 64, 128, 192, 256, 384, 512];

#[cfg(windows)]
fn env_f(name: &str) -> Option<f32> {
    std::env::var(name).ok()?.trim().parse().ok()
}

#[cfg(windows)]
fn stage_now() -> u64 {
    stage_mb().unwrap_or_else(|| std::env::var("VICTORY_TEXTURE_STAGE_MB").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0))
}

#[cfg(windows)]
fn wait_now() -> u64 {
    crate::pipeline::wait_budget_ms()
}

#[cfg(windows)]
impl Menu {
    pub(crate) fn up(&mut self) {
        self.sel = (self.sel + ROWS - 1) % ROWS;
    }

    pub(crate) fn down(&mut self) {
        self.sel = (self.sel + 1) % ROWS;
    }

    /// `dir` is -1 or 1. The last two rows only show what the shaders were made with.
    pub(crate) fn change(&mut self, dir: i32) {
        let d = dir as f32;
        match self.sel {
            0 => set_ambient((((crate::ambient::share_now() + 0.1 * d) * 10.0).round() / 10.0).clamp(0.0, 2.0)),
            1 => set_air(((crate::ambient::air() + 0.25 * d).clamp(0.0, 4.0) * 4.0).round() / 4.0),
            2 => set_most(((crate::ambient::most() + 0.5 * d).clamp(0.5, 8.0) * 2.0).round() / 2.0),
            3 => set_wait_ms((wait_now() as i64 + 500 * i64::from(dir)).clamp(0, 30_000) as u64),
            4 => {
                let at = STAGES.iter().position(|&m| m >= stage_now()).unwrap_or(STAGES.len() - 1);
                let to = (at as i32 + dir).clamp(0, STAGES.len() as i32 - 1) as usize;
                set_stage_mb(STAGES[to]);
            }
            _ => {}
        }
    }

    /// The page's lines, 40 characters at most.
    pub(crate) fn lines(&self) -> Vec<String> {
        let share = crate::ambient::share_now();
        let stage = stage_now();
        let on = |name: &str| if env_f(name) == Some(1.0) { "ON" } else { "OFF" };
        let rows = [
            ("AMBIENT SHARE", if share > 0.0 { format!("{share:.1}") } else { "OFF".to_owned() }),
            ("AIR LIMIT", match crate::ambient::air() { a if a > 0.0 => format!("{a:.2}"), _ => "TITLE".to_owned() }),
            ("KEEP LIMIT", format!("{:.1}", crate::ambient::most())),
            ("PIPELINE WAIT MS", wait_now().to_string()),
            ("TEXTURE MB/FRAME", if stage == 0 { "AUTO".to_owned() } else { stage.to_string() }),
            ("NAN SCRUB RESTART", on("VICTORY_NAN_SCRUB").to_owned()),
            ("RAY MISS RESTART", on("VICTORY_BVH_MISS").to_owned()),
        ];
        let mut lines = vec!["QUALITY  F8 CLOSE  ARROWS".to_owned()];
        for (i, (label, value)) in rows.iter().enumerate() {
            lines.push(format!("{}{label:<18}{value}", if i == self.sel { ">" } else { " " }));
        }
        lines.push("CHANGES LAST FOR THIS RUN".to_owned());
        lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn untouched_knobs_have_no_override_and_set_ones_do() {
        assert_eq!(get_f(&AtomicU32::new(NONE_F)), None);
        assert_eq!(get_i(&AtomicU64::new(NONE_I)), None);
        let a = AtomicU32::new(NONE_F);
        a.store(1.5f32.to_bits(), Relaxed);
        assert_eq!(get_f(&a), Some(1.5));
    }
}
