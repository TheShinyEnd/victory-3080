//! `quality.toml`: one file for the runtime's picture and speed options.
//!
//! The runtime is steered by `VICTORY_*` environment variables. This file names the ones that change what the
//! picture looks like or what it costs, gives them readable keys and four presets, and is turned into those
//! variables at start ([`apply`]). **A variable that is already set wins**, so every launcher script keeps working
//! and a single run can still override the file (`VICTORY_RESIDENCY=legacy victory run ...`).
//!
//! ```toml
//! preset = "balanced"        # performance | balanced | quality | ultra
//! [options]                  # anything here beats the preset
//! display_mode = "2560x1440"
//! ambient = 1.0
//! ```
//!
//! An option's value `"default"` (or `""`) leaves its variable unset. See `docs/quality.md`-style text in
//! `victory quality --list`, and `tools/VIctory-Settings.ps1` for a window that edits the file.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

/// The presets, in order of cost.
pub const PRESETS: [&str; 4] = ["performance", "balanced", "quality", "ultra"];

/// Whether this build contains the game-settings hooks (`victory-loader` `hooks.rs`: render scale, the title's
/// own upscaler). v7 excludes them, so the two options below are listed but never turned into variables.
pub const HOOKS_BUILT: bool = false;

#[derive(Clone, Copy, Debug)]
pub enum Kind {
    /// `true`/`false`; the variable gets `on` or `off` ("" leaves it unset).
    Bool { on: &'static str, off: &'static str },
    /// A number between `min` and `max`.
    Num { min: f64, max: f64 },
    /// One of these words; each maps to a variable value (word, value) ("" leaves it unset).
    Choice(&'static [(&'static str, &'static str)]),
    /// Any text, passed on as it is.
    Text,
}

#[derive(Clone, Copy, Debug)]
pub struct Opt {
    pub key: &'static str,
    pub env: &'static str,
    pub group: &'static str,
    pub kind: Kind,
    /// Value for performance, balanced, quality, ultra ("" leaves the variable unset).
    pub presets: [&'static str; 4],
    /// Changes only take effect at the next start (all of them but the live ones, `live.rs`).
    pub live: bool,
    /// Needs the game-settings hooks (class b): listed, never applied by a build without them.
    pub needs_hooks: bool,
    /// Another variable set with it when the value is not empty: (name, value).
    pub also: Option<(&'static str, &'static str)>,
    pub help: &'static str,
    /// What it costs or buys in frame rate.
    pub cost: &'static str,
}

const ON1: Kind = Kind::Bool { on: "1", off: "" };
const ON1_OFF0: Kind = Kind::Bool { on: "1", off: "0" };

macro_rules! opt {
    ($key:literal, $env:literal, $group:literal, $kind:expr, [$a:literal, $b:literal, $c:literal, $d:literal], $help:literal, $cost:literal) => {
        Opt { key: $key, env: $env, group: $group, kind: $kind, presets: [$a, $b, $c, $d], live: false, needs_hooks: false, also: None, help: $help, cost: $cost }
    };
}

pub const GROUPS: [&str; 6] = ["Resolution & profile", "Display & HDR", "Lighting & ray tracing", "Textures & memory", "Performance", "Advanced / diagnostics"];

/// Every option. The order is the order of the menu.
pub static OPTIONS: &[Opt] = &[
    // --- Resolution & profile
    Opt {
        also: Some(("VICTORY_COMPOSE", "1")),
        ..opt!("console", "VICTORY_FORCE_CONSOLE", "Resolution & profile",
            Kind::Choice(&[("series-x", ""), ("series-s", "series-s")]),
            ["series-s", "series-x", "series-x", "series-x"],
            "The game's own console profile. Series S is its low preset (smaller textures, lower detail, 1080p); Series X its high one (2160p).",
            "Series S is about 2-3x faster than Series X")
    },
    opt!("display_mode", "VICTORY_DISPLAY_MODE", "Resolution & profile", Kind::Text,
        ["1920x1080", "2560x1440", "2560x1440", "3840x2160"],
        "The screen size the game is told the console is plugged into (WxH, 640x360 to 7680x4320). Measured on the Series X profile: the game renders its scene at 2560x1440 whatever this says; the setting sets the size of its final picture (and drops the 3840x2160 colour, depth and upscale targets when it is smaller: about 0.5 GB of card memory). On a 1440p monitor 2560x1440 is native.",
        "Series X: only the final plane and about 0.5 GB of memory change (the scene stays 1440p); the Series S console option lowers the scene's size"),
    opt!("adapter", "VICTORY_ADAPTER", "Resolution & profile", Kind::Text, ["", "", "", ""],
        "Which graphics card to use on a PC with more than one: part of its name (5060) or #index. Empty: Windows' default card.",
        "Pick the strongest card"),
    opt!("window_max", "VICTORY_WINDOW_MAX", "Resolution & profile", Kind::Text, ["", "", "", ""],
        "Shrinks the game's window to at most WxH (keeps aspect, centred). The picture is stretched: softer, same frame rate. Not a render-resolution setting.",
        "None"),
    Opt {
        needs_hooks: true,
        ..opt!("render_scale", "VICTORY_RENDER_SCALE", "Resolution & profile", Kind::Num { min: 0.2, max: 1.0 }, ["0.5", "0.5", "0.75", "1.0"],
            "Game-side: the fraction of the output height the game renders at before its own upscaler (0.2 to 1.0; the hook cannot go above native).",
            "Lower is much faster and softer")
    },
    Opt {
        needs_hooks: true,
        ..opt!("own_upscaler", "VICTORY_OWN_UPSCALER", "Resolution & profile", ON1, ["1", "1", "1", ""],
            "Game-side: forces the game's own temporal upscaler on (the console profile Series S needs it at any scale below 1).",
            "Small cost, hides the lower render scale")
    },
    // --- Display & HDR
    opt!("monitor", "VICTORY_MONITOR", "Display & HDR", Kind::Text, ["", "", "", ""],
        "Which screen the game's window opens on: part of the monitor's name (Alienware), DISPLAY1, #index (0 is the first output) or primary. Empty: the primary screen. The log lists every screen at start.",
        "None"),
    opt!("fullscreen", "VICTORY_FULLSCREEN", "Display & HDR", ON1, ["", "", "", ""],
        "A borderless window the size of the chosen screen instead of a small window (the picture is stretched to it).",
        "A bigger window costs a little GPU time"),
    opt!("hdr", "VICTORY_HDR", "Display & HDR",
        Kind::Choice(&[("auto", ""), ("on", "on"), ("off", "off"), ("stretch", "stretch")]), ["auto", "auto", "auto", "auto"],
        "auto: HDR output when Windows HDR is on for the chosen screen (turn on Use HDR for it in Windows Settings; VIctory never changes that). on: always tell the game HDR is available and present PQ/BT.2020. off: SDR. stretch: APPROXIMATION, the game stays SDR and its picture is expanded to HDR.",
        "None"),
    opt!("hdr_paper_nits", "VICTORY_HDR_PAPER_NITS", "Display & HDR", Kind::Num { min: 80.0, max: 500.0 }, ["", "", "", ""],
        "HDR only: the brightness in nits of SDR white (the interface, menus and the numbers panel; default 200). The game's own HDR picture is not changed by it.",
        "None"),
    // --- Lighting & ray tracing
    opt!("ray_tracing", "VICTORY_RT", "Lighting & ray tracing",
        Kind::Choice(&[("miss", ""), ("off", "off"), ("cpu", "cpu")]), ["miss", "miss", "miss", "miss"],
        "miss: rays end at the root (no traced shadows or bounced light; the stand-in light below fills in). off: as miss, and the ~1,034 ray-tracing resolve and traced-shadow shaders are skipped (fastest). cpu: the title's ray-tracing structures are built on the CPU (XboxBvhBuilder replaced) and the rays are walked for real: traced shadows, bounced light, reflections; EXPERIMENTAL.",
        "off is fastest; cpu costs CPU time for every build and GPU time for every ray"),
    opt!("ray_miss", "VICTORY_BVH_MISS", "Lighting & ray tracing", ON1, ["1", "1", "1", "1"],
        "Rays end at the root (miss) while the structure builder is off. Off: the title's ray walks run and may hang on stale structures.",
        "On is faster and avoids hangs; off needs the structure builder (not available on NVIDIA)"),
    opt!("ray_skip", "VICTORY_SKIP_PIPELINES", "Lighting & ray tracing",
        Kind::Choice(&[("none", "none"), ("list", "list")]), ["list", "none", "none", "none"],
        "list: skip the ~1,034 ray-tracing resolve and traced-shadow shaders of scripts/rt_off_hashes.txt (diagnostics build only). none: run them.",
        "list saves GPU time; the skipped passes give less bounced light"),
    Opt { live: true, ..opt!("ambient", "VICTORY_AMBIENT", "Lighting & ray tracing", Kind::Text, ["0.8", "0.8", "1.0", "1.0"],
        "Stand-in for the indirect light the rays would give: this share of an estimate worked out from the frame (0 or off: the title's own, black indoors). Also constant:<level>.",
        "Almost free (one small compute pass)") },
    Opt { live: true, ..opt!("ambient_air", "VICTORY_AMBIENT_AIR", "Lighting & ray tracing", Kind::Num { min: 0.0, max: 8.0 }, ["1.5", "1.5", "1.5", "1.5"],
        "Bound for the light the air scatters, as a multiple of the light around (0: the title's own). Lower removes brown fog veils indoors.",
        "Free") },
    Opt { live: true, ..opt!("ambient_most", "VICTORY_AMBIENT_MOST", "Lighting & ray tracing", Kind::Num { min: 0.1, max: 16.0 }, ["2", "2", "2", "2"],
        "How many times the estimate the title's own light may keep (4 when unset). Lower stops rooms by a window from glowing.",
        "Free") },
    opt!("ambient_shade", "VICTORY_AMBIENT_SHADE", "Lighting & ray tracing", Kind::Text, ["", "", "", ""],
        "Crease shading of the stand-in: <strength>,<radius in metres> (default 1,0.6).", "Free"),
    opt!("nan_scrub", "VICTORY_NAN_SCRUB", "Lighting & ray tracing", ON1, ["1", "1", "1", "1"],
        "A NaN a shader writes is made 0 before blur and upscaler spread it: removes black blobs. Baked into translated shaders.",
        "Tiny cost"),
    // --- Textures & memory
    opt!("residency", "VICTORY_RESIDENCY", "Textures & memory", Kind::Choice(&[("auto", ""), ("legacy", "legacy")]), ["auto", "auto", "auto", "auto"],
        "Texture residency policy: auto = the new budgeted policy (floor, reserve, eviction); legacy = the base's rule.",
        "auto keeps more detail resident within the card's budget"),
    opt!("texture_budget_mb", "VICTORY_TEXTURE_BUDGET_MB", "Textures & memory", Kind::Num { min: 256.0, max: 65536.0 }, ["", "", "", ""],
        "Hard cap on host texture copies in MB (legacy residency only). Empty: derived from the card's budget.",
        "Higher is sharper but risks spilling card memory into PC RAM (stutter)"),
    opt!("texture_cap_mb", "VICTORY_TEXTURE_CAP_MB", "Textures & memory", Kind::Num { min: 0.0, max: 65536.0 }, ["", "", "128", "0"],
        "Top mips of a host copy bigger than this are dropped (0: none; empty: the build's default: 8-64 MB by card at 7c0cdf6, off at 28d8475 because a cap banded the picture). Higher keeps more close-up detail.",
        "Higher costs video memory (and can spill into PC RAM on small cards); lower fits small cards"),
    opt!("residency_reserve_mb", "VICTORY_RESIDENCY_RESERVE_MB", "Textures & memory", Kind::Num { min: 0.0, max: 65536.0 }, ["", "", "", ""],
        "Memory the residency policy keeps free for everything else.", "Higher is safer, textures get less room"),
    opt!("residency_floor_mb", "VICTORY_RESIDENCY_FLOOR_MB", "Textures & memory", Kind::Num { min: 0.0, max: 65536.0 }, ["", "", "", ""],
        "Texture bytes always admitted whatever the memory use.", "Higher keeps more detail under pressure"),
    opt!("residency_stage_mb", "VICTORY_RESIDENCY_STAGE_MB", "Textures & memory", Kind::Num { min: 16.0, max: 1024.0 }, ["", "", "", ""],
        "Texture MB staged to the card per frame under the new policy (automatic: 48-192 MB by card).", "Higher fills textures faster, longer frames"),
    Opt { live: true, ..opt!("texture_stage_mb", "VICTORY_TEXTURE_STAGE_MB", "Textures & memory", Kind::Num { min: 0.0, max: 4096.0 }, ["", "", "", ""],
        "Texture MB uploaded per frame (empty: automatic). Higher drains the first minutes of black surfaces sooner.", "Higher: faster fill, hitchier frames") },
    opt!("texture_no_wait", "VICTORY_TEXTURE_NO_WAIT", "Textures & memory", Kind::Choice(&[("wait", "0"), ("no-wait", "1"), ("no-wait-all", "2")]),
        ["no-wait", "wait", "wait", "wait"],
        "Whether texture refreshes wait for the title's memory to hold still. wait: correct skin/materials; no-wait: faster, can show rainbow skin.",
        "no-wait saves about 7 fps"),
    opt!("upload_ahead", "VICTORY_UPLOAD_AHEAD", "Textures & memory", ON1_OFF0, ["1", "1", "1", "1"],
        "Upload textures ahead of the draw lists that use them (less pop-in).", "Slightly less hitching"),
    // --- Performance
    opt!("relay", "VICTORY_RELAY", "Performance", Kind::Choice(&[("0", "0"), ("1", "1"), ("2", "2")]), ["2", "2", "1", "1"],
        "Relay stage for the title's copies. 1: correct bodies (no stretched or jumping characters); 2: faster.", "Stage 1 costs about 7 fps with card steps 0"),
    opt!("card_steps", "VICTORY_CARD_STEPS", "Performance", Kind::Choice(&[("0", "0"), ("1", "1"), ("2", "2")]), ["2", "2", "0", "0"],
        "Card-side steps of the relay. 0: correct bodies; 2: fastest.", "0 costs the submitting thread about 30 ms a frame"),
    Opt { live: true, ..opt!("pipeline_wait_ms", "VICTORY_PIPELINE_WAIT_MS", "Performance", Kind::Num { min: 0.0, max: 120000.0 }, ["10000", "10000", "10000", "10000"],
        "Milliseconds a frame may wait for new shader pipelines to compile instead of skipping their draws (default 500). Higher: fewer missing objects, longer first-time hitches.",
        "Only costs time the first time a place is seen") },
    opt!("dynamic_slots", "VICTORY_DYNAMIC_SLOTS", "Performance", Kind::Num { min: 1024.0, max: 1000000.0 }, ["524288", "524288", "524288", "524288"],
        "Descriptor slots for dynamic views. Too few drops draws (missing objects) on cards capped at 1,000,000.", "None"),
    opt!("card_buffers_mb", "VICTORY_CARD_BUFFERS_MB", "Performance", Kind::Num { min: 0.0, max: 16384.0 }, ["2048", "2048", "2048", "2048"],
        "Video memory for card-side copies of the title's buffers (memory compares on the card, not the CPU: about 2x fps on Series S). Measured on an RTX 3080 with the smallest size below at 64: 2048 holds all copies (about 1.9 GB made).",
        "Larger is faster until memory runs out"),
    opt!("card_buffers_min_kb", "VICTORY_CARD_BUFFERS_MIN_KB", "Performance", Kind::Num { min: 1.0, max: 1048576.0 }, ["64", "64", "64", "64"],
        "The smallest title buffer (KiB) that gets a copy in the card's memory; smaller ones are read by the shaders across the PCIe bus. Measured (RTX 3080, 1440p, 5,000 draws): 64 instead of 1024 cut the GPU's command time from 82 to 48 ms a frame and the frame from 135 to 110 ms, for about 540 MB of card memory. 4, 16 and 64 measured the same. Overrides the MB setting of the same name.",
        "Costs about 0.5 GB of card memory; saves about 30 ms of GPU time a frame"),
    opt!("zero_copy", "VICTORY_UMA", "Performance", ON1, ["", "", "", ""],
        "Zero-copy memory on integrated or unified-memory GPUs.", "Faster only on shared-memory hardware"),
    opt!("texture_room_percent", "VICTORY_TEXTURE_ROOM_PERCENT", "Textures & memory", Kind::Num { min: 50.0, max: 97.0 }, ["", "", "", ""],
        "Legacy residency only: the share of card memory textures may fill (default 90).", "Higher keeps more textures, risks spilling into PC RAM"),
    opt!("texture_keep", "VICTORY_TEXTURE_KEEP", "Textures & memory", Kind::Choice(&[("keep", ""), ("release-early", "0")]), ["keep", "keep", "keep", "keep"],
        "keep: idle texture copies stay up to 10 minutes while memory is free; release-early: dropped after 5 s.", "release-early saves memory, causes re-uploads"),
    opt!("window_idle_secs", "VICTORY_WINDOW_IDLE_SECS", "Textures & memory", Kind::Num { min: 0.0, max: 3600.0 }, ["30", "30", "30", "30"],
        "Seconds a copy of title memory (a \"window\") that nothing holds may go unused before it is freed (0: only the free-RAM rule, five minutes with plenty of RAM). Frees PC memory only (these copies are shared memory, not the card's); a window the game comes back to is copied again.",
        "Frees about 1 GB of PC memory in a long run; a re-copy of 4 MB per window the game returns to"),
    opt!("levels_on_demand", "VICTORY_LEVELS_ON_DEMAND", "Textures & memory", ON1, ["", "", "", ""],
        "Copies of textures are made level by level on demand (needs tiled-resources tier 2). Saves memory.", "Saves memory, small CPU cost"),
    opt!("map_threads", "VICTORY_MAP_THREADS", "Performance", Kind::Num { min: 1.0, max: 32.0 }, ["", "", "", ""],
        "Threads that map the title's memory (empty: 8).", "More can help streaming, costs CPU"),
    opt!("pipeline_workers", "VICTORY_PIPELINE_WORKERS", "Performance", Kind::Num { min: 1.0, max: 64.0 }, ["", "", "", ""],
        "Threads that compile shader pipelines (empty: two thirds of the cores).", "More shortens first-time hitches, takes CPU from the game"),
    opt!("rt_resolve", "VICTORY_RT_RESOLVE", "Lighting & ray tracing", ON1, ["", "", "", ""],
        "Keep the ~1,000 ray-tracing resolve shaders instead of leaving them out while the structure builder is off.", "Costs GPU time; the light they give is wrong without structures"),
    opt!("bvh_builder", "VICTORY_BVH_BUILDER", "Lighting & ray tracing", ON1, ["", "", "", ""],
        "EXPERIMENTAL: builds the ray-tracing structures. Cannot run on NVIDIA cards (64-lane programs refused).", "Large cost; may hang"),
    opt!("f16_nearest", "VICTORY_F16_NEAREST", "Advanced / diagnostics", ON1, ["", "", "", ""],
        "Exact half-float rounding in translated shaders. Baked into shaders: the first run re-translates everything.", "Extra instructions"),
    // --- Advanced / diagnostics
    opt!("panel", "VICTORY_PANEL", "Advanced / diagnostics", ON1, ["", "", "", ""],
        "Shows the numbers panel (F3 toggles). F8 opens the quality page whether or not this is on.", "About 50 microseconds a frame"),
    opt!("profile", "VICTORY_PROFILE", "Advanced / diagnostics", ON1, ["", "", "", ""],
        "Scoped timers and a stall report at the end of the run.", "A few percent"),
    opt!("gpu_timing", "VICTORY_GPU_TIMING", "Advanced / diagnostics", ON1, ["", "", "", ""],
        "GPU timestamps per pass for the numbers panel and profile.", "A few percent"),
];

pub fn find(key: &str) -> Option<&'static Opt> {
    OPTIONS.iter().find(|o| o.key == key)
}

/// A file's content.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct File {
    pub preset: Option<String>,
    #[serde(default)]
    pub options: BTreeMap<String, toml::Value>,
}

impl File {
    pub fn parse(text: &str) -> Result<File> {
        toml::from_str(text).context("parsing quality.toml")
    }

    /// The preset's index (default `balanced`).
    pub fn preset_index(&self) -> Result<usize> {
        let name = self.preset.as_deref().unwrap_or("balanced").trim().to_ascii_lowercase();
        PRESETS.iter().position(|p| *p == name).with_context(|| format!("unknown preset {name:?}; use one of {}", PRESETS.join(", ")))
    }
}

pub fn value_text(v: &toml::Value) -> String {
    match v {
        toml::Value::String(s) => s.trim().to_owned(),
        toml::Value::Boolean(b) => b.to_string(),
        toml::Value::Integer(i) => i.to_string(),
        toml::Value::Float(f) => format!("{f}"),
        other => other.to_string(),
    }
}

/// An option's value as the variable's text ("" = leave it unset), or why it is not acceptable.
pub fn variable_value(o: &Opt, text: &str) -> Result<String> {
    let t = text.trim();
    if t.is_empty() || t.eq_ignore_ascii_case("default") {
        return Ok(String::new());
    }
    Ok(match o.kind {
        Kind::Bool { on, off } => match t.to_ascii_lowercase().as_str() {
            "true" | "1" | "on" | "yes" => on.to_owned(),
            "false" | "0" | "off" | "no" => off.to_owned(),
            _ => bail!("{}: expected true or false, got {t:?}", o.key),
        },
        Kind::Num { min, max } => {
            let n: f64 = t.parse().with_context(|| format!("{}: {t:?} is not a number", o.key))?;
            if !n.is_finite() || n < min || n > max {
                bail!("{}: {t} is outside {min}..{max}", o.key);
            }
            t.to_owned()
        }
        Kind::Choice(words) => match words.iter().find(|(w, _)| w.eq_ignore_ascii_case(t)) {
            Some((_, v)) => (*v).to_owned(),
            None => bail!("{}: expected one of {}, got {t:?}", o.key, words.iter().map(|(w, _)| *w).collect::<Vec<_>>().join(", ")),
        },
        Kind::Text => t.to_owned(),
    })
}

/// What the file asks for: (variable, value) pairs, options with an empty value left out. Errors name every bad key.
pub fn resolve(file: &File) -> Result<Vec<(String, String)>> {
    let index = file.preset_index()?;
    let mut errors = Vec::new();
    for key in file.options.keys() {
        if find(key).is_none() {
            errors.push(format!("unknown option {key:?}"));
        }
    }
    let mut out = Vec::new();
    for o in OPTIONS {
        if o.needs_hooks && !HOOKS_BUILT {
            continue;
        }
        let text = file.options.get(o.key).map_or_else(|| o.presets[index].to_owned(), value_text);
        match variable_value(o, &text) {
            Ok(v) if v.is_empty() => {}
            Ok(v) => {
                out.push((o.env.to_owned(), v));
                if let Some((n, v)) = o.also {
                    out.push((n.to_owned(), v.to_owned()));
                }
            }
            Err(e) => errors.push(e.to_string()),
        }
    }
    if !errors.is_empty() {
        bail!("{}", errors.join("; "));
    }
    ray_tracing_links(&mut out);
    Ok(out)
}

/// `ray_tracing` decides two other variables: `cpu` walks the rays for real (no miss restart, nothing skipped), `off`
/// misses them and skips the passes that trace.
fn ray_tracing_links(out: &mut Vec<(String, String)>) {
    let Some(mode) = out.iter().find(|(n, _)| n == "VICTORY_RT").map(|(_, v)| v.clone()) else { return };
    let (miss, skip) = match mode.as_str() {
        "cpu" => ("0", "none"),
        "off" => ("1", "list"),
        _ => return,
    };
    out.retain(|(n, _)| n != "VICTORY_BVH_MISS" && n != "VICTORY_SKIP_PIPELINES");
    out.push(("VICTORY_BVH_MISS".to_owned(), miss.to_owned()));
    out.push(("VICTORY_SKIP_PIPELINES".to_owned(), skip.to_owned()));
}

/// Expands the `list` word of `VICTORY_SKIP_PIPELINES` to the hashes of `rt_off_hashes.txt` (found through
/// `VICTORY_RT_LIST`, next to the executable's `..\..\scripts`, or `scripts\` here); unknown: none.
fn expand_skip(value: &str) -> String {
    if value.trim() != "list" {
        return value.to_owned();
    }
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(p) = std::env::var_os("VICTORY_RT_LIST") {
        candidates.push(p.into());
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(root) = exe.parent().and_then(Path::parent).and_then(Path::parent) {
            candidates.push(root.join("scripts").join("rt_off_hashes.txt"));
        }
    }
    candidates.push(PathBuf::from("scripts").join("rt_off_hashes.txt"));
    for c in candidates {
        if let Ok(text) = std::fs::read_to_string(&c) {
            let joined: Vec<&str> = text.split([',', '\n', '\r', ' ']).filter(|s| !s.is_empty()).collect();
            if !joined.is_empty() {
                return joined.join(",");
            }
        }
    }
    eprintln!("quality: rt_off_hashes.txt not found (set VICTORY_RT_LIST); the ray-tracing skip list is not used");
    "none".to_owned()
}

/// Where the file is looked for: `VICTORY_QUALITY` (a path), else `quality.toml` in the data directory's parent
/// (`%LOCALAPPDATA%\victory\quality.toml`, next to `data` and `cache`), else `quality.toml` in the working
/// directory beside `victory.toml`.
pub fn locate(env: impl Fn(&str) -> Option<std::ffi::OsString>) -> Option<PathBuf> {
    if let Some(p) = env("VICTORY_QUALITY").filter(|v| !v.is_empty()) {
        return Some(PathBuf::from(p));
    }
    let mut candidates = Vec::new();
    if let Some(d) = env("VICTORY_DATA_DIR").filter(|v| !v.is_empty()) {
        let d = PathBuf::from(d);
        candidates.push(d.parent().unwrap_or(&d).join("quality.toml"));
    } else if let Some(l) = env("LOCALAPPDATA").filter(|v| !v.is_empty()) {
        candidates.push(PathBuf::from(l).join("victory").join("quality.toml"));
    }
    candidates.push(PathBuf::from("quality.toml"));
    candidates.into_iter().find(|p| p.is_file())
}

/// The default place a new file is written.
pub fn default_path(env: impl Fn(&str) -> Option<std::ffi::OsString>) -> PathBuf {
    if let Some(p) = env("VICTORY_QUALITY").filter(|v| !v.is_empty()) {
        return PathBuf::from(p);
    }
    if let Some(d) = env("VICTORY_DATA_DIR").filter(|v| !v.is_empty()) {
        let d = PathBuf::from(d);
        return d.parent().unwrap_or(&d).join("quality.toml");
    }
    let base = env("LOCALAPPDATA").filter(|v| !v.is_empty()).map_or_else(|| PathBuf::from("."), PathBuf::from);
    base.join("victory").join("quality.toml")
}

/// Sets the variables the file asks for that are not set yet. Called first thing in `main`, before any other
/// thread exists. A broken file is reported and ignored: a typo must not stop the game from starting.
pub fn apply() {
    if let Some(v) = std::env::var_os("VICTORY_SKIP_PIPELINES") {
        if v == "list" {
            // SAFETY: single-threaded start-up.
            unsafe { std::env::set_var("VICTORY_SKIP_PIPELINES", expand_skip("list")) };
        }
    }
    if std::env::var_os("VICTORY_QUALITY_OFF").is_some_and(|v| v != "0") {
        return;
    }
    let Some(path) = locate(|k| std::env::var_os(k)) else { return };
    let result = std::fs::read_to_string(&path)
        .with_context(|| format!("reading {}", path.display()))
        .and_then(|t| File::parse(&t))
        .and_then(|f| resolve(&f));
    match result {
        Ok(vars) => {
            let mut set = 0;
            for (name, value) in vars {
                if std::env::var_os(&name).is_none() {
                    let value = if name == "VICTORY_SKIP_PIPELINES" { expand_skip(&value) } else { value };
                    // SAFETY: single-threaded start-up.
                    unsafe { std::env::set_var(&name, value) };
                    set += 1;
                }
            }
            tracing::info!(file = %path.display(), set, "quality settings applied (variables already set win)");
        }
        Err(e) => eprintln!("quality: {} ignored: {e:#}", path.display()),
    }
}

/// The file's text for a preset with `overrides` (key, value) as `[options]`.
pub fn render(preset: &str, overrides: &BTreeMap<String, String>) -> String {
    let mut s = String::from("# VIctory quality settings (victory quality --list). Environment variables win over this file.\n");
    s.push_str(&format!("preset = \"{preset}\"\n\n[options]\n"));
    for (k, v) in overrides {
        let quoted = v.parse::<f64>().is_ok() || v == "true" || v == "false";
        if quoted {
            s.push_str(&format!("{k} = {v}\n"));
        } else {
            s.push_str(&format!("{k} = \"{v}\"\n"));
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_preset_value_is_acceptable() {
        for o in OPTIONS {
            for (i, p) in o.presets.iter().enumerate() {
                variable_value(o, p).unwrap_or_else(|e| panic!("{} preset {}: {e}", o.key, PRESETS[i]));
            }
        }
    }

    #[test]
    fn keys_and_variables_are_unique() {
        for (i, a) in OPTIONS.iter().enumerate() {
            for b in &OPTIONS[i + 1..] {
                assert_ne!(a.key, b.key);
                assert_ne!(a.env, b.env);
            }
            assert!(GROUPS.contains(&a.group), "{}", a.key);
        }
    }

    #[test]
    fn balanced_is_the_launchers_set() {
        let f = File::parse("preset = \"balanced\"").unwrap();
        let v: BTreeMap<_, _> = resolve(&f).unwrap().into_iter().collect();
        assert_eq!(v["VICTORY_AMBIENT"], "0.8");
        assert_eq!(v["VICTORY_NAN_SCRUB"], "1");
        assert_eq!(v["VICTORY_RELAY"], "2");
        assert_eq!(v["VICTORY_TEXTURE_NO_WAIT"], "0");
        assert!(!v.contains_key("VICTORY_FORCE_CONSOLE"));
        assert!(!v.contains_key("VICTORY_RENDER_SCALE"));
    }

    #[test]
    fn overrides_beat_the_preset_and_bad_ones_are_named() {
        let f = File::parse("preset = \"performance\"\n[options]\nconsole = \"series-x\"\nambient = 1.25").unwrap();
        let v: BTreeMap<_, _> = resolve(&f).unwrap().into_iter().collect();
        assert!(!v.contains_key("VICTORY_FORCE_CONSOLE"));
        assert_eq!(v["VICTORY_AMBIENT"], "1.25");
        let bad = File::parse("[options]\nambient_air = 99\nnope = 1").unwrap();
        let e = resolve(&bad).unwrap_err().to_string();
        assert!(e.contains("ambient_air") && e.contains("nope"), "{e}");
        assert!(File::parse("preset = \"x\"").unwrap().preset_index().is_err());
    }

    #[test]
    fn ray_tracing_modes_set_the_miss_restart_and_the_skip_list() {
        let v = |text: &str| -> BTreeMap<_, _> { resolve(&File::parse(text).unwrap()).unwrap().into_iter().collect() };
        let cpu = v("preset = \"performance\"
[options]
ray_tracing = \"cpu\"");
        assert_eq!(cpu["VICTORY_RT"], "cpu");
        assert_eq!(cpu["VICTORY_BVH_MISS"], "0");
        assert_eq!(cpu["VICTORY_SKIP_PIPELINES"], "none");
        let off = v("preset = \"quality\"
[options]
ray_tracing = \"off\"");
        assert_eq!((off["VICTORY_BVH_MISS"].as_str(), off["VICTORY_SKIP_PIPELINES"].as_str()), ("1", "list"));
        let miss = v("preset = \"quality\"");
        assert!(!miss.contains_key("VICTORY_RT"));
        assert_eq!(miss["VICTORY_BVH_MISS"], "1");
    }

    #[test]
    fn series_s_turns_compose_on() {
        let f = File::parse("preset = \"performance\"").unwrap();
        let v: BTreeMap<_, _> = resolve(&f).unwrap().into_iter().collect();
        assert_eq!(v["VICTORY_FORCE_CONSOLE"], "series-s");
        assert_eq!(v["VICTORY_COMPOSE"], "1");
    }
}
