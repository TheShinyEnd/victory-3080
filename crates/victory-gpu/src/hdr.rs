//! The host's screens and HDR output.
//!
//! * `VICTORY_MONITOR=<part of the monitor's name | \\.\DISPLAYn | DISPLAYn | #index | primary>`: the screen the
//!   title's window goes on (default: the primary one). `VICTORY_FULLSCREEN=1` makes it a borderless window of that
//!   screen's size. Every output is logged at start with its name, bounds, refresh, colour space, bits per colour and
//!   luminance range, and whether Windows has HDR on for it.
//! * `VICTORY_HDR=auto|on|off|stretch` (default auto: on when Windows HDR is on for the chosen screen):
//!   - `on`: the title is told what a console tells it on an HDR TV (`XDisplayTryEnableHdrMode`: enabled, with the
//!     screen's luminance), the swap chain is `R10G10B10A2` with `G2084/P2020` (PQ) and HDR10 metadata, planes the
//!     title presents as PQ go through unchanged and planes it presents as SDR (sRGB/709) are converted to PQ at
//!     `VICTORY_HDR_PAPER_NITS` (default 200) in the compose shader (`compose.rs`).
//!   - `stretch`: the title stays SDR; the PQ swap chain is fed from its SDR planes with an APPROXIMATE inverse tone map
//!     (highlights are pushed up towards the screen's peak). Not the title's HDR: an explicit opt-in.
//!   - `off`: SDR as before.
//!
//! The runtime never changes Windows' display settings: if HDR is off for the chosen screen it says so once.

use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

/// `DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020`.
pub const CS_PQ_2020: i32 = 12;

/// One output of a graphics card, with the monitor on it.
#[derive(Clone, Debug)]
pub struct Monitor {
    pub index: usize,
    /// `\\.\DISPLAYn`.
    pub device: String,
    /// The monitor's own name (EDID), if Windows says it.
    pub name: String,
    /// left, top, right, bottom of the screen and of its work area.
    pub bounds: [i32; 4],
    pub work: [i32; 4],
    pub primary: bool,
    pub refresh: u32,
    pub bits: u32,
    pub colour_space: i32,
    pub min_nits: f32,
    pub max_nits: f32,
    pub full_frame_nits: f32,
    /// Windows' HDR switch is on for it.
    pub hdr_on: bool,
}

/// What `VICTORY_HDR` came to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Off,
    On,
    Stretch,
}

/// Where the window goes.
#[derive(Clone, Copy, Debug)]
pub struct Target {
    pub bounds: [i32; 4],
    pub work: [i32; 4],
    pub fullscreen: bool,
}

struct State {
    monitors: Vec<Monitor>,
    chosen: Option<usize>,
    /// `VICTORY_MONITOR` asked for something.
    asked: bool,
    fullscreen: bool,
    mode: Mode,
    paper: f32,
}

static STATE: OnceLock<State> = OnceLock::new();
/// The swap chain really presents PQ.
static ACTIVE: AtomicBool = AtomicBool::new(false);
/// The title was told HDR is on.
static TOLD: AtomicBool = AtomicBool::new(false);

fn on(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| !matches!(v.trim().to_ascii_lowercase().as_str(), "" | "0" | "off" | "false" | "no"))
}

fn state() -> &'static State {
    STATE.get_or_init(init)
}

fn init() -> State {
    let monitors = enumerate();
    let want = std::env::var("VICTORY_MONITOR").unwrap_or_default();
    let want = want.trim().to_owned();
    let asked = !want.is_empty() && !want.eq_ignore_ascii_case("primary");
    let primary = monitors.iter().position(|m| m.primary).or(if monitors.is_empty() { None } else { Some(0) });
    let chosen = if asked { pick(&monitors, &want).or_else(|| {
        tracing::warn!(want, "VICTORY_MONITOR names no monitor found; the primary one is used");
        primary
    }) } else { primary };
    let fullscreen = on("VICTORY_FULLSCREEN");
    let paper = std::env::var("VICTORY_HDR_PAPER_NITS")
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .map_or(200.0, |v| v.clamp(80.0, 500.0));
    for m in &monitors {
        tracing::info!(
            index = m.index,
            device = %m.device,
            name = %m.name,
            bounds = ?m.bounds,
            primary = m.primary,
            refresh_hz = m.refresh,
            bits_per_colour = m.bits,
            colour_space = m.colour_space,
            min_nits = m.min_nits,
            max_nits = m.max_nits,
            max_full_frame_nits = m.full_frame_nits,
            windows_hdr_on = m.hdr_on,
            chosen = chosen == Some(m.index),
            "display output{}",
            if chosen == Some(m.index) { " (CHOSEN)" } else { "" }
        );
    }
    if monitors.is_empty() {
        tracing::warn!("no display output found through DXGI");
    }
    let requested = std::env::var("VICTORY_HDR").unwrap_or_default().trim().to_ascii_lowercase();
    let screen = chosen.and_then(|i| monitors.get(i));
    let mode = match requested.as_str() {
        "" | "auto" => match screen {
            Some(m) if m.hdr_on => Mode::On,
            Some(m) => {
                tracing::warn!(
                    "HDR is OFF in Windows for {} ({}): turn on \"Use HDR\" for that display in Windows Settings > System > Display > HDR to get HDR output (VIctory never changes display settings); showing SDR",
                    m.name_or_device(),
                    m.device
                );
                Mode::Off
            }
            None => Mode::Off,
        },
        "on" | "1" | "true" | "yes" => {
            if let Some(m) = screen.filter(|m| !m.hdr_on) {
                tracing::warn!("VICTORY_HDR=on but Windows HDR is OFF for {} ({}): the PQ picture will be shown without HDR; turn on \"Use HDR\" for it in Windows Settings", m.name_or_device(), m.device);
            }
            Mode::On
        }
        "off" | "0" | "false" | "no" => Mode::Off,
        "stretch" => {
            tracing::warn!("VICTORY_HDR=stretch: the title stays SDR and its picture is expanded to HDR by an APPROXIMATE inverse tone map (not real HDR)");
            if let Some(m) = screen.filter(|m| !m.hdr_on) {
                tracing::warn!("Windows HDR is OFF for {} ({}): turn on \"Use HDR\" for it in Windows Settings", m.name_or_device(), m.device);
            }
            Mode::Stretch
        }
        other => {
            tracing::warn!(other, "VICTORY_HDR: use auto, on, off or stretch; auto is used");
            screen.map_or(Mode::Off, |m| if m.hdr_on { Mode::On } else { Mode::Off })
        }
    };
    tracing::info!(
        chosen = ?screen.map(|m| m.device.as_str()),
        hdr_mode = ?mode,
        paper_nits = paper,
        fullscreen,
        asked_monitor = %want,
        "display: window screen and HDR mode resolved"
    );
    State { monitors, chosen, asked, fullscreen, mode, paper }
}

impl Monitor {
    fn name_or_device(&self) -> &str {
        if self.name.is_empty() { &self.device } else { &self.name }
    }
}

fn pick(monitors: &[Monitor], want: &str) -> Option<usize> {
    let lower = want.to_ascii_lowercase();
    if let Some(n) = lower.strip_prefix('#').and_then(|n| n.trim().parse::<usize>().ok()) {
        return monitors.iter().find(|m| m.index == n).map(|m| m.index);
    }
    let device = lower.trim_start_matches("\\\\.\\").to_owned();
    if device.starts_with("display") {
        let hit = monitors.iter().find(|m| m.device.to_ascii_lowercase().trim_start_matches("\\\\.\\") == device);
        if hit.is_some() {
            return hit.map(|m| m.index);
        }
    }
    monitors
        .iter()
        .find(|m| {
            // (Dell's gaming brand is not in the monitor's EDID name: AW... models are Alienware.)
            let brand = if m.name.to_ascii_uppercase().starts_with("AW") { " alienware" } else { "" };
            format!("{}{brand}", m.name.to_ascii_lowercase()).contains(&lower) || m.device.to_ascii_lowercase().contains(&lower)
        })
        .map(|m| m.index)
}

/// Every output found (logged on the first call).
pub fn monitors() -> &'static [Monitor] {
    &state().monitors
}

/// The screen the window goes on.
pub fn chosen() -> Option<&'static Monitor> {
    let s = state();
    s.chosen.and_then(|i| s.monitors.get(i))
}

/// The window's place, when `VICTORY_MONITOR` or `VICTORY_FULLSCREEN` asks for one (else the title's own).
pub fn target() -> Option<Target> {
    let s = state();
    if !s.asked && !s.fullscreen {
        return None;
    }
    let m = chosen()?;
    Some(Target { bounds: m.bounds, work: m.work, fullscreen: s.fullscreen })
}

/// What `VICTORY_HDR` came to.
pub fn wanted() -> Mode {
    state().mode
}

/// Paper white in nits (`VICTORY_HDR_PAPER_NITS`).
pub fn paper_nits() -> f32 {
    state().paper
}

/// The screen's peak, for the stretch and the metadata.
///
/// Windows reports what the monitor's EDID claims (4200 nits for a QD-OLED that peaks near 1000), and a title that
/// tone maps to 4200 nits is clipped or squashed by the display: the value is capped at 1000 nits unless
/// `VICTORY_HDR_PEAK_NITS` says otherwise (80 to 10000).
pub fn peak_nits() -> f32 {
    if let Some(v) = std::env::var("VICTORY_HDR_PEAK_NITS").ok().and_then(|v| v.trim().parse::<f32>().ok()) {
        return v.clamp(80.0, 10_000.0);
    }
    chosen().map(|m| m.max_nits).filter(|&n| n >= 80.0).map_or(1000.0, |n| n.min(1000.0))
}

/// What the title is told on `XDisplayTryEnableHdrMode`: (min, max, max full frame) in nits when HDR is on. Said once
/// to the log with the values.
pub fn title_answer() -> Option<(f32, f32, f32)> {
    if wanted() != Mode::On {
        return None;
    }
    let m = chosen();
    let min = m.map_or(0.0, |m| m.min_nits).max(0.0);
    let max = peak_nits();
    let full = m.map(|m| m.full_frame_nits).filter(|&n| n >= 80.0).map_or(max.min(600.0), |n| n.min(max).min(600.0));
    TOLD.store(true, Relaxed);
    Some((min, max, full))
}

/// The title was told HDR is on.
pub fn told() -> bool {
    TOLD.load(Relaxed)
}

pub(crate) fn set_active(on: bool) {
    ACTIVE.store(on, Relaxed);
}

/// The swap chain presents PQ.
pub fn active() -> bool {
    ACTIVE.load(Relaxed)
}

/// For the F8 page and the log: the HDR state in a few words (the lines stay under 21 characters).
pub fn summary() -> (String, String) {
    let screen = chosen().map_or_else(|| "NO SCREEN".to_owned(), |m| m.device.trim_start_matches("\\\\.\\").to_owned());
    let hdr = match (wanted(), active()) {
        (Mode::Off, _) => match chosen() {
            Some(m) if !m.hdr_on => "SDR WINDOWS HDR OFF".to_owned(),
            _ => "SDR".to_owned(),
        },
        (Mode::On, true) => format!("PQ PAPER {:.0}", paper_nits()),
        (Mode::Stretch, true) => format!("STRETCH PAPER {:.0}", paper_nits()),
        (_, false) => "SDR PQ NOT SET".to_owned(),
    };
    let name = chosen().map_or("", |m| m.name.as_str());
    let cut = |s: String| s.to_ascii_uppercase().chars().take(21).collect::<String>();
    (cut(hdr), cut(format!("{screen} {name}")))
}

/// The shader's input mode of a plane: 0 sRGB/709, 1 PQ/2020, 2 linear 709 (scRGB, 1.0 = 80 nits), 3 gamma 2.2 / 2020.
/// `space` is the colour space the title gave the plane (`u32::MAX`: not known), `format` its DXGI format.
pub fn plane_mode(space: u32, format: i32) -> u32 {
    if let Ok(v) = std::env::var("VICTORY_HDR_PLANE") {
        match v.trim().to_ascii_lowercase().as_str() {
            "pq" => return 1,
            "srgb" => return 0,
            "linear" => return 2,
            _ => {}
        }
    }
    match space {
        12 | 13 => 1,
        1 => 2,
        14 | 17 => 3,
        0 | 2 => 0,
        _ => match format {
            // Half floats: scRGB.
            10 => 2,
            // A 10-bit plane of a title that was told HDR is on, with no colour space we know: PQ.
            23 | 24 if told() => 1,
            _ => 0,
        },
    }
}

static SPACES: Mutex<[(u64, u32, u32); 8]> = Mutex::new([(0, u32::MAX, u32::MAX); 8]);

/// The colour spaces the title gave a present's planes (`u32::MAX`: none).
pub fn note_spaces(frame: u64, base: u32, over: u32) {
    let mut s = SPACES.lock().unwrap_or_else(|e| e.into_inner());
    s[(frame % 8) as usize] = (frame, base, over);
}

pub fn spaces(frame: u64) -> (u32, u32) {
    let s = SPACES.lock().unwrap_or_else(|e| e.into_inner());
    let e = s[(frame % 8) as usize];
    if e.0 == frame { (e.1, e.2) } else { (u32::MAX, u32::MAX) }
}

/// Written once per distinct colour space seen on a plane: the evidence of what the title presents.
pub fn log_space(frame: u64, plane: usize, space: u32, format: i32) {
    static SEEN: Mutex<Vec<(usize, u32, i32)>> = Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if seen.contains(&(plane, space, format)) || seen.len() > 32 {
        return;
    }
    seen.push((plane, space, format));
    tracing::info!(
        frame,
        plane,
        colour_space = if space == u32::MAX { -1 } else { space as i64 },
        format,
        mode = plane_mode(space, format),
        told_hdr = told(),
        "present plane colour space as the title gave it (DXGI_COLOR_SPACE_TYPE; 12 = PQ/2020, 0 = sRGB/709, 1 = scRGB; mode 0 sRGB, 1 PQ, 2 linear, 3 g2.2/2020)"
    );
}

/// Perceptual quantiser, nits to code value 0..1.
pub fn pq_encode(nits: f32) -> f32 {
    const M1: f32 = 0.159_301_76;
    const M2: f32 = 78.84375;
    const C1: f32 = 0.8359375;
    const C2: f32 = 18.851_563;
    const C3: f32 = 18.6875;
    let y = (nits / 10_000.0).clamp(0.0, 1.0).powf(M1);
    ((C1 + C2 * y) / (1.0 + C3 * y)).powf(M2)
}

/// The bytes of the panel's white pixel in an `R10G10B10A2` swap chain.
pub fn panel_white() -> &'static [u8] {
    static WHITE: OnceLock<[u8; 4]> = OnceLock::new();
    WHITE.get_or_init(|| {
        if active() {
            let v = (pq_encode(paper_nits()) * 1023.0).round() as u32;
            (v | v << 10 | v << 20 | 3 << 30).to_le_bytes()
        } else {
            [0xFF; 4]
        }
    })
}

#[cfg(windows)]
pub(crate) use win::configure;

#[cfg(not(windows))]
fn enumerate() -> Vec<Monitor> {
    Vec::new()
}

#[cfg(windows)]
use win::enumerate;

#[cfg(windows)]
mod win {
    use super::*;
    use windows::Win32::Devices::Display::*;
    use windows::Win32::Foundation::{LUID, RECT};
    use windows::Win32::Graphics::Dxgi::Common::*;
    use windows::Win32::Graphics::Dxgi::*;
    use windows::Win32::Graphics::Gdi::*;
    use windows::core::{Interface, PCWSTR};

    fn text(w: &[u16]) -> String {
        let len = w.iter().position(|&c| c == 0).unwrap_or(w.len());
        String::from_utf16_lossy(&w[..len])
    }

    /// (GDI device name, the monitor's own name) of every active path.
    fn friendly_names() -> Vec<(String, String)> {
        let mut out = Vec::new();
        // SAFETY: plain display configuration queries into our buffers.
        unsafe {
            let (mut paths, mut modes) = (0u32, 0u32);
            if GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &mut paths, &mut modes) != windows::Win32::Foundation::ERROR_SUCCESS {
                return out;
            }
            let mut path_info = vec![DISPLAYCONFIG_PATH_INFO::default(); paths as usize];
            let mut mode_info = vec![DISPLAYCONFIG_MODE_INFO::default(); modes as usize];
            if QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, &mut paths, path_info.as_mut_ptr(), &mut modes, mode_info.as_mut_ptr(), None)
                != windows::Win32::Foundation::ERROR_SUCCESS
            {
                return out;
            }
            for p in &path_info[..paths as usize] {
                let mut source = DISPLAYCONFIG_SOURCE_DEVICE_NAME::default();
                source.header.r#type = DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME;
                source.header.size = size_of::<DISPLAYCONFIG_SOURCE_DEVICE_NAME>() as u32;
                source.header.adapterId = LUID { LowPart: p.sourceInfo.adapterId.LowPart, HighPart: p.sourceInfo.adapterId.HighPart };
                source.header.id = p.sourceInfo.id;
                let mut target = DISPLAYCONFIG_TARGET_DEVICE_NAME::default();
                target.header.r#type = DISPLAYCONFIG_DEVICE_INFO_GET_TARGET_NAME;
                target.header.size = size_of::<DISPLAYCONFIG_TARGET_DEVICE_NAME>() as u32;
                target.header.adapterId = LUID { LowPart: p.targetInfo.adapterId.LowPart, HighPart: p.targetInfo.adapterId.HighPart };
                target.header.id = p.targetInfo.id;
                if DisplayConfigGetDeviceInfo(&mut source.header) != 0 || DisplayConfigGetDeviceInfo(&mut target.header) != 0 {
                    continue;
                }
                out.push((text(&source.viewGdiDeviceName), text(&target.monitorFriendlyDeviceName)));
            }
        }
        out
    }

    pub(super) fn enumerate() -> Vec<Monitor> {
        let names = friendly_names();
        let mut out: Vec<Monitor> = Vec::new();
        // SAFETY: plain DXGI queries.
        unsafe {
            let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory1>(DXGI_CREATE_FACTORY_FLAGS(0)) else { return out };
            for a in 0.. {
                let Ok(adapter) = factory.EnumAdapters1(a) else { break };
                for o in 0.. {
                    let Ok(output) = adapter.EnumOutputs(o) else { break };
                    let Ok(output) = output.cast::<IDXGIOutput6>() else { continue };
                    let Ok(d) = output.GetDesc1() else { continue };
                    let device = text(&d.DeviceName);
                    let name = names.iter().find(|(g, _)| g.eq_ignore_ascii_case(&device)).map(|(_, n)| n.clone()).unwrap_or_default();
                    let mut info = MONITORINFOEXW::default();
                    info.monitorInfo.cbSize = size_of::<MONITORINFOEXW>() as u32;
                    let work = if GetMonitorInfoW(d.Monitor, (&raw mut info).cast()).as_bool() {
                        info.monitorInfo.rcWork
                    } else {
                        d.DesktopCoordinates
                    };
                    let primary = info.monitorInfo.dwFlags & 1 != 0;
                    let mut mode = DEVMODEW { dmSize: size_of::<DEVMODEW>() as u16, ..Default::default() };
                    let wide: Vec<u16> = device.encode_utf16().chain(Some(0)).collect();
                    let refresh = if EnumDisplaySettingsW(PCWSTR(wide.as_ptr()), ENUM_CURRENT_SETTINGS, &mut mode).as_bool() {
                        mode.dmDisplayFrequency
                    } else {
                        0
                    };
                    let rect = |r: RECT| [r.left, r.top, r.right, r.bottom];
                    out.push(Monitor {
                        index: out.len(),
                        device,
                        name,
                        bounds: rect(d.DesktopCoordinates),
                        work: rect(work),
                        primary,
                        refresh,
                        bits: d.BitsPerColor,
                        colour_space: d.ColorSpace.0,
                        min_nits: d.MinLuminance,
                        max_nits: d.MaxLuminance,
                        full_frame_nits: d.MaxFullFrameLuminance,
                        hdr_on: d.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020,
                    });
                }
            }
        }
        out
    }

    /// Puts the swap chain into PQ/2020 and gives it HDR10 metadata from the screen's values. `false`: not done (said
    /// why), the chain stays as it was.
    pub(crate) fn configure(chain: &IDXGISwapChain3) -> bool {
        // SAFETY: plain swap chain calls.
        unsafe {
            let space = DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
            let support = match chain.CheckColorSpaceSupport(space) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(error = %e, "CheckColorSpaceSupport failed; presenting SDR");
                    return false;
                }
            };
            if support & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT.0 as u32 == 0 {
                tracing::warn!(support, "the swap chain cannot present G2084/P2020 on this output; presenting SDR");
                return false;
            }
            if let Err(e) = chain.SetColorSpace1(space) {
                tracing::warn!(error = %e, "SetColorSpace1(G2084/P2020) failed; presenting SDR");
                return false;
            }
            let max = peak_nits();
            let m = chosen();
            let min = m.map_or(0.0, |m| m.min_nits).max(0.0);
            let full = m.map(|m| m.full_frame_nits).filter(|&n| n >= 80.0).map_or(max.min(600.0), |n| n.min(max).min(600.0));
            let xy = |v: f32| (v * 50_000.0) as u16;
            let meta = DXGI_HDR_METADATA_HDR10 {
                RedPrimary: [xy(0.708), xy(0.292)],
                GreenPrimary: [xy(0.170), xy(0.797)],
                BluePrimary: [xy(0.131), xy(0.046)],
                WhitePoint: [xy(0.3127), xy(0.3290)],
                MaxMasteringLuminance: max as u32,
                MinMasteringLuminance: (min * 10_000.0) as u32,
                MaxContentLightLevel: max as u16,
                MaxFrameAverageLightLevel: full as u16,
            };
            let set = chain.cast::<IDXGISwapChain4>().and_then(|c| {
                c.SetHDRMetaData(
                    DXGI_HDR_METADATA_TYPE_HDR10,
                    Some(std::slice::from_raw_parts((&raw const meta).cast::<u8>(), size_of::<DXGI_HDR_METADATA_HDR10>())),
                )
            });
            match set {
                Ok(()) => tracing::info!(
                    colour_space = space.0,
                    max_nits = max,
                    min_nits = min,
                    max_full_frame_nits = full,
                    paper_nits = paper_nits(),
                    "HDR output: swap chain set to RGB_FULL_G2084_NONE_P2020 (PQ) with HDR10 metadata"
                ),
                Err(e) => tracing::warn!(error = %e, "SetHDRMetaData failed; the colour space is set, without metadata"),
            }
            true
        }
    }
}
