//! Mouse and keyboard as a gamepad: the mouse drives the right stick, the
//! keys a profile maps to buttons, triggers and the left stick.
//!
//! Adapted from ItsMK1/VIctory-MnK-patches 727b819 (GPL-3). Differences: the
//! profile is built in (`bindings.ini` at the root of the tree; a file named
//! by `VICTORY_MNK_BINDINGS` replaces it), a bad file falls back to the
//! built-in one, keys for the right stick (`Look*`) exist, host pads and the
//! scripted keys (`VICTORY_KEYS*`) keep working beside it, and it is on by
//! default only when no scripted keys are set (`VICTORY_MNK=1`/`0` forces).
//!
//! While the title's window is in front, the cursor is hidden and put back
//! to the window's centre at every poll; its displacement is the look (the
//! default; `VICTORY_MNK_MOUSE=device` offers a GameInput mouse device
//! instead, which this title never reads: see `gameinput.rs`'s counters). This
//! module also maps the keys and the mouse buttons to pad buttons.

use super::{
    A, B, DOWN, LEFT, LEFT_SHOULDER, LEFT_STICK, MENU, PadState, RIGHT, RIGHT_SHOULDER,
    RIGHT_STICK, UP, VIEW, X, Y,
};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};
use windows_sys::Win32::{
    Foundation::{HWND, POINT, RECT},
    UI::{Input::KeyboardAndMouse::GetAsyncKeyState, WindowsAndMessaging as wm},
};

const DEFAULT_PROFILE: &str = include_str!("../../../../bindings.ini");

const BUTTONS: &[(&str, u32)] = &[
    ("A", A),
    ("B", B),
    ("X", X),
    ("Y", Y),
    ("LB", LEFT_SHOULDER),
    ("RB", RIGHT_SHOULDER),
    ("LS", LEFT_STICK),
    ("RS", RIGHT_STICK),
    ("Menu", MENU),
    ("View", VIEW),
    ("DpadUp", UP),
    ("DpadDown", DOWN),
    ("DpadLeft", LEFT),
    ("DpadRight", RIGHT),
];
const AXES: [&str; 10] = [
    "LT", "RT", "MoveLeft", "MoveRight", "MoveDown", "MoveUp", "LookLeft", "LookRight", "LookDown",
    "LookUp",
];

struct Bindings {
    buttons: Vec<(u8, u32)>,
    axes: [Vec<u8>; 10],
    sensitivity: f32,
}

fn key(name: &str) -> Option<u8> {
    let name = name.trim().to_ascii_uppercase();
    if name.len() == 1 && name.as_bytes()[0].is_ascii_alphanumeric() {
        return Some(name.as_bytes()[0]);
    }
    if let Some(hex) = name.strip_prefix("0X") {
        return u8::from_str_radix(hex, 16)
            .ok()
            .filter(|v| *v > 0 && *v < 255);
    }
    Some(match name.as_str() {
        "MOUSE1" => 1,
        "MOUSE2" => 2,
        "MOUSE3" => 4,
        "MOUSE4" => 5,
        "MOUSE5" => 6,
        "BACKSPACE" => 8,
        "TAB" => 9,
        "ENTER" => 13,
        "SHIFT" => 16,
        "CTRL" => 17,
        "ALT" => 18,
        "ESC" => 27,
        "SPACE" => 32,
        "LEFT" => 37,
        "UP" => 38,
        "RIGHT" => 39,
        "DOWN" => 40,
        _ => return None,
    })
}

fn parse(file: &str) -> Result<Bindings, String> {
    let mut b = Bindings {
        buttons: Vec::new(),
        axes: Default::default(),
        sensitivity: 0.045,
    };
    for (i, line) in file.lines().enumerate() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let err = || format!("Invalid binding on line {}: {}", i + 1, line);
        let (out, input) = line.split_once('=').ok_or_else(err)?;
        let out = out.trim();
        if out.eq_ignore_ascii_case("Sensitivity") {
            b.sensitivity = input
                .trim()
                .parse::<f32>()
                .ok()
                .filter(|s| s.is_finite() && *s > 0.0)
                .ok_or_else(err)?;
            continue;
        }
        let keys = input
            .split(',')
            .map(|k| key(k).ok_or_else(err))
            .collect::<Result<Vec<_>, _>>()?;
        if let Some((_, bit)) = BUTTONS
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(out))
        {
            b.buttons.extend(keys.into_iter().map(|k| (k, *bit)));
        } else {
            let index = AXES
                .iter()
                .position(|s| s.eq_ignore_ascii_case(out))
                .ok_or_else(err)?;
            b.axes[index].extend(keys);
        }
    }
    Ok(b)
}

fn config() -> Option<&'static Bindings> {
    static CONFIG: OnceLock<Option<Bindings>> = OnceLock::new();
    CONFIG
        .get_or_init(|| {
            if !wanted() {
                return None;
            }
            if let Some(file) = std::env::var_os("VICTORY_MNK_BINDINGS") {
                match std::fs::read_to_string(&file)
                    .map_err(|e| e.to_string())
                    .and_then(|s| parse(&s))
                {
                    Ok(b) => return Some(b),
                    Err(e) => tracing::warn!(
                        "mouse and keyboard profile {file:?} not used, the built-in one is: {e}"
                    ),
                }
            }
            parse(DEFAULT_PROFILE).ok()
        })
        .as_ref()
}

/// `VICTORY_MNK=1` or `0` decides; unset: on unless scripted keys are set
/// (the test harness), so that an interactive session has it.
fn wanted() -> bool {
    match std::env::var("VICTORY_MNK").as_deref().map(str::trim) {
        Ok("0" | "off" | "false") => false,
        Ok("1" | "on" | "true") => true,
        _ => ["VICTORY_KEYS", "VICTORY_KEYS_FILE", "VICTORY_KEYS_EXTRA"]
            .iter()
            .all(|v| std::env::var_os(v).is_none()),
    }
}

pub(crate) fn enabled() -> bool {
    config().is_some()
}

/// The mouse look is the right stick of the stand-in pad (ItsMK1/VIctory-MnK-patches:
/// the cursor is held at the window's centre) and there is no GameInput mouse
/// device. This is the default (`VICTORY_MNK_MOUSE=device` offers the device
/// of vnvidia 12b1d16 instead): the title never asks GameInput for a mouse
/// (no registration or reading of kind 0x20, no GetMouseState in 5 minutes of
/// polling with the device present: the counters in `gameinput.rs`), only
/// for the gamepad and the keyboard, so a device cannot turn the camera.
pub(crate) fn stick_look() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !std::env::var("VICTORY_MNK_MOUSE").is_ok_and(|v| v.trim() == "device"))
}

#[derive(Default)]
struct Mouse {
    window: usize,
    velocity: [f32; 2],
    stick: [f32; 2],
    last: Option<Instant>,
    last_motion: Option<Instant>,
    centered: bool,
}
fn mouse() -> std::sync::MutexGuard<'static, Mouse> {
    static STATE: OnceLock<Mutex<Mouse>> = OnceLock::new();
    STATE
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

// Sample cursor displacement on the input polling thread, independently of frame delivery.
fn recenter(hwnd: HWND) -> Option<[i32; 2]> {
    let mut rect = RECT::default();
    let mut cursor = POINT::default();
    // SAFETY: plain queries into structs of our own.
    unsafe {
        if wm::GetClientRect(hwnd, &mut rect) == 0 || rect.right <= 0 || rect.bottom <= 0 {
            return None;
        }
        let mut center = POINT {
            x: rect.right / 2,
            y: rect.bottom / 2,
        };
        if windows_sys::Win32::Graphics::Gdi::ClientToScreen(hwnd, &mut center) == 0
            || wm::GetCursorPos(&mut cursor) == 0
            || wm::GetForegroundWindow() != hwnd
        {
            return None;
        }
        let delta = [cursor.x - center.x, cursor.y - center.y];
        if delta != [0, 0] && wm::SetCursorPos(center.x, center.y) == 0 {
            return None;
        }
        Some(delta)
    }
}

impl Mouse {
    fn update_look(&mut self, delta: [i32; 2], now: Instant, elapsed: Duration, sensitivity: f32) {
        if delta != [0, 0] {
            self.last_motion = Some(now);
        }
        if self
            .last_motion
            .is_none_or(|t| now.duration_since(t) >= Duration::from_millis(40))
        {
            self.velocity = [0.0; 2];
            self.stick = [0.0; 2];
            return;
        }
        let seconds = elapsed.as_secs_f32();
        let weight = 1.0 - (-seconds / 0.020).exp();
        // Filter velocity before limiting it. Clamping bursts first loses motion at low FPS.
        for (v, d) in self
            .velocity
            .iter_mut()
            .zip([delta[0] as f32, -(delta[1] as f32)])
        {
            let target = d * sensitivity * 0.008 / seconds;
            *v += weight * (target - *v);
        }
        let length = self.velocity[0].hypot(self.velocity[1]);
        if length <= 0.001 {
            self.stick = [0.0; 2];
        } else {
            // Map small movements beyond the controller dead zone, retaining direction.
            let gain = (0.24 + 0.76 * length.min(1.0)) / length;
            self.stick = self.velocity.map(|v| v * gain);
        }
    }
}

/// The pad the mouse and keys make now; `None` when the profile is off.
/// Nothing is pressed while another process's window is in front.
/// `VICTORY_MOUSE_TEST=dx,dy,period_ms`: synthetic mouse motion, `dx,dy` every
/// `period_ms` (at least 1), entering where a real cursor's displacement does
/// (the device state in `gameinput.rs`, or this module's stick), and with no
/// window in front. For live runs, where no mouse can be moved.
pub(super) struct TestHook {
    delta: [i32; 2],
    period: Duration,
    last: Mutex<Option<Instant>>,
}

impl TestHook {
    /// The injected displacement if a period has passed since the last one.
    pub(super) fn next(&self, now: Instant) -> Option<[i32; 2]> {
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        if last.is_none_or(|at| now.duration_since(at) >= self.period) {
            *last = Some(now);
            Some(self.delta)
        } else {
            None
        }
    }
}

fn parse_test(text: &str) -> Option<TestHook> {
    let mut parts = text.split(',').map(|p| p.trim().parse::<i32>().ok());
    let (dx, dy, period) = (parts.next()??, parts.next()??, parts.next()??);
    Some(TestHook { delta: [dx, dy], period: Duration::from_millis(period.max(1) as u64), last: Mutex::new(None) })
}

pub(super) fn test_hook() -> Option<&'static TestHook> {
    static HOOK: OnceLock<Option<TestHook>> = OnceLock::new();
    HOOK.get_or_init(|| std::env::var("VICTORY_MOUSE_TEST").ok().and_then(|v| parse_test(&v))).as_ref()
}

pub(super) fn sample() -> Option<PadState> {
    let b = config()?;
    let mut m = mouse();
    // SAFETY: plain queries.
    let hwnd = unsafe { wm::GetForegroundWindow() };
    let mut pid = 0;
    // SAFETY: as above.
    unsafe { wm::GetWindowThreadProcessId(hwnd, &mut pid) };
    if (hwnd.is_null() || pid != std::process::id()) && !(stick_look() && test_hook().is_some()) {
        *m = Mouse::default();
        return Some(PadState::default());
    }
    if m.window != hwnd as usize {
        *m = Mouse {
            window: hwnd as usize,
            ..Default::default()
        };
    }
    let now = Instant::now();
    let elapsed = m
        .last
        .map_or(Duration::from_millis(8), |last| now.duration_since(last));
    if let (true, Some(hook)) = (stick_look(), test_hook()) {
        if elapsed >= Duration::from_millis(8) {
            let delta = hook.next(now).unwrap_or([0, 0]);
            if delta != [0, 0] {
                super::count(super::stat::RAW_EVENTS, 1);
                super::count(super::stat::RAW_SUM, u64::from(delta[0].unsigned_abs() + delta[1].unsigned_abs()));
            }
            m.update_look(delta, now, elapsed, b.sensitivity);
            m.last = Some(now);
        }
    } else if stick_look() && elapsed >= Duration::from_millis(8) {
        let displacement = recenter(hwnd);
        let delta = if m.centered {
            displacement.unwrap_or_default()
        } else {
            [0, 0]
        };
        m.centered = displacement.is_some();
        if delta != [0, 0] {
            super::count(super::stat::RAW_EVENTS, 1);
            super::count(super::stat::RAW_SUM, u64::from(delta[0].unsigned_abs() + delta[1].unsigned_abs()));
        }
        m.update_look(delta, now, elapsed, b.sensitivity);
        m.last = Some(now);
    }
    // SAFETY: a plain query.
    let held = |vk: u8| (unsafe { GetAsyncKeyState(vk.into()) }) < 0;
    let axes = b
        .axes
        .each_ref()
        .map(|keys| u8::from(keys.iter().copied().any(held)) as f32);
    let look = |v: f32, less: f32, more: f32| (v + more - less).clamp(-1.0, 1.0);
    Some(PadState {
        buttons: b
            .buttons
            .iter()
            .filter(|(vk, _)| held(*vk))
            .fold(0, |v, (_, bit)| v | bit),
        left_trigger: axes[0],
        right_trigger: axes[1],
        left_x: axes[3] - axes[2],
        left_y: axes[5] - axes[4],
        right_x: look(m.stick[0], axes[6], axes[7]),
        right_y: look(m.stick[1], axes[8], axes[9]),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn continuous_motion_and_release() {
        let mut m = Mouse::default();
        let start = Instant::now();
        let dt = Duration::from_millis(8);
        m.update_look([1, 0], start, dt, 0.135);
        assert!(m.stick[0] > 0.24);
        for i in 1..12 {
            m.update_look([20, -20], start + dt * i, dt, 0.135);
        }
        assert!((m.stick[0] - m.stick[1]).abs() < 0.001);
        assert!((m.stick[0].hypot(m.stick[1]) - 1.0).abs() < 0.001);
        m.update_look([0, 0], start + dt * 12, dt, 0.135);
        assert!(m.stick[0] > 0.0);
        m.update_look([0, 0], start + dt * 17, dt, 0.135);
        assert_eq!(m.stick, [0.0; 2]);
    }

    #[test]
    fn test_hook_parses_and_paces() {
        assert!(parse_test("x,1,2").is_none() && parse_test("1,2").is_none());
        let h = parse_test("5,-3,16").unwrap();
        let t = Instant::now();
        assert_eq!(h.next(t), Some([5, -3]));
        assert_eq!(h.next(t + Duration::from_millis(8)), None);
        assert_eq!(h.next(t + Duration::from_millis(16)), Some([5, -3]));
    }

    /// What the title polls: a drag right and up (cursor y falls) gives a
    /// right stick with x > 0 and y > 0 (up positive) at a moderate
    /// sensitivity, and it falls to zero 40 ms after the last motion.
    #[test]
    fn mouse_motion_makes_a_right_stick_that_decays() {
        let mut m = Mouse::default();
        let start = Instant::now();
        let dt = Duration::from_millis(8);
        for i in 0..20 {
            m.update_look([3, -3], start + dt * i, dt, 0.135);
        }
        assert!(m.stick[0] > 0.24 && m.stick[0] < 1.0 && m.stick[1] > 0.24, "{:?}", m.stick);
        m.update_look([-3, 3], start + dt * 20, dt, 0.135);
        for i in 21..30 {
            m.update_look([-3, 3], start + dt * i, dt, 0.135);
        }
        assert!(m.stick[0] < 0.0 && m.stick[1] < 0.0, "{:?}", m.stick);
        m.update_look([0, 0], start + dt * 36, dt, 0.135);
        assert_eq!(m.stick, [0.0; 2]);
    }

    #[test]
    fn built_in_profile_maps_triggers_vehicle_and_look() {
        let b = parse(DEFAULT_PROFILE).unwrap();
        assert_eq!(b.axes[1], [1, b'C']);
        assert_eq!(b.axes[0], [2, b'Z']);
        assert_eq!(b.axes[9], [b'I']);
        assert!(b.buttons.contains(&(b'F', Y)));
        assert!((b.sensitivity - 0.135).abs() < 1e-6);
        assert!(parse("RT=NotAKey").is_err());
    }
}
