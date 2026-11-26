//! `victory quality`: look at and change `quality.toml` (`victory_core::quality`).

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use victory_core::quality::{self, File, GROUPS, OPTIONS, PRESETS};

#[derive(clap::Args)]
pub struct Args {
    /// List every option: key, variable, the four presets' values, live or restart.
    #[arg(long)]
    list: bool,
    /// Make this the file's preset (performance, balanced, quality, ultra). Keeps the overrides unless --reset.
    #[arg(long)]
    preset: Option<String>,
    /// An override, `key=value` (`key=default` removes it). May be repeated.
    #[arg(long = "set", value_name = "KEY=VALUE")]
    set: Vec<String>,
    /// With --list: as JSON (what the settings window reads).
    #[arg(long)]
    json: bool,
    /// Print the variables the file resolves to as NAME=value lines (what a launcher passes on).
    #[arg(long)]
    env: bool,
    /// Drop every override first.
    #[arg(long)]
    reset: bool,
    /// The file (default: VICTORY_QUALITY, else %LOCALAPPDATA%\victory\quality.toml).
    #[arg(long)]
    file: Option<PathBuf>,
}

pub fn run(args: Args) -> Result<()> {
    let env = |k: &str| std::env::var_os(k);
    let path = args.file.clone().unwrap_or_else(|| quality::locate(env).unwrap_or_else(|| quality::default_path(env)));
    if args.list {
        if args.json {
            list_json();
        } else {
            list();
        }
        return Ok(());
    }
    if args.env {
        let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        for (k, v) in quality::resolve(&File::parse(&text)?)? {
            println!("{k}={v}");
        }
        return Ok(());
    }
    let existing = match std::fs::read_to_string(&path) {
        Ok(t) => Some(File::parse(&t).with_context(|| path.display().to_string())?),
        Err(_) => None,
    };
    if args.preset.is_none() && args.set.is_empty() && !args.reset {
        return show(&path, existing.as_ref());
    }
    let mut preset = existing.as_ref().and_then(|f| f.preset.clone()).unwrap_or_else(|| "balanced".to_owned());
    let mut overrides: BTreeMap<String, String> = if args.reset {
        BTreeMap::new()
    } else {
        existing.iter().flat_map(|f| f.options.iter()).map(|(k, v)| (k.clone(), quality::value_text(v))).collect()
    };
    if let Some(p) = &args.preset {
        preset = p.to_ascii_lowercase();
    }
    for pair in &args.set {
        let Some((k, v)) = pair.split_once('=') else { bail!("--set expects key=value, got {pair:?}") };
        let (k, v) = (k.trim(), v.trim());
        let Some(o) = quality::find(k) else { bail!("unknown option {k:?} (victory quality --list)") };
        quality::variable_value(o, v)?;
        if v.is_empty() || v.eq_ignore_ascii_case("default") {
            overrides.remove(k);
        } else {
            overrides.insert(k.to_owned(), v.to_owned());
        }
    }
    let text = quality::render(&preset, &overrides);
    // Check what is written before writing it.
    let file = File::parse(&text)?;
    quality::resolve(&file)?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).ok();
    }
    std::fs::write(&path, text).with_context(|| format!("writing {}", path.display()))?;
    println!("wrote {}", path.display());
    show(&path, Some(&file))
}

fn show(path: &std::path::Path, file: Option<&File>) -> Result<()> {
    println!("quality file: {}{}", path.display(), if file.is_some() { "" } else { "  (none yet: nothing is applied)" });
    let Some(file) = file else { return Ok(()) };
    println!("preset: {}", file.preset.as_deref().unwrap_or("balanced"));
    println!("variables it sets (a variable already set in the environment wins):");
    for (k, v) in quality::resolve(file)? {
        println!("  {k}={v}");
    }
    if !quality::HOOKS_BUILT {
        println!("(render_scale and own_upscaler need the game-settings hooks, which this build does not contain)");
    }
    Ok(())
}

fn list() {
    println!("presets: {}", PRESETS.join(" | "));
    for g in GROUPS {
        println!("\n[{g}]");
        for o in OPTIONS.iter().filter(|o| o.group == g) {
            let when = if o.live { "live (in-game page)" } else { "restart" };
            let hooks = if o.needs_hooks { "  NEEDS GAME-SETTINGS HOOKS (not built in)" } else { "" };
            let shown = |s: &str| if s.is_empty() { "-".to_owned() } else { s.to_owned() };
            println!(
                "  {:<22} {}  [{} / {} / {} / {}]  {when}{hooks}\n      {}\n      cost: {}",
                o.key, o.env, shown(o.presets[0]), shown(o.presets[1]), shown(o.presets[2]), shown(o.presets[3]), o.help, o.cost
            );
        }
    }
}

fn json_str(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// The catalogue as JSON: `tools/VIctory-Settings.ps1` builds its window from it.
fn list_json() {
    use quality::Kind;
    let strs = |v: &[&str]| format!("[{}]", v.iter().map(|s| json_str(s)).collect::<Vec<_>>().join(","));
    let mut items = Vec::new();
    for o in OPTIONS {
        let (kind, words, min, max) = match o.kind {
            Kind::Bool { .. } => ("bool", Vec::new(), 0.0, 0.0),
            Kind::Num { min, max } => ("num", Vec::new(), min, max),
            Kind::Choice(w) => ("choice", w.iter().map(|(w, _)| *w).collect(), 0.0, 0.0),
            Kind::Text => ("text", Vec::new(), 0.0, 0.0),
        };
        items.push(format!(
            "{{\"key\":{},\"env\":{},\"group\":{},\"kind\":\"{kind}\",\"words\":{},\"min\":{min},\"max\":{max},\"presets\":{},\"live\":{},\"needs_hooks\":{},\"help\":{},\"cost\":{}}}",
            json_str(o.key), json_str(o.env), json_str(o.group), strs(&words), strs(&o.presets), o.live, o.needs_hooks, json_str(o.help), json_str(o.cost)
        ));
    }
    println!("{{\"presets\":{},\"groups\":{},\"hooks_built\":{},\"options\":[{}]}}", strs(&PRESETS), strs(&GROUPS), quality::HOOKS_BUILT, items.join(",\n"));
}
