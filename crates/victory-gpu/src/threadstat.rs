//! Who uses the processor ("stream2" agent): once per call (the memory line,
//! every 5 s) one line with the CPU time of the process's threads since the
//! last call, grouped by the module their start routine lies in (the title's
//! threads start in the title's image, ours in the runtime's) and by name,
//! with the busiest named. `VICTORY_THREAD_STATS=0` turns it off.
//!
//! The question it answers: do the title's loader threads get processor time,
//! or do our 8+ list, landing, pipeline and card threads starve them (the
//! whole process at 100 % of the machine, the title's share small)?
#![cfg(windows)]

use std::collections::HashMap;
use std::sync::Mutex;

#[link(name = "ntdll")]
unsafe extern "system" {
    fn NtQueryInformationThread(
        thread: *mut std::ffi::c_void,
        class: i32,
        out: *mut std::ffi::c_void,
        len: u32,
        returned: *mut u32,
    ) -> i32;
}

struct Seen {
    at: std::time::Instant,
    /// By thread id: CPU time (100 ns units, kernel + user) so far.
    cpu: HashMap<u32, u64>,
}

static LAST: Mutex<Option<Seen>> = Mutex::new(None);

fn module_of(address: usize) -> String {
    use windows_sys::Win32::System::LibraryLoader::{
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, GetModuleFileNameW,
        GetModuleHandleExW,
    };
    let mut module = std::ptr::null_mut();
    // SAFETY: asks the loader which module holds an address.
    let ok = unsafe {
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            address as *const u16,
            &mut module,
        )
    };
    if ok == 0 {
        return "unmapped".into();
    }
    let mut buffer = [0u16; 260];
    // SAFETY: a buffer of the length given.
    let len = unsafe { GetModuleFileNameW(module, buffer.as_mut_ptr(), buffer.len() as u32) } as usize;
    let full = String::from_utf16_lossy(&buffer[..len.min(buffer.len())]);
    full.rsplit(['\\', '/']).next().unwrap_or("?").to_string()
}

/// Logs the line (the first call only takes the first look).
pub(crate) fn tick() {
    use windows_sys::Win32::Foundation::{CloseHandle, FILETIME, INVALID_HANDLE_VALUE, LocalFree};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next,
    };
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcessId, GetThreadDescription, GetThreadTimes, OpenThread, THREAD_QUERY_LIMITED_INFORMATION,
    };
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_THREAD_STATS").is_none_or(|v| v != "0")) {
        return;
    }
    const START: i32 = 9;
    let now = std::time::Instant::now();
    // (id, cpu, name, module, start)
    let mut rows: Vec<(u32, u64, String, String)> = Vec::new();
    let mut ready_threads = 0u32;
    // SAFETY: a snapshot of our own threads; handles opened are closed; names are freed as asked.
    unsafe {
        let own = GetCurrentProcessId();
        let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if snapshot == INVALID_HANDLE_VALUE {
            return;
        }
        let mut entry = THREADENTRY32 { dwSize: size_of::<THREADENTRY32>() as u32, ..Default::default() };
        let mut more = Thread32First(snapshot, &mut entry) != 0;
        while more {
            if entry.th32OwnerProcessID == own {
                let thread = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, 0, entry.th32ThreadID);
                if !thread.is_null() {
                    let zero = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
                    let (mut c, mut e, mut k, mut u) = (zero, zero, zero, zero);
                    GetThreadTimes(thread, &mut c, &mut e, &mut k, &mut u);
                    let ft = |t: FILETIME| u64::from(t.dwHighDateTime) << 32 | u64::from(t.dwLowDateTime);
                    let cpu = ft(k) + ft(u);
                    let mut start = 0usize;
                    NtQueryInformationThread(thread, START, (&raw mut start).cast(), 8, std::ptr::null_mut());
                    let mut name = std::ptr::null_mut();
                    let mut text = String::new();
                    if GetThreadDescription(thread, &mut name) >= 0 && !name.is_null() {
                        let len = (0..).take_while(|&i| *name.add(i) != 0).count();
                        text = String::from_utf16_lossy(std::slice::from_raw_parts(name, len));
                        LocalFree(name.cast());
                    }
                    CloseHandle(thread);
                    ready_threads += 1;
                    rows.push((entry.th32ThreadID, cpu, text, module_of(start)));
                }
            }
            more = Thread32Next(snapshot, &mut entry) != 0;
        }
        CloseHandle(snapshot);
    }
    let cpu_now: HashMap<u32, u64> = rows.iter().map(|r| (r.0, r.1)).collect();
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let Some(before) = last.replace(Seen { at: now, cpu: cpu_now }) else {
        return;
    };
    let secs = now.duration_since(before.at).as_secs_f64().max(0.001);
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    // Per group (module): threads, cores used; and per thread the busiest.
    let mut groups: HashMap<String, (u32, f64)> = HashMap::new();
    let mut busy: Vec<(f64, u32, String, String)> = Vec::new();
    let mut total = 0.0;
    for (id, cpu, name, module) in &rows {
        let delta = cpu.saturating_sub(before.cpu.get(id).copied().unwrap_or(*cpu)) as f64 / 1e7 / secs;
        total += delta;
        let g = groups.entry(module.clone()).or_default();
        g.0 += 1;
        g.1 += delta;
        if delta >= 0.05 {
            busy.push((delta, *id, name.clone(), module.clone()));
        }
    }
    busy.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut groups: Vec<_> = groups.into_iter().collect();
    groups.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
    let groups: Vec<String> =
        groups.iter().map(|(m, (n, c))| format!("{m}: {n} threads {c:.2} cores")).collect();
    let top: Vec<String> = busy
        .iter()
        .take(14)
        .map(|(c, id, name, module)| format!("{id} {} [{module}] {c:.2}", if name.is_empty() { "-" } else { name }))
        .collect();
    tracing::info!(
        "threads (last {secs:.1} s): {ready_threads} threads use {total:.2} of {cores} logical cores ({:.0} % of the machine); by start module: {}; busiest (cores): {}",
        total / cores as f64 * 100.0,
        groups.join("; "),
        top.join(", ")
    );
}
