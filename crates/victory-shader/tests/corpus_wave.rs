//! Offline corpus run: every compute container of `VICTORY_CORPUS` through
//! `emit::translate` as for a host of 32-lane waves. Writes a TSV
//! (`VICTORY_CORPUS_OUT`): name, ok/refused, form taken (`emit::last_form`),
//! FNV-1a hash of the DXIL and host bindings (what "byte-identical"
//! compares), size or reason. `VICTORY_EXACT64=0` turns the exact wave64
//! stage off for the "before" run.
use std::sync::{Arc, Mutex};

#[test]
#[ignore = "harness: needs VICTORY_CORPUS and VICTORY_CORPUS_OUT"]
fn corpus_translate() {
    let (Some(dir), Some(out)) = (std::env::var_os("VICTORY_CORPUS"), std::env::var_os("VICTORY_CORPUS_OUT")) else {
        return;
    };
    let lanes: u32 = std::env::var("VICTORY_HOST_WAVE").ok().and_then(|v| v.parse().ok()).unwrap_or(32);
    victory_shader::emit::set_host_max_wave(lanes);
    // The live host's view-heap facts (`victory-gpu` descriptor/host.rs): raw twins past a
    // 2^21-slot heap whose last slot is null.
    if std::env::var_os("VICTORY_HOST_TWINS").is_some_and(|v| v == "1") {
        victory_shader::emit::set_raw_views(1 << 21);
        victory_shader::emit::set_view_heap_limit((1 << 21) - 1);
    }
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "dxbc"))
        .collect();
    files.sort();
    let files = Arc::new(files);
    let next = Arc::new(Mutex::new(0usize));
    let rows = Arc::new(Mutex::new(Vec::<(usize, String)>::new()));
    let threads: usize = std::env::var("VICTORY_CORPUS_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(6);
    let hs: Vec<_> = (0..threads)
        .map(|_| {
            let (files, next, rows) = (files.clone(), next.clone(), rows.clone());
            std::thread::Builder::new()
                .stack_size(64 << 20)
                .spawn(move || loop {
                    let i = {
                        let mut n = next.lock().unwrap();
                        let i = *n;
                        *n += 1;
                        i
                    };
                    let Some(path) = files.get(i) else { break };
                    let name = path.file_name().unwrap().to_string_lossy().to_string();
                    let bytes = std::fs::read(path).unwrap();
                    let r = std::panic::catch_unwind(|| victory_shader::emit::translate(&bytes));
                    let form = victory_shader::emit::last_form();
                    let line = match r {
                        Ok(Ok(h)) => {
                            let mut x: u64 = 0xcbf29ce484222325;
                            let mut eat = |v: u64| x = (x ^ v).wrapping_mul(0x100000001b3);
                            for b in &h.dxil {
                                eat(*b as u64);
                            }
                            for u in &h.user_data {
                                eat(u.index as u64 | (u.slot as u64) << 32 | (u.dword as u64) << 40);
                            }
                            for s in h.root_cbvs.iter().chain(&h.root_srvs).chain(&h.root_uavs) {
                                eat(*s as u64);
                            }
                            format!("{name}\tok\t{form}\t{x:016x}\t{} bytes", h.dxil.len())
                        }
                        Ok(Err(e)) => format!("{name}\trefused\t-\t-\t{e}"),
                        Err(_) => format!("{name}\tpanic\t-\t-\t"),
                    };
                    rows.lock().unwrap().push((i, line));
                })
                .unwrap()
        })
        .collect();
    for h in hs {
        h.join().unwrap();
    }
    let mut rows = rows.lock().unwrap().clone();
    rows.sort();
    std::fs::write(out, rows.iter().map(|r| r.1.clone() + "\n").collect::<String>()).unwrap();
}
