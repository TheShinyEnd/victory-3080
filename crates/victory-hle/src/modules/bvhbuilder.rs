//! `XboxBvhBuilder.dll`, answered on the CPU (`VICTORY_RT=cpu`).
//!
//! The title builds its ray-tracing structures (the console's own node
//! format, not DXR) through one object of that DLL, whose builder stages are
//! compute programs for 64-lane waves. This module takes the DLL's exports in
//! its place (the title's imports bind to us first), and writes the structures
//! itself with [`victory_shader::bvh_build`], at record time, into the memory
//! the title named: the GPU reads title memory by address, and the runtime's
//! write watching sees our stores as the title's own.
//!
//! Which calls, and what each does:
//!
//! | export | here |
//! |---|---|
//! | constructor, destructor, `Set64BitMortonCodes` | nothing (no GPU builder is made) |
//! | `GetRaytracingAccelerationStructurePrebuildInfo` | the DLL's own CPU arithmetic (it reads only its inputs), so the title allocates the console's sizes |
//! | `BuildRaytracingAccelerationStructureBatched2X`, `BuildRaytracingAccelerationStructure` | a BLAS from the geometry descs' vertex/index buffers, a TLAS from the instance descs; an update refits; the result must fit the console's size |
//! | `CopyRaytracingAccelerationStructure` | the bytes (node pointers are relative to the structure) |
//! | `RehydrateRaytracingAccelerationStructureBatched2X`/`BatchedX` | the DLL's own CPU rehydration of the structures the title ships packed (`RehydrateRaytracingAccelerationStructureX`), on addresses we checked are mapped |
//!
//! After FiveStar's findings for the same title (GPL-3.0; see
//! `victory_shader::bvh_build`). Not done: procedural (AABB) geometry,
//! serialization and deserialization copies, anything the title asks of the
//! other exports of the DLL.

#![allow(unsafe_op_in_unsafe_fn)]

use std::ffi::{CString, c_void};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use victory_shader::bvh_build::{self as bb, Instance, Tri};
use windows_sys::Win32::System::LibraryLoader::GetProcAddress;

use crate::{HleExport, HleModule};

const DLL: &str = "XboxBvhBuilder.dll";
const PREBUILD: &str = "?GetRaytracingAccelerationStructurePrebuildInfo@XboxBvhBuilder@@QEAAXPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS@@PEBUD3D12XBOX_XDXR_STREAM_DESC@@PEAUD3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO@@@Z";
const REHYDRATE: &str = "?RehydrateRaytracingAccelerationStructureX@XboxBvhBuilder@@QEAAJPEAUID3D12Device11@@PEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@@Z";

/// Whether `VICTORY_RT=cpu`.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_RT").is_ok_and(|v| v.eq_ignore_ascii_case("cpu")))
}

/// `VICTORY_RT_BOX=16|32`: the box node size to build (default: 32, then 16
/// for what does not fit the console's size).
fn forced_box() -> Option<bool> {
    static BOX: OnceLock<Option<bool>> = OnceLock::new();
    *BOX.get_or_init(|| match std::env::var("VICTORY_RT_BOX").as_deref() {
        Ok("16") => Some(false),
        Ok("32") => Some(true),
        _ => None,
    })
}

// ---- counters ------------------------------------------------------------

#[derive(Default)]
struct Stats {
    blas: AtomicU64,
    refit: AtomicU64,
    triangles: AtomicU64,
    tlas: AtomicU64,
    instances: AtomicU64,
    instances_dropped: AtomicU64,
    rehydrated: AtomicU64,
    rehydrate_skipped: AtomicU64,
    rehydrate_faults: AtomicU64,
    invalid: AtomicU64,
    copies: AtomicU64,
    skipped_geometry: AtomicU64,
    skipped_size: AtomicU64,
    skipped_memory: AtomicU64,
    narrow: AtomicU64,
    procedural: AtomicU64,
    build_ns: AtomicU64,
    gather_ns: AtomicU64,
    read_ns: AtomicU64,
    sizes_ns: AtomicU64,
    tree_ns: AtomicU64,
    store_ns: AtomicU64,
    rehydrate_ns: AtomicU64,
    /// The largest ours/console size ratio seen (per mille) and its bytes.
    worst_ratio: AtomicU64,
    ours_total: AtomicU64,
    console_total: AtomicU64,
}

static STATS: Stats = Stats {
    blas: AtomicU64::new(0),
    refit: AtomicU64::new(0),
    triangles: AtomicU64::new(0),
    tlas: AtomicU64::new(0),
    instances: AtomicU64::new(0),
    instances_dropped: AtomicU64::new(0),
    rehydrated: AtomicU64::new(0),
    rehydrate_skipped: AtomicU64::new(0),
    rehydrate_faults: AtomicU64::new(0),
    invalid: AtomicU64::new(0),
    copies: AtomicU64::new(0),
    skipped_geometry: AtomicU64::new(0),
    skipped_size: AtomicU64::new(0),
    skipped_memory: AtomicU64::new(0),
    narrow: AtomicU64::new(0),
    procedural: AtomicU64::new(0),
    build_ns: AtomicU64::new(0),
    gather_ns: AtomicU64::new(0),
    read_ns: AtomicU64::new(0),
    sizes_ns: AtomicU64::new(0),
    tree_ns: AtomicU64::new(0),
    store_ns: AtomicU64::new(0),
    rehydrate_ns: AtomicU64::new(0),
    worst_ratio: AtomicU64::new(0),
    ours_total: AtomicU64::new(0),
    console_total: AtomicU64::new(0),
};

/// Why rehydrations were skipped: source, not packed, memory, geometry,
/// geometry buffers, the DLL faulted, not bottom-level, other.
static SKIP_REASONS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
static LAST_LOG: Mutex<Option<Instant>> = Mutex::new(None);
static NARROW_FIRST: AtomicBool = AtomicBool::new(false);

fn log_now() {
    let mut last = LAST_LOG.lock().unwrap_or_else(|e| e.into_inner());
    let now = Instant::now();
    if last.is_some_and(|t| now.duration_since(t).as_secs() < 5) {
        return;
    }
    *last = Some(now);
    let s = &STATS;
    tracing::info!(
        "rt cpu: blas {} (refit {}, box16 {}, {} triangles), tlas {} ({} instances, {} dropped), rehydrated {} (skipped {}, DLL faults {}, invalid {}), copies {}, \
         skipped: geometry {} size {} memory {}, procedural geometries ignored {}, build {:.1} ms total (read {:.1}, sizes {:.1}, gather {:.1}, tree {:.1}, store {:.1}), rehydrate {:.1} ms total, \
         size ours/console {} / {} bytes, worst ratio {:.2}, rehydration skips {:?}, probes {} calls {:.1} ms",
        s.blas.load(Relaxed),
        s.refit.load(Relaxed),
        s.narrow.load(Relaxed),
        s.triangles.load(Relaxed),
        s.tlas.load(Relaxed),
        s.instances.load(Relaxed),
        s.instances_dropped.load(Relaxed),
        s.rehydrated.load(Relaxed),
        s.rehydrate_skipped.load(Relaxed),
        s.rehydrate_faults.load(Relaxed),
        s.invalid.load(Relaxed),
        s.copies.load(Relaxed),
        s.skipped_geometry.load(Relaxed),
        s.skipped_size.load(Relaxed),
        s.skipped_memory.load(Relaxed),
        s.procedural.load(Relaxed),
        s.build_ns.load(Relaxed) as f64 / 1e6,
        s.read_ns.load(Relaxed) as f64 / 1e6,
        s.sizes_ns.load(Relaxed) as f64 / 1e6,
        s.gather_ns.load(Relaxed) as f64 / 1e6,
        s.tree_ns.load(Relaxed) as f64 / 1e6,
        s.store_ns.load(Relaxed) as f64 / 1e6,
        s.rehydrate_ns.load(Relaxed) as f64 / 1e6,
        s.ours_total.load(Relaxed),
        s.console_total.load(Relaxed),
        s.worst_ratio.load(Relaxed) as f64 / 1000.0,
        SKIP_REASONS.iter().map(|r| r.load(Relaxed)).collect::<Vec<_>>(),
        VQ_CALLS.load(Relaxed),
        VQ_NS.load(Relaxed) as f64 / 1e6,
    );
}

// ---- memory --------------------------------------------------------------

/// Whether `len` bytes from `addr` are committed and accessible (a guard the
/// runtime put for write watching is a protection, not a refusal: reading
/// and writing through it is what the title does).
static VQ_CALLS: AtomicU64 = AtomicU64::new(0);
static VQ_NS: AtomicU64 = AtomicU64::new(0);

fn mapped(addr: u64, len: u64) -> bool {
    let t = Instant::now();
    let r = mapped_inner(addr, len);
    VQ_NS.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
    r
}

fn mapped_inner(addr: u64, len: u64) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    if addr == 0 || addr.checked_add(len).is_none() {
        return false;
    }
    // One byte of every page is read through the kernel, which fails where
    // an access would fault. (VirtualQuery was 5 ms a call in this address
    // space: 16 s of 20 in run rt-cpu5.)
    let (mut at, end) = (addr, addr + len.max(1));
    let mut byte = 0u8;
    while at < end {
        VQ_CALLS.fetch_add(1, Relaxed);
        // SAFETY: a read of our own process into a local byte.
        let ok = unsafe { ReadProcessMemory(GetCurrentProcess(), at as *const c_void, (&raw mut byte).cast(), 1, std::ptr::null_mut()) };
        if ok == 0 {
            return false;
        }
        // Title memory is mapped in blocks of 64 KiB (`XMemMapPages`), so one
        // probe per block does (rt-cpu7: 9.4 M probes of 4 KiB cost 9.9 s).
        at = (at | 0xffff) + 1;
    }
    // The last byte too (a range ending in a block of its own).
    if end - 1 >= (addr | 0xffff) + 1 {
        VQ_CALLS.fetch_add(1, Relaxed);
        // SAFETY: as above.
        let ok = unsafe { ReadProcessMemory(GetCurrentProcess(), (end - 1) as *const c_void, (&raw mut byte).cast(), 1, std::ptr::null_mut()) };
        return ok != 0;
    }
    true
}

/// `len` bytes at `addr` if they are mapped.
unsafe fn bytes<'a>(addr: u64, len: usize) -> Option<&'a [u8]> {
    if len == 0 {
        return Some(&[]);
    }
    // SAFETY: mapped and only read, by the caller's promise.
    mapped(addr, len as u64).then(|| unsafe { std::slice::from_raw_parts(addr as *const u8, len) })
}

unsafe fn bytes_mut<'a>(addr: u64, len: usize) -> Option<&'a mut [u8]> {
    // SAFETY: mapped; the title's memory, which the caller means to write.
    mapped(addr, len as u64).then(|| unsafe { std::slice::from_raw_parts_mut(addr as *mut u8, len) })
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

fn f32_at(b: &[u8], at: usize) -> f32 {
    f32::from_bits(u32_at(b, at))
}

// ---- the DLL's own CPU entry points ---------------------------------------

type PrebuildFn = unsafe extern "system" fn(*mut c_void, *const c_void, *const c_void, *mut u64);
type RehydrateFn = unsafe extern "system" fn(*mut c_void, *mut c_void, *const c_void) -> i32;

struct Dll {
    prebuild: Option<PrebuildFn>,
    rehydrate: Option<RehydrateFn>,
    /// A stand-in object: neither entry point reads it.
    this: usize,
}

fn dll() -> &'static Dll {
    static DLL_ENTRIES: OnceLock<Dll> = OnceLock::new();
    DLL_ENTRIES.get_or_init(|| {
        let entry = |name: &str| {
            let base = crate::runtime::image_named(DLL)?;
            let name = CString::new(name).ok()?;
            // SAFETY: the image is a mapped module of ours, the name NUL-terminated.
            unsafe { GetProcAddress(base as _, name.as_ptr().cast()) }
        };
        let prebuild = entry(PREBUILD).map(|f| unsafe { std::mem::transmute::<_, PrebuildFn>(f) });
        let rehydrate = entry(REHYDRATE).map(|f| unsafe { std::mem::transmute::<_, RehydrateFn>(f) });
        tracing::info!(
            prebuild = prebuild.is_some(),
            rehydrate = rehydrate.is_some(),
            "rt cpu: {DLL}'s CPU entry points (console sizes, rehydration of shipped structures)"
        );
        let this = Box::leak(vec![0u8; 256].into_boxed_slice()).as_mut_ptr() as usize;
        Dll { prebuild, rehydrate, this }
    })
}

/// The console's `[result, scratch, update scratch]` sizes for the inputs.
unsafe fn console_sizes(inputs: *const u8, stream: *const c_void) -> Option<[u64; 3]> {
    let d = dll();
    let f = d.prebuild?;
    let mut info = [0u64; 3];
    // SAFETY: the DLL's own arithmetic over the inputs the title gave.
    unsafe { f(d.this as *mut c_void, inputs.cast(), stream, info.as_mut_ptr()) };
    Some(info)
}

// ---- a fault in the DLL's own code -------------------------------------------
//
// The DLL's CPU rehydration reads more than the ranges the desc names (runs
// rt-cpu4/5: an access at the start of an unmapped page). It runs on a worker
// thread of ours; an access violation there (a vectored handler added after
// the runtime's own, so write-watch faults are still resolved by it) ends the
// thread cleanly instead of the process, and the caller gets "faulted".

struct Job {
    f: RehydrateFn,
    this: usize,
    desc: usize,
}

struct Worker {
    jobs: std::sync::mpsc::Sender<Job>,
    done: std::sync::mpsc::Receiver<Option<i32>>,
}

static WORKER: Mutex<Option<Worker>> = Mutex::new(None);
static WORKER_DONE: Mutex<Option<std::sync::mpsc::Sender<Option<i32>>>> = Mutex::new(None);
static WORKER_THREAD: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Where a faulted worker lands: reports and ends the thread.
extern "system" fn worker_fault() -> ! {
    use windows_sys::Win32::System::Threading::ExitThread;
    WORKER_THREAD.store(0, Relaxed);
    if let Some(done) = WORKER_DONE.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
        let _ = done.send(None);
    }
    // SAFETY: ends this thread.
    unsafe { ExitThread(0) }
}

unsafe extern "system" fn guard_handler(p: *mut windows_sys::Win32::System::Diagnostics::Debug::EXCEPTION_POINTERS) -> i32 {
    use windows_sys::Win32::System::Threading::GetCurrentThreadId;
    const ACCESS_VIOLATION: i32 = 0xC000_0005u32 as i32;
    // SAFETY: the pointers are the system's, valid for the call.
    unsafe {
        let me = GetCurrentThreadId();
        if (*(*p).ExceptionRecord).ExceptionCode == ACCESS_VIOLATION && WORKER_THREAD.load(Relaxed) == me {
            let c = &mut *(*p).ContextRecord;
            c.Rip = worker_fault as *const () as usize as u64;
            c.Rsp = ((c.Rsp - 0x200) & !0xf) - 8;
            return -1; // EXCEPTION_CONTINUE_EXECUTION
        }
    }
    0 // EXCEPTION_CONTINUE_SEARCH
}

fn spawn_worker() -> Worker {
    use windows_sys::Win32::System::Diagnostics::Debug::AddVectoredExceptionHandler;
    use windows_sys::Win32::System::Threading::GetCurrentThreadId;
    static HANDLER: std::sync::Once = std::sync::Once::new();
    // After the runtime's own handlers (which resolve its write-watch faults).
    HANDLER.call_once(|| unsafe {
        AddVectoredExceptionHandler(0, Some(guard_handler));
    });
    let (jobs, rx) = std::sync::mpsc::channel::<Job>();
    let (done_tx, done) = std::sync::mpsc::channel::<Option<i32>>();
    *WORKER_DONE.lock().unwrap_or_else(|e| e.into_inner()) = Some(done_tx.clone());
    std::thread::Builder::new()
        .name("rt-rehydrate".into())
        .stack_size(8 << 20)
        .spawn(move || {
            while let Ok(job) = rx.recv() {
                // SAFETY: the thread id of this thread.
                WORKER_THREAD.store(unsafe { GetCurrentThreadId() }, Relaxed);
                // SAFETY: the DLL's CPU rehydration on a probed desc; see above.
                let r = unsafe { (job.f)(job.this as *mut c_void, std::ptr::null_mut(), job.desc as *const c_void) };
                WORKER_THREAD.store(0, Relaxed);
                if done_tx.send(Some(r)).is_err() {
                    return;
                }
            }
        })
        .expect("the rehydration thread");
    Worker { jobs, done }
}

/// Calls the DLL's rehydration on the worker; None if it took an access
/// violation.
unsafe fn guarded(f: RehydrateFn, this: *mut c_void, desc: *const c_void) -> Option<i32> {
    let mut w = WORKER.lock().unwrap_or_else(|e| e.into_inner());
    let worker = w.get_or_insert_with(spawn_worker);
    if worker.jobs.send(Job { f, this: this as usize, desc: desc as usize }).is_err() {
        *w = None;
        return None;
    }
    match worker.done.recv_timeout(std::time::Duration::from_secs(60)) {
        Ok(Some(r)) => Some(r),
        _ => {
            // Faulted (the thread is gone) or stuck: a new thread next time.
            *w = None;
            None
        }
    }
}

// ---- geometry --------------------------------------------------------------

const GEOMETRY: usize = 0x38;

/// Notes why geometry was refused (the first few, in the log).
fn bad(why: String) {
    static SAID: AtomicU64 = AtomicU64::new(0);
    if SAID.fetch_add(1, Relaxed) < 12 {
        tracing::warn!("rt cpu: geometry refused: {why}");
    }
}

/// One triangle geometry desc (`D3D12_RAYTRACING_GEOMETRY_DESC`), read.
struct Geo {
    flags: u8,
    transform: Option<[f32; 12]>,
    vertex_format: u32,
    stride: u64,
    vertices: u64,
    vertex_count: u32,
    index_size: usize,
    indices: u64,
    index_count: u32,
}

impl Geo {
    /// None for a type that is not triangles (the caller counts it).
    unsafe fn read(desc: &[u8]) -> Result<Option<Geo>, ()> {
        if u32_at(desc, 0) != 0 {
            return Ok(None);
        }
        let (index_format, vertex_format) = (u32_at(desc, 0x10), u32_at(desc, 0x14));
        let index_size = match index_format {
            0 => 0,
            // R16_UINT, and R16_TYPELESS as the title passes it; R32_UINT / R32_TYPELESS.
            57 | 53 => 2,
            42 | 39 => 4,
            other => return Err(bad(format!("index format {other}"))),
        };
        if !bb::supported_format(vertex_format) {
            return Err(bad(format!("vertex format {vertex_format}")));
        }
        let (index_count, vertex_count) = (u32_at(desc, 0x18), u32_at(desc, 0x1c));
        let (transform, indices, vertices, stride) = (u64_at(desc, 8), u64_at(desc, 0x20), u64_at(desc, 0x28), u64_at(desc, 0x30));
        let transform = if transform == 0 {
            None
        } else {
            // SAFETY: checked mapped by `bytes`.
            let b = unsafe { bytes(transform, 48) }.ok_or_else(|| bad(format!("transform at {transform:#x} not mapped")))?;
            Some(std::array::from_fn(|k| f32_at(b, 4 * k)))
        };
        let need_v = stride.saturating_mul(u64::from(vertex_count)).min(1 << 33);
        // The last vertex needs only its own size, not a whole stride.
        let need_v = if vertex_count == 0 { 0 } else { need_v - stride + bb::format_size(vertex_format) as u64 };
        if vertex_count > 0 && (vertices == 0 || !mapped(vertices, need_v)) {
            return Err(bad(format!("vertices at {vertices:#x} x{vertex_count} stride {stride} not mapped")));
        }
        if index_size > 0 && index_count > 0 && (indices == 0 || !mapped(indices, u64::from(index_count) * index_size as u64)) {
            return Err(bad(format!("indices at {indices:#x} x{index_count} not mapped")));
        }
        Ok(Some(Geo {
            flags: desc[4],
            transform,
            vertex_format,
            stride,
            vertices,
            vertex_count,
            index_size,
            indices,
            index_count,
        }))
    }

    fn triangles(&self) -> u32 {
        (if self.index_size > 0 { self.index_count } else { self.vertex_count }) / 3
    }

    fn vertex(&self, index: u32) -> Option<[f32; 3]> {
        if index >= self.vertex_count {
            return None;
        }
        let at = self.vertices + u64::from(index) * self.stride;
        // SAFETY: inside the range checked in `read`.
        let b = unsafe { std::slice::from_raw_parts(at as *const u8, bb::format_size(self.vertex_format)) };
        let p = bb::decode_position(self.vertex_format, b)?;
        Some(match &self.transform {
            None => p,
            Some(m) => std::array::from_fn(|r| m[4 * r] * p[0] + m[4 * r + 1] * p[1] + m[4 * r + 2] * p[2] + m[4 * r + 3]),
        })
    }

    fn corner(&self, k: u32) -> u32 {
        match self.index_size {
            0 => k,
            // SAFETY: inside the range checked in `read`.
            2 => u32::from(unsafe { std::ptr::read_unaligned((self.indices + 2 * u64::from(k)) as *const u16) }),
            _ => unsafe { std::ptr::read_unaligned((self.indices + 4 * u64::from(k)) as *const u32) },
        }
    }

    /// Triangle `primitive`: NaN vertices for one that cannot be read.
    fn triangle(&self, geometry: u32, primitive: u32) -> Tri {
        let bad = [f32::NAN; 3];
        let v = std::array::from_fn(|c| self.vertex(self.corner(3 * primitive + c as u32)).unwrap_or(bad));
        Tri { v, geometry, primitive, flags: self.flags }
    }
}

/// The geometry descs of `inputs` (type, flags, count, layout, pointer).
/// Triangles' only: the count of procedural ones is returned as well.
unsafe fn geometries(inputs: &[u8]) -> Result<(Vec<Option<Geo>>, usize), ()> {
    let count = u32_at(inputs, 8) as usize;
    let pointers = u32_at(inputs, 12) == 1;
    let array = u64_at(inputs, 16);
    let mut out = Vec::with_capacity(count);
    let mut procedural = 0;
    for k in 0..count {
        let at = if pointers {
            // SAFETY: mapped checked.
            u64_at(unsafe { bytes(array + 8 * k as u64, 8) }.ok_or_else(|| bad(format!("geometry pointer array at {array:#x} not mapped")))?, 0)
        } else {
            array + (GEOMETRY * k) as u64
        };
        // SAFETY: mapped checked.
        let desc = unsafe { bytes(at, GEOMETRY) }.ok_or_else(|| bad(format!("geometry desc at {at:#x} not mapped")))?;
        match unsafe { Geo::read(desc) }? {
            Some(g) => out.push(Some(g)),
            None => {
                procedural += 1;
                out.push(None);
            }
        }
    }
    Ok((out, procedural))
}

// ---- builds ------------------------------------------------------------------

static BUILD: Mutex<()> = Mutex::new(());

/// Writes the postbuild sizes the title asked for (compacted, current).
unsafe fn postbuild(count: u32, infos: u64, size: u64) {
    for k in 0..count as u64 {
        // SAFETY: mapped checked.
        let Some(info) = (unsafe { bytes(infos + 16 * k, 16) }) else { return };
        let ty = u32_at(info, 8);
        if (ty == 0 || ty == 3)
            && let Some(slot) = unsafe { bytes_mut(u64_at(info, 0), 8) }
        {
            slot.copy_from_slice(&size.to_le_bytes());
        }
    }
}

/// Stores `structure` at `dest` if it fits the console's `allowed` size.
unsafe fn store(dest: u64, structure: &[u8], allowed: Option<u64>) -> bool {
    if let Some(a) = allowed {
        STATS.ours_total.fetch_add(structure.len() as u64, Relaxed);
        STATS.console_total.fetch_add(a, Relaxed);
        let ratio = structure.len() as u64 * 1000 / a.max(1);
        STATS.worst_ratio.fetch_max(ratio, Relaxed);
        if structure.len() as u64 > a {
            return false;
        }
    }
    // A structure no walker could get lost in, or an empty one.
    let empty;
    let structure = if bb::validate(structure) {
        structure
    } else {
        STATS.invalid.fetch_add(1, Relaxed);
        if STATS.invalid.load(Relaxed) <= 8 {
            tracing::error!(bytes = structure.len(), "rt cpu: a structure failed validation; an empty one is published");
        }
        empty = bb::build_blas(&[]);
        &empty
    };
    // SAFETY: mapped checked.
    match unsafe { bytes_mut(dest, structure.len()) } {
        Some(d) => {
            d.copy_from_slice(structure);
            true
        }
        None => {
            STATS.skipped_memory.fetch_add(1, Relaxed);
            false
        }
    }
}

unsafe fn build_blas(desc: &[u8], pb_count: u32, pb_infos: u64) {
    let dest = u64_at(desc, 0);
    let flags = u32_at(desc, 0xc);
    let source = u64_at(desc, 0x20);
    let tr = Instant::now();
    // SAFETY: the inputs are inside the desc.
    let geos = unsafe { geometries(&desc[8..]) };
    STATS.read_ns.fetch_add(tr.elapsed().as_nanos() as u64, Relaxed);
    let ts = Instant::now();
    let allowed = unsafe { console_sizes(desc[8..].as_ptr(), std::ptr::null()) }.map(|s| s[0]);
    STATS.sizes_ns.fetch_add(ts.elapsed().as_nanos() as u64, Relaxed);
    let Ok((geos, procedural)) = geos else {
        // A structure the title will copy and walk anyway: an empty one.
        STATS.skipped_geometry.fetch_add(1, Relaxed);
        let empty = bb::build_blas(&[]);
        if unsafe { store(dest, &empty, allowed) } {
            unsafe { postbuild(pb_count, pb_infos, empty.len() as u64) };
        }
        return;
    };
    STATS.procedural.fetch_add(procedural as u64, Relaxed);
    // An update refits what is there, when it is ours.
    if flags & 0x20 != 0 && source != 0 {
        // SAFETY: mapped checked.
        if let Some(head) = unsafe { bytes(source, bb::HEADER) }
            && u32_at(head, 0) == bb::BLAS_FLAGS
        {
            let size = (u32_at(head, 4) as usize + 1) * 64;
            if allowed.is_none_or(|a| size as u64 <= a)
                && (source == dest || unsafe { bytes(source, size) }.is_some_and(|s| unsafe { store(dest, s, None) }))
                && let Some(d) = unsafe { bytes_mut(dest, size) }
                && bb::refit(d, &mut |g, p| geos.get(g as usize).and_then(|x| x.as_ref()).filter(|x| p < x.triangles()).map(|x| x.triangle(g, p)))
            {
                STATS.refit.fetch_add(1, Relaxed);
                unsafe { postbuild(pb_count, pb_infos, size as u64) };
                return;
            }
        }
    }
    let t0 = Instant::now();
    let total: usize = geos.iter().flatten().map(|g| g.triangles() as usize).sum();
    let mut tris = Vec::with_capacity(total);
    for (k, g) in geos.iter().enumerate() {
        if let Some(g) = g {
            tris.extend((0..g.triangles()).map(|p| g.triangle(k as u32, p)));
        }
    }
    STATS.gather_ns.fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
    let t1 = Instant::now();
    let first_wide = forced_box().unwrap_or(!NARROW_FIRST.load(Relaxed));
    let mut wide = first_wide;
    let mut built = bb::build_blas_with(&tris, wide);
    if forced_box().is_none() && allowed.is_some_and(|a| built.len() as u64 > a) {
        // Too big at this node size: the other one (the smaller, when wide).
        if wide {
            wide = false;
            built = bb::build_blas_with(&tris, false);
            if allowed.is_some_and(|a| built.len() as u64 <= a) {
                NARROW_FIRST.store(true, Relaxed);
            }
        }
    }
    STATS.tree_ns.fetch_add(t1.elapsed().as_nanos() as u64, Relaxed);
    let t2 = Instant::now();
    // SAFETY: mapped checked in `store`.
    let stored = unsafe { store(dest, &built, allowed) };
    STATS.store_ns.fetch_add(t2.elapsed().as_nanos() as u64, Relaxed);
    if stored {
        STATS.blas.fetch_add(1, Relaxed);
        STATS.triangles.fetch_add(tris.len() as u64, Relaxed);
        STATS.narrow.fetch_add(u64::from(!wide), Relaxed);
        unsafe { postbuild(pb_count, pb_infos, built.len() as u64) };
    } else {
        STATS.skipped_size.fetch_add(1, Relaxed);
        if STATS.skipped_size.load(Relaxed) <= 8 {
            tracing::warn!(triangles = tris.len(), ours = built.len(), console = ?allowed, "rt cpu: a bottom-level structure does not fit the console's size; an empty one is written");
        }
        let empty = bb::build_blas(&[]);
        if unsafe { store(dest, &empty, allowed) } {
            unsafe { postbuild(pb_count, pb_infos, empty.len() as u64) };
        }
    }
}

unsafe fn build_tlas(desc: &[u8], pb_count: u32, pb_infos: u64) {
    let dest = u64_at(desc, 0);
    let count = u32_at(desc, 0x10) as usize;
    let pointers = u32_at(desc, 0x14) == 1;
    let array = u64_at(desc, 0x18);
    let allowed = unsafe { console_sizes(desc[8..].as_ptr(), std::ptr::null()) }.map(|s| s[0]);
    let mut instances = Vec::with_capacity(count);
    let mut dropped = 0u64;
    for k in 0..count as u64 {
        let at = if pointers {
            // SAFETY: mapped checked.
            match unsafe { bytes(array + 8 * k, 8) } {
                Some(p) => u64_at(p, 0),
                None => {
                    STATS.skipped_memory.fetch_add(1, Relaxed);
                    return;
                }
            }
        } else {
            array + 64 * k
        };
        // SAFETY: mapped checked.
        let Some(d) = (unsafe { bytes(at, 64) }) else {
            STATS.skipped_memory.fetch_add(1, Relaxed);
            return;
        };
        let (id, contribution, blas) = (u32_at(d, 0x30), u32_at(d, 0x34), u64_at(d, 0x38));
        let (mut root, mut bounds) = (bb::NO_NODE, [f32::NAN; 6]);
        // SAFETY: mapped checked.
        if let Some(h) = unsafe { bytes(blas, bb::HEADER) }
            && u32_at(h, 0) & 0x8000_0000 != 0
            && u32_at(h, 0) & 0x0100_0000 == 0
        {
            root = u32_at(h, 0x30);
            bounds = std::array::from_fn(|a| f32_at(h, 16 + 4 * a));
        } else if blas != 0 {
            dropped += 1;
        }
        instances.push(Instance {
            transform: std::array::from_fn(|i| f32_at(d, 4 * i)),
            id: id & 0xff_ffff,
            mask: (id >> 24) as u8,
            contribution: contribution & 0xff_ffff,
            flags: (contribution >> 24) as u8,
            blas,
            root,
            bounds,
        });
    }
    let built = bb::build_tlas(&instances);
    // SAFETY: mapped checked in `store`.
    if unsafe { store(dest, &built, allowed) } {
        STATS.tlas.fetch_add(1, Relaxed);
        STATS.instances.fetch_add(u64::from(u32_at(&built, 12)), Relaxed);
        STATS.instances_dropped.fetch_add(dropped, Relaxed);
        unsafe { postbuild(pb_count, pb_infos, built.len() as u64) };
    } else {
        STATS.skipped_size.fetch_add(1, Relaxed);
        if STATS.skipped_size.load(Relaxed) <= 8 {
            tracing::warn!(instances = count, ours = built.len(), console = ?allowed, "rt cpu: a top-level structure does not fit the console's size; not written");
        }
    }
}

unsafe fn build_one(desc: &[u8], pb_count: u32, pb_infos: u64) {
    match u32_at(desc, 8) {
        0 => unsafe { build_tlas(desc, pb_count, pb_infos) },
        1 => unsafe { build_blas(desc, pb_count, pb_infos) },
        _ => {
            STATS.skipped_geometry.fetch_add(1, Relaxed);
        }
    }
}

// ---- exports -----------------------------------------------------------------

extern "system" fn construct(this: *mut c_void, _device: *mut c_void, _no_gpu_builder: u8) -> *mut c_void {
    // `VICTORY_RT_SAH=1`: binned-SAH trees (about 5 % fewer node tests in the unit
    // test's scene, 2-3x the build time) instead of Morton order.
    bb::set_sah(std::env::var("VICTORY_RT_SAH").is_ok_and(|v| v == "1"));
    tracing::info!("rt cpu: XboxBvhBuilder constructed (no GPU builder; structures are built on the CPU)");
    this
}

extern "system" fn destruct(_this: *mut c_void) {}

extern "system" fn set_morton(_this: *mut c_void, _enable: u8) {}

extern "system" fn prebuild_info(_this: *mut c_void, inputs: *const u8, stream: *const c_void, info: *mut u64) {
    // SAFETY: the title's own inputs; `info` has room for three sizes.
    unsafe {
        match console_sizes(inputs, stream) {
            Some(s) => std::ptr::copy_nonoverlapping(s.as_ptr(), info, 3),
            None => {
                // No DLL: a bound of ours.
                let (n, ty) = (*(inputs.add(8) as *const u32) as usize, *(inputs as *const u32));
                let size = if ty == 0 { bb::HEADER + n * (bb::INSTANCE + bb::BOX32) } else { bb::blas_size_bound(n * 1000) };
                let size = size.next_multiple_of(256) as u64;
                std::ptr::copy_nonoverlapping([size, 4096 + size, 4096 + size].as_ptr(), info, 3);
            }
        }
    }
}

extern "system" fn build_batched(_this: *mut c_void, _list: *mut c_void, count: u32, descs: *const u8) {
    if descs.is_null() {
        return;
    }
    let _g = BUILD.lock().unwrap_or_else(|e| e.into_inner());
    let _p = victory_core::prof_scope!("rt.cpu.build");
    let t = Instant::now();
    for k in 0..count as usize {
        // SAFETY: the title's array of descs (0x50 bytes each).
        let desc = unsafe { std::slice::from_raw_parts(descs.add(0x50 * k), 0x50) };
        unsafe { build_one(desc, u32_at(desc, 0x30), u64_at(desc, 0x38)) };
    }
    STATS.build_ns.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
    log_now();
}

extern "system" fn build_single(_this: *mut c_void, _list: *mut c_void, desc: *const u8, postbuild_count: u32, postbuilds: *const u8) {
    if desc.is_null() {
        return;
    }
    let _g = BUILD.lock().unwrap_or_else(|e| e.into_inner());
    let _p = victory_core::prof_scope!("rt.cpu.build");
    let t = Instant::now();
    // SAFETY: the title's desc (0x30 bytes; we read the first 0x30).
    let d = unsafe { std::slice::from_raw_parts(desc, 0x30) };
    unsafe { build_one(d, postbuild_count, postbuilds as u64) };
    STATS.build_ns.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
    log_now();
}

extern "system" fn copy(_this: *mut c_void, _list: *mut c_void, dest: u64, source: u64, mode: i32) {
    // Clone and compact; serialization and visualisation are not ours.
    if mode != 0 && mode != 1 {
        return;
    }
    let _g = BUILD.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: mapped checked.
    let Some(head) = (unsafe { bytes(source, bb::HEADER) }) else { return };
    if u32_at(head, 0) & 0x0100_0000 != 0 {
        // Still packed: the rehydration, not a copy, makes it.
        return;
    }
    let size = (u32_at(head, 4) as usize + 1) * 64;
    // SAFETY: mapped checked; the ranges may overlap.
    if let (Some(from), true) = (unsafe { bytes(source, size) }, mapped(dest, size as u64)) {
        unsafe { std::ptr::copy(from.as_ptr(), dest as *mut u8, size) };
        STATS.copies.fetch_add(1, Relaxed);
    } else {
        STATS.skipped_memory.fetch_add(1, Relaxed);
    }
}

/// Rehydrates the packed structure of `desc` (a 0x30-byte
/// `D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC`) by the DLL's CPU
/// code, once its source, destination and geometry are known to be mapped.
unsafe fn rehydrate_one(desc: &[u8], source: u64) -> i32 {
    let d = dll();
    // `VICTORY_RT_REHYDRATE=0`: the shipped structures stay packed (diagnostics).
    static OFF: OnceLock<bool> = OnceLock::new();
    if *OFF.get_or_init(|| std::env::var("VICTORY_RT_REHYDRATE").is_ok_and(|v| v == "0")) {
        return 0;
    }
    let Some(f) = d.rehydrate else { return 0 };
    let skip = |why: &str| {
        STATS.rehydrate_skipped.fetch_add(1, Relaxed);
        let reason = ["source", "not packed", "memory", "geometry", "geometry buffers", "the DLL faulted", "not bottom-level"]
            .iter()
            .position(|r| *r == why)
            .unwrap_or(7);
        SKIP_REASONS[reason].fetch_add(1, Relaxed);
        0
    };
    // SAFETY: mapped checked.
    let Some(head) = (unsafe { bytes(source, bb::HEADER) }) else { return skip("source") };
    if u32_at(desc, 8) != 1 || u32_at(head, 0) & 0x0100_0000 == 0 {
        return skip("not packed");
    }
    let size = (u32_at(head, 4) as usize + 1) * 64;
    let dest = u64_at(desc, 0);
    if unsafe { bytes(source, size) }.is_none() || !mapped(dest, size as u64) {
        return skip("memory");
    }
    // The geometry descs as one array, with their buffers checked.
    let count = u32_at(desc, 0x10) as usize;
    let pointers = u32_at(desc, 0x14) == 1;
    let array = u64_at(desc, 0x18);
    let mut flat = vec![0u8; count * GEOMETRY];
    for k in 0..count {
        let at = if pointers {
            match unsafe { bytes(array + 8 * k as u64, 8) } {
                Some(p) => u64_at(p, 0),
                None => return skip("geometry"),
            }
        } else {
            array + (GEOMETRY * k) as u64
        };
        let Some(g) = (unsafe { bytes(at, GEOMETRY) }) else { return skip("geometry") };
        match unsafe { Geo::read(g) } {
            Err(()) => return skip("geometry buffers"),
            _ => {}
        }
        flat[GEOMETRY * k..GEOMETRY * (k + 1)].copy_from_slice(g);
    }
    static TRACED: AtomicU64 = AtomicU64::new(0);
    if TRACED.fetch_add(1, Relaxed) < 6 {
        let mut text = format!(
            "rt cpu: rehydrate dest {dest:#x} source {source:#x} size {size:#x} header {:08x} {:08x} {:08x} {:08x}, {count} geometries:",
            u32_at(head, 0),
            u32_at(head, 4),
            u32_at(head, 8),
            u32_at(head, 12)
        );
        for k in 0..count.min(4) {
            let g = &flat[GEOMETRY * k..GEOMETRY * (k + 1)];
            text += &format!(
                " [vf {} if {} v {:#x}+{}x{} i {:#x} x{} t {:#x}]",
                u32_at(g, 0x14),
                u32_at(g, 0x10),
                u64_at(g, 0x28),
                u64_at(g, 0x30),
                u32_at(g, 0x1c),
                u64_at(g, 0x20),
                u32_at(g, 0x18),
                u64_at(g, 8)
            );
        }
        tracing::info!("{text}");
    }
    let mut copy = [0u8; 0x30];
    copy.copy_from_slice(&desc[..0x30]);
    copy[0x14..0x18].copy_from_slice(&0u32.to_le_bytes());
    copy[0x18..0x20].copy_from_slice(&(flat.as_ptr() as u64).to_le_bytes());
    copy[0x20..0x28].copy_from_slice(&source.to_le_bytes());
    // SAFETY: the DLL's CPU rehydration, on a desc whose every range was
    // probed; it reads more than that sometimes (rt-cpu4/5: a read at the start
    // of an unmapped page), so a fault in it is caught.
    let Some(r) = (unsafe { guarded(f, d.this as *mut c_void, copy.as_ptr().cast()) }) else {
        STATS.rehydrate_faults.fetch_add(1, Relaxed);
        // The half-made structure must never be walked.
        // (An empty structure of its own size: root NONE, not zeros, which
        // decode as a root pointing at the header.)
        let empty = bb::build_blas(&[]);
        if let Some(d) = unsafe { bytes_mut(dest, empty.len()) } {
            d.copy_from_slice(&empty);
        }
        return skip("the DLL faulted");
    };
    if r >= 0 {
        // What the DLL made must be a tree a walker can leave.
        // SAFETY: probed above.
        let made = unsafe { bytes(dest, size) };
        if made.is_some_and(|m| !bb::validate_graph(m)) {
            STATS.invalid.fetch_add(1, Relaxed);
            if STATS.invalid.load(Relaxed) <= 8 {
                tracing::error!(dest = format_args!("{dest:#x}"), size, "rt cpu: a rehydrated structure failed validation (a cycle, a pointer out of it or a node twice); an empty one is published");
            }
            let empty = bb::build_blas(&[]);
            if let Some(d) = unsafe { bytes_mut(dest, empty.len()) } {
                d.copy_from_slice(&empty);
            }
            return 0;
        }
        STATS.rehydrated.fetch_add(1, Relaxed);
    } else {
        STATS.rehydrate_skipped.fetch_add(1, Relaxed);
    }
    r
}

extern "system" fn rehydrate_batched2(_this: *mut c_void, _list: *mut c_void, count: u32, descs: *const u8, sources: *const u64) -> i32 {
    if descs.is_null() {
        return 0;
    }
    let _g = BUILD.lock().unwrap_or_else(|e| e.into_inner());
    let _p = victory_core::prof_scope!("rt.cpu.rehydrate");
    let t = Instant::now();
    let mut result = 0;
    for k in 0..count as usize {
        // SAFETY: the title's array of 0x30-byte descs, and of source addresses.
        let desc = unsafe { std::slice::from_raw_parts(descs.add(0x30 * k), 0x30) };
        let source = if sources.is_null() { u64_at(desc, 0x20) } else { unsafe { *sources.add(k) } };
        let r = unsafe { rehydrate_one(desc, source) };
        if r < 0 && result == 0 {
            result = r;
        }
    }
    STATS.rehydrate_ns.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
    log_now();
    result
}

extern "system" fn rehydrate_batched(this: *mut c_void, list: *mut c_void, count: u32, descs: *const u8) -> i32 {
    rehydrate_batched2(this, list, count, descs, std::ptr::null())
}

pub(crate) fn module() -> HleModule {
    let on = enabled();
    let export = |name: &'static str, address: usize| HleExport { name, address };
    HleModule {
        name: "bvhbuilder",
        dlls: &["xboxbvhbuilder.dll"],
        exports: if on {
            vec![
                export("??0XboxBvhBuilder@@QEAA@PEAUID3D12Device11@@_N@Z", construct as *const () as usize),
                export("??1XboxBvhBuilder@@QEAA@XZ", destruct as *const () as usize),
                export("?Set64BitMortonCodes@XboxBvhBuilder@@QEAAX_N@Z", set_morton as *const () as usize),
                export(PREBUILD, prebuild_info as *const () as usize),
                export(
                    "?BuildRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList6@@IPEBUD3D12XBOX_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_BATCHED_DESC2@@@Z",
                    build_batched as *const () as usize,
                ),
                export(
                    "?BuildRaytracingAccelerationStructure@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList4@@PEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@IPEBUD3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC@@@Z",
                    build_single as *const () as usize,
                ),
                export(
                    "?CopyRaytracingAccelerationStructure@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList4@@_K1W4D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE@@@Z",
                    copy as *const () as usize,
                ),
                export(
                    "?RehydrateRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAJPEAUID3D12GraphicsCommandList8@@IPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@PEB_K@Z",
                    rehydrate_batched2 as *const () as usize,
                ),
                export(
                    "?RehydrateRaytracingAccelerationStructureBatchedX@XboxBvhBuilder@@QEAAJPEAUID3D12GraphicsCommandList8@@IPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@@Z",
                    rehydrate_batched as *const () as usize,
                ),
            ]
        } else {
            Vec::new()
        },
    }
}
