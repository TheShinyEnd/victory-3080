//! The title's own progress watchdog, taken out (on by default;
//! `VICTORY_NO_WATCHDOG=0` leaves it in).
//!
//! What the title does (game_scarlett_master.exe 1.0.2.1): a watchdog thread
//! ends the process on purpose when the main thread has not advanced for
//! about 30 one-second ticks, which a layer that runs the title slower than a
//! console (shader builds, a heavy scene at a few frames a second) can
//! trigger without anything being wrong. Two places do it: the handler the
//! thread calls (`mov dword ptr [0], imm32`) and, when the handler returns or
//! is unset, a fallback right after the call (`mov dword ptr [0], 0; int3`).
//!
//! Patch 1 (image offset 0x2215c27, 7 bytes): `mov rax, [rip+disp32]` (the
//! handler pointer) becomes `jmp +0x18` and five NOPs, so the loop goes on
//! its normal path past the call and the fallback fault.
//! Patch 2 (0x23cf683, 11 bytes): the handler's deliberate fault
//! `mov dword ptr [0], 0x4d8bd421` becomes an eleven-byte NOP.
//!
//! Both sites' bytes are compared before anything is written; if either
//! differs (another build of the title) NOTHING is written and one line says
//! so. After it, a real hang of the title is no longer ended by the title:
//! the runtime's own long-block warning (`VICTORY_LONG_BLOCK_MS`, profiler)
//! stays on to show it in the log.
//!
//! Taken from the loader hooks of the NVIDIA fork (itself from ScarlettPort,
//! GPL-3), reduced to these two patches.

/// One patch: `at` (an offset in the image) must hold exactly `expect`.
struct Patch {
    at: usize,
    expect: &'static [u8],
    new: &'static [u8],
}

const PATCHES: [Patch; 2] = [
    Patch {
        at: 0x2215c27,
        expect: &[0x48, 0x8B, 0x05, 0x1A, 0x07, 0xFA, 0x04],
        new: &[0xEB, 0x18, 0x90, 0x90, 0x90, 0x90, 0x90],
    },
    Patch {
        at: 0x23cf683,
        expect: &[0xC7, 0x04, 0x25, 0, 0, 0, 0, 0x21, 0xD4, 0x8B, 0x4D],
        new: &[0x66, 0x66, 0x2E, 0x0F, 0x1F, 0x84, 0, 0, 0, 0, 0],
    },
];

/// Whether the watchdog is to be taken out: on unless `VICTORY_NO_WATCHDOG`
/// is `0`.
pub fn wanted() -> bool {
    !std::env::var("VICTORY_NO_WATCHDOG").is_ok_and(|v| v.trim() == "0")
}

/// Takes the watchdog out of the mapped image; returns whether it did (both
/// patches written). Says in one line what happened.
///
/// # Safety
/// The image stays mapped, `size` is its size, and no thread runs its code yet.
#[cfg(windows)]
pub unsafe fn disarm(base: usize, size: usize) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    for p in &PATCHES {
        if p.at.checked_add(p.expect.len()).is_none_or(|end| end > size) {
            println!("title watchdog: NOT disabled (a site lies outside the image)");
            return false;
        }
        // SAFETY: inside the mapped image (checked above).
        let there = unsafe { std::slice::from_raw_parts((base + p.at) as *const u8, p.expect.len()) };
        if there != p.expect {
            println!(
                "title watchdog: NOT disabled (the bytes at {:#x} are not the ones of title 1.0.2.1; another build?)",
                p.at
            );
            return false;
        }
    }
    for p in &PATCHES {
        let at = base + p.at;
        let mut old = 0;
        // SAFETY: the site's bytes, made writable for this and put back;
        // nothing runs them yet.
        unsafe {
            if VirtualProtect(at as *const _, p.new.len(), PAGE_EXECUTE_READWRITE, &mut old) == 0 {
                println!("title watchdog: NOT disabled (Windows would not make {:#x} writable)", p.at);
                return false;
            }
            std::ptr::copy_nonoverlapping(p.new.as_ptr(), at as *mut u8, p.new.len());
            VirtualProtect(at as *const _, p.new.len(), old, &mut old);
            FlushInstructionCache(GetCurrentProcess(), at as *const _, p.new.len());
        }
    }
    println!(
        "title watchdog: DISABLED (the title's 30 s progress watchdog is patched out at {:#x} and {:#x}; VICTORY_NO_WATCHDOG=0 leaves it in). A real freeze is shown by the long-block warning (VICTORY_LONG_BLOCK_MS), not ended by the title.",
        PATCHES[0].at, PATCHES[1].at
    );
    true
}
