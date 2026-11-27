//! The stash behind `crate::lazy` (`VICTORY_BACK_ON_READ=1`): read-back
//! memory in slots of 64 KiB that deferred pieces are fetched into, on a list,
//! fence and (copy) queue of their own, from the relay thread only. A slot
//! stays as fetched until it is freed, so a fault merges from it with no
//! call to the card (`crate::lazy::fault`).

use super::*;

const SLOT: u64 = 64 << 10;
/// Slots to a buffer (16 MiB).
const PER: u32 = 256;

struct Stash {
    fetch: Option<RelayFetch>,
    /// Buffers of `PER` slots and their mappings.
    chunks: Vec<(ID3D12Resource, usize)>,
    free: Vec<u32>,
    /// The fence value of the last send, and whether it was waited for.
    sent: u64,
    waited: bool,
}

// SAFETY: COM interfaces of the free-threaded D3D12 device and a kernel event
// handle, used under the mutex only.
unsafe impl Send for Stash {}

static STASH: Mutex<Option<Stash>> = Mutex::new(None);

/// `VICTORY_BACK_ON_READ_MB`: read-back memory for the stash at most (1024).
fn most_chunks() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        let mb = std::env::var("VICTORY_BACK_ON_READ_MB").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(1024usize);
        (mb / 16).clamp(1, 1024)
    })
}

/// A free slot: its number and the address of its mapping.
pub(crate) fn lazy_stash_slot() -> Option<(u32, usize)> {
    let host = host()?;
    let mut guard = STASH.lock().unwrap_or_else(|e| e.into_inner());
    let s = guard.get_or_insert_with(|| Stash { fetch: None, chunks: Vec::new(), free: Vec::new(), sent: 0, waited: true });
    if s.free.is_empty() {
        if s.chunks.len() >= most_chunks() {
            return None;
        }
        let (buffer, data) = host.shared_buffer(SLOT * u64::from(PER))?;
        let first = s.chunks.len() as u32 * PER;
        s.chunks.push((buffer, data as usize));
        s.free.extend((first..first + PER).rev());
    }
    let slot = s.free.pop()?;
    let (_, base) = s.chunks[(slot / PER) as usize];
    Some((slot, base + (u64::from(slot % PER) * SLOT) as usize))
}

/// Slot `slot` is not needed any more.
pub(crate) fn lazy_stash_free(slot: u32) {
    let mut guard = STASH.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(s) = guard.as_mut() {
        s.free.push(slot);
    }
}

/// Fetches `copies` (copy, offsets in it, slot, offset in the slot) from the
/// card's copies as they are now; not waited for ([`lazy_stash_wait`]).
/// Whether it was sent.
pub(crate) fn lazy_stash_send(copies: &[(std::sync::Arc<crate::shadow::Shadow>, usize, usize, u32, usize)]) -> bool {
    let Some(host) = host() else {
        return false;
    };
    let mut guard = STASH.lock().unwrap_or_else(|e| e.into_inner());
    let Some(s) = guard.as_mut() else {
        return false;
    };
    if !s.waited {
        // (The last one first: one list.)
        wait(s);
    }
    if s.fetch.is_none() {
        s.fetch = host.make_relay_fetch();
    }
    let Some(f) = s.fetch.as_mut() else {
        return false;
    };
    f.sent += 1;
    let (fence, sent) = (f.fence.clone(), f.sent);
    let (allocator, list) = f.list.clone();
    // (Only on a copy queue of its own, `VICTORY_RELAY_COPY=1`.)
    let Some(queue) = f.queue.clone() else {
        return false;
    };
    let chunks = &s.chunks;
    // SAFETY: the stash's own list and fence under the mutex; live buffers
    // (the copies are held by the caller's groups; the slots are the
    // stash's).
    let ran = unsafe {
        (|| -> windows::core::Result<()> {
            allocator.Reset()?;
            list.Reset(&allocator, None)?;
            for (shadow, lo, hi, slot, at) in copies {
                let (buffer, _) = &chunks[(slot / PER) as usize];
                let dst = u64::from(slot % PER) * SLOT + *at as u64;
                list.CopyBufferRegion(buffer, dst, &shadow.buffer, *lo as u64, (*hi - *lo) as u64);
            }
            list.Close()?;
            let lists = [Some(list.cast::<ID3D12CommandList>()?)];
            queue.ExecuteCommandLists(&lists);
            queue.Signal(&fence, sent)?;
            Ok(())
        })()
    };
    if let Err(e) = ran {
        tracing::warn!(error = %e, "deferred pieces could not be fetched into the stash");
        return false;
    }
    s.sent = sent;
    s.waited = false;
    true
}

fn wait(s: &mut Stash) {
    let Some(f) = s.fetch.as_ref() else {
        s.waited = true;
        return;
    };
    // SAFETY: the stash's own fence and event under the mutex.
    unsafe {
        if f.fence.GetCompletedValue() < s.sent && f.fence.SetEventOnCompletion(s.sent, f.event).is_ok() {
            WaitForSingleObject(f.event, INFINITE);
        }
    }
    s.waited = true;
}

/// The last [`lazy_stash_send`] has run.
pub(crate) fn lazy_stash_wait() {
    let mut guard = STASH.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(s) = guard.as_mut()
        && !s.waited
    {
        wait(s);
    }
}
