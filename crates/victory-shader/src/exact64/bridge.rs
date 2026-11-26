//! The boundary between the cool_fox emitter and this copy: the program goes
//! in as container bytes, a v7 [`crate::emit::HostShader`] comes out, and a
//! translation is taken only when everything it asks of the host is
//! something the cool_fox host binds the same way.
//!
//! The copy's host contract differs from v7's in title memory (`memory`,
//! `view_addresses`, write marks), heap-picked descriptors (`heap`, the
//! view-stride and view-map root SRVs) and root descriptors read through
//! views. A translation that needs any of those is refused here
//! (`Err(reason)`), never handed to a host that would bind it differently.

use super::emit as ours;
use super::native::RootType as OurRoot;
use crate::emit::{HostShader, StaticSampler, UserData};
use crate::native::RootType;

/// `VICTORY_EXACT64=0` turns the whole stage off.
pub fn enabled() -> bool {
    std::env::var_os("VICTORY_EXACT64").is_none_or(|v| v != "0")
}

fn root(r: OurRoot) -> Option<RootType> {
    Some(match r {
        OurRoot::Constants => RootType::Constants,
        OurRoot::Table => RootType::Table,
        _ => return None,
    })
}

/// The program as the copy's exact wave64 emulation, for a host of any wave
/// size (no `WaveSize`; the lanes of a console wave are threads of the
/// group).
pub fn translate(container: &[u8]) -> Result<HostShader, String> {
    let options = ours::Options { view_marks: false, wave64: ours::Wave64Mode::Emulate };
    let s = ours::translate_with(container, options).map_err(|e| e.to_string())?;
    let flags = [
        (s.memory, "title memory"),
        (s.memory_writes, "title memory writes"),
        (s.view_addresses, "view addresses"),
        (s.view_marks, "write marks"),
    ];
    if let Some((_, what)) = flags.iter().find(|(on, _)| *on) {
        return Err(format!("needs {what}, which the host binds differently"));
    }
    // (Heap-picked descriptors are the host's own business there: it reads the title's root
    // layout, and binds the view-stride table `t255`, `SLOT_STRIDES`, as it does for its own
    // programs. `VICTORY_EXACT64_HEAP=0` keeps such programs out.)
    if (s.heap || s.heap_writes) && std::env::var_os("VICTORY_EXACT64_HEAP").is_some_and(|v| v == "0") {
        return Err("picks descriptors from the heap (VICTORY_EXACT64_HEAP=0)".into());
    }
    if s.root_srvs.iter().chain(&s.root_uavs).any(|&slot| slot >= 240 && slot != crate::emit::SLOT_STRIDES) {
        return Err("reads a host view table, which the host binds differently".into());
    }
    let mut user_data = Vec::with_capacity(s.user_data.len());
    for u in &s.user_data {
        let Some(root) = root(u.root) else {
            return Err("reads a root descriptor, which the host binds differently".into());
        };
        user_data.push(UserData { index: u.index, slot: u.slot, root, dword: u.dword });
    }
    let stage = match s.stage {
        super::native::HwStage::Cs => crate::native::HwStage::Cs,
        _ => return Err("not a compute program".into()),
    };
    Ok(HostShader {
        stage,
        dxil: s.dxil,
        user_data,
        samplers: s.samplers,
        static_samplers: s
            .static_samplers
            .iter()
            .map(|q| StaticSampler {
                filter: q.filter,
                address: q.address,
                mip_lod_bias: q.mip_lod_bias,
                max_anisotropy: q.max_anisotropy,
                comparison: q.comparison,
                border: q.border,
                min_lod: q.min_lod,
                max_lod: q.max_lod,
            })
            .collect(),
        root_cbvs: s.root_cbvs,
        root_srvs: s.root_srvs,
        root_uavs: s.root_uavs,
        root_indexed_views: Vec::new(),
        half_views: Vec::new(),
    })
}
