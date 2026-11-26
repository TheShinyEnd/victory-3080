//! The exact wave64 translator: a self-contained copy of the VIctory emitter
//! whose console-wave lowering ("predicated/console flow" over upstream PR
//! #189's emulate mode) runs every block under a lane predicate with the lane
//! masks exchanged through group-shared memory. It shares nothing with the
//! cool_fox lowering around it (its own decoder, IR, resource tracker and
//! lowering, all under this module), and is entered only through
//! [`translate`], from `emit::emit`'s retry chain for a wave64 compute
//! program the cool_fox forms cannot lower exactly (`VICTORY_EXACT64=0` turns
//! it off).

pub mod bridge;
pub mod dxbc;
pub mod dxil;
pub mod emit;
pub mod gfx10;
pub mod io;
pub mod ir;
pub mod native;
pub mod range;
pub mod resource;
pub mod structure;

pub use dxbc::{Container, FourCC, Part};
pub use dxil::{DxilProgram, ShaderKind, XboxExtras};
pub use native::{
    HwStage, NativeProgram, ResourceType, RootParam, RootType, UsedResource, native_programs,
};
pub use crate::ShaderError;

/// What this copy is built from (it is part of the crate's own hash).
pub const SOURCE_HASH: &str = "exact64";
