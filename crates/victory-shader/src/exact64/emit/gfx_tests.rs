//! Pixel-stage tests of this round: barycentrics read as data, and the
//! real containers (when `VICTORY_CONTAINER` names one) they came from.

use super::{Options, translate_with};

/// Translates the container at `VICTORY_CONTAINER` (a manual aid, ignored
/// by default) and prints the outcome.
#[test]
#[ignore]
fn translate_the_named_container() {
    let Ok(path) = std::env::var("VICTORY_CONTAINER") else {
        return;
    };
    let bytes = std::fs::read(&path).expect("read the container");
    match translate_with(&bytes, Options::default()) {
        Ok(h) => {
            let checked = match super::validator::Validator::load(
                super::validator::Validator::default_path().as_deref(),
            ) {
                Ok(mut v) => format!("{:?}", v.validate(&mut h.dxil.clone())),
                Err(e) => format!("no validator: {e:?}"),
            };
            println!("{path}: translated, {} bytes of DXIL, validator {checked}", h.dxil.len());
            if let Ok(c) = Container::parse(&h.dxil) {
                for (label, fourcc) in [("in", FourCC::ISG1), ("out", FourCC::OSG1), ("prim", FourCC::PSG1)] {
                    if let Some(part) = c.part(fourcc) {
                        for e in super::signature(part.data).unwrap_or_default() {
                            println!(
                                "  {label}: {} {} kind {} comp {} row {:?} col {} cols {}",
                                e.name, e.index, e.kind, e.comp, e.row, e.col, e.cols
                            );
                        }
                    }
                }
            }
        }
        Err(e) => println!("{path}: {e}"),
    }
}

use super::container::SigElement;
use super::{lower, validator};
use crate::exact64::dxbc::{Container, FourCC};
use crate::exact64::native::{HwStage, NativeProgram, StageIo};
use crate::exact64::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

fn exp_mrt0(v0: u8, v1: u8, v2: u8, v3: u8) -> [u32; 2] {
    // `exp mrt0 v0, v1, v2, v3 done vm`.
    [
        0xF800_0000 | (1 << 12) | (1 << 11) | 0xF,
        u32::from(v0) | (u32::from(v1) << 8) | (u32::from(v2) << 16) | (u32::from(v3) << 24),
    ]
}

/// `v_add_f32 dst, v_a, v_b`.
fn v_add_f32(dst: u32, a: u32, b: u32) -> u32 {
    (0x03 << 25) | (dst << 17) | (b << 9) | (256 + a)
}

/// A pixel shader with `input_addr` VGPR inputs that exports `v2` (the sum
/// of v0 and v1, the barycentrics of the first enabled set) in red.
fn barycentric_sum(input_addr: u32) -> Result<lower::Lowered, crate::ShaderError> {
    let mut code = vec![v_add_f32(2, 0, 1)];
    code.extend(exp_mrt0(2, 2, 2, 2));
    code.push(S_ENDPGM);
    let native = NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 8,
        layout: Vec::new(),
        used: Vec::new(),
        io: StageIo::Ps {
            inputs: Vec::new(),
            input_addr,
            db_shader_control: 0,
            patches: Vec::new(),
        },
        code: code.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let target = SigElement {
        name: "SV_Target".into(),
        index: 0,
        kind: 16,
        comp: 9,
        interp: 0,
        row: Some(0),
        col: 0,
        cols: 4,
        usage: 0,
        half: false,
    };
    lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: vec![target],
        prims: Vec::new(),
        mesh_stride: 0,
        options: Options::default(),
    })
}

#[test]
fn barycentrics_read_directly_come_from_sv_barycentrics() {
    // SPI_PS_INPUT_ADDR bit 1: perspective, center; bit 5: linear, center.
    for (addr, label) in [(1 << 1, "perspective center"), (1 << 5, "linear center")] {
        let l = barycentric_sum(addr).unwrap_or_else(|e| panic!("{label}: {e}"));
        match validator::Validator::load(validator::Validator::default_path().as_deref()) {
            Ok(mut v) => v
                .validate(&mut l.dxil.clone())
                .unwrap_or_else(|e| panic!("{label}: DXIL validation failed: {e:?}")),
            Err(e) => eprintln!("{label}: no DXIL validator ({e:?}): not validated"),
        }
        let c = Container::parse(&l.dxil).expect("container");
        let inputs = super::signature(c.part(FourCC::ISG1).expect("ISG1").data).expect("ISG1");
        let [b] = &inputs[..] else {
            panic!("{label}: {inputs:?}")
        };
        assert_eq!((b.name.as_str(), b.kind, b.cols), ("SV_Barycentrics", 28, 3));
        // The console's i and j are the weights of vertices 1 and 2: y, z.
        let isg1 = c.part(FourCC::ISG1).expect("ISG1").data;
        assert_eq!((isg1[8 + 24], isg1[8 + 25]), (0b111, 0b110), "{label}: mask, read");
        assert!(b.row.is_none(), "{label}: a special register, as DXC");
        // The feature bit D3D_SHADER_FEATURE_BARYCENTRICS.
        let sfi0 = c.part(FourCC::SFI0).expect("SFI0").data;
        let features = u64::from_le_bytes(sfi0.try_into().unwrap());
        assert_ne!(features & 0x2_0000, 0, "{label}: SFI0 {features:#x}");
    }
}

#[test]
fn barycentrics_read_directly_name_their_interpolation() {
    use crate::exact64::dxbc::u32_at;
    // PSV0 element records: the interpolation mode is byte 5 of 6 after
    // the name and index dwords.
    let mode = |addr: u32| -> u8 {
        let l = barycentric_sum(addr).expect("lower");
        let c = Container::parse(&l.dxil).expect("container");
        let psv = c.part(FourCC::PSV0).expect("PSV0").data;
        let size = u32_at(psv, 0).unwrap() as usize;
        let strings_at = 4 + size + 4 + 4 * 0;
        // Past the (empty) bindings dword, the strings and the indices:
        // find the element table by its record size, 16.
        let mut at = strings_at;
        let strings = u32_at(psv, at).unwrap() as usize;
        at += 4 + strings;
        let indices = u32_at(psv, at).unwrap() as usize;
        at += 4 + 4 * indices;
        assert_eq!(u32_at(psv, at).unwrap(), 16);
        psv[at + 4 + 8 + 5]
    };
    assert_eq!(mode(1 << 1), 2, "perspective center");
    assert_eq!(mode(1 << 2), 3, "perspective centroid");
    assert_eq!(mode(1 << 0), 6, "perspective sample");
    assert_eq!(mode(1 << 5), 4, "linear center");
    assert_eq!(mode(1 << 6), 5, "linear centroid");
    assert_eq!(mode(1 << 4), 7, "linear sample");
}

#[test]
fn pull_model_barycentrics_are_still_refused() {
    let Err(e) = barycentric_sum(1 << 3) else {
        panic!("translated pull-model barycentrics");
    };
    assert!(e.to_string().contains("pull-model"), "{e}");
}

// ---- Descriptor dwords read as data (upstream PR #2363) -------------------

use crate::exact64::native::{ComputeIo, ResourceType, RootParam, RootType, UsedResource};

const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

/// SMEM `s_load_dword*` (`op` 0, 1, 2: one, two, four dwords) at
/// `s[base:base+1] + offset`.
const fn s_load(op: u32, sdata: u8, sbase: u8, offset: u32) -> [u32; 2] {
    [
        0x3D << 26 | op << 18 | (sdata as u32) << 6 | (sbase as u32 >> 1),
        125 << 25 | offset,
    ]
}

const fn v_mov_b32_s(vdst: u8, sgpr: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | 0x01 << 9 | sgpr as u32
}

const fn buffer_store_dword(vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
    ]
}

/// A compute program that loads two dwords of descriptor 1 of its table
/// (byte offset `at` from the table: 0x20 is dword 0) and stores them as
/// numbers through the result view (descriptor 0).
fn reads_descriptor_dwords(at: u32) -> Result<lower::Lowered, crate::ShaderError> {
    let mut w = vec![s_mov_b32(5, 0x80)]; // s[4:5]: the table at s4
    w.extend(s_load(2, 8, 4, 0)); // s[8:11]: the result view
    w.extend(s_load(1, 16, 4, at)); // s[16:17]: two dwords of descriptor 1
    w.push(0xBF8C_0000); // s_waitcnt 0
    w.push(v_mov_b32_s(5, 16));
    w.push(v_mov_b32_s(7, 17));
    w.extend(buffer_store_dword(5, 8, 0));
    w.extend(buffer_store_dword(7, 8, 4));
    w.push(S_ENDPGM);
    let param = |root, user_data, slot, sgprs| RootParam {
        root,
        table_offset_dwords: 0,
        sgprs,
        base_register: 0,
        user_data,
        slot,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 10,
        layout: vec![param(RootType::Constants, 0, 0, 4), param(RootType::Table, 4, 1, 1)],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 8 << 1,
            threads: [1, 1, 1],
        }),
        code: w.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        options: Options::default(),
    })
}

#[test]
fn a_descriptors_record_count_and_format_are_read_from_the_host_table() {
    // Dwords 0-1: the address; 2-3: the record count and the format.
    for (at, label) in [(0x20, "address"), (0x28, "record count and format")] {
        let l = reads_descriptor_dwords(at).unwrap_or_else(|e| panic!("{label}: {e}"));
        assert!(l.view_addresses, "{label}: the host's view addresses are read");
        if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
            v.validate(&mut l.dxil.clone())
                .unwrap_or_else(|e| panic!("{label}: DXIL validation failed: {e:?}"));
        }
    }
}

#[test]
fn descriptor_dwords_past_the_host_table_are_refused_by_name() {
    // Dwords 4-5 and 6-7 are not in the table (four dwords a slot).
    for (at, dword) in [(0x30, 4), (0x38, 6)] {
        let Err(e) = reads_descriptor_dwords(at) else {
            panic!("dword {dword} translated");
        };
        let text = e.to_string();
        assert!(
            text.contains(&format!("a descriptor's dword {dword} read as data")),
            "{text}"
        );
    }
}
