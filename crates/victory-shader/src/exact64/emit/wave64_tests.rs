//! Tests for Wave64-on-Wave32 and portable LDS Wave64 compute lowering,
//! using synthetic GFX10 machine code on hardware GPU and WARP.

use super::container::{self, SigElement};
use super::{HostShader, Options, Wave64Mode, emit_with, translate_with};
use crate::exact64::dxbc::{Container, FourCC};
use crate::exact64::native::{
    ComputeIo, HwStage, MeshIo, NativeProgram, PsInput, RootParam, RootType, StageIo, Usage,
    VsOutput,
};
use crate::exact64::{ShaderError, io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
const EXEC_LO: u32 = 126;
const EXEC_HI: u32 = 127;
const VCC: u32 = 106;
const NULL_OP: u32 = 125;

// Inline constants and register operands in 9-bit GFX10 source fields.
fn vgpr(v: u32) -> u32 {
    256 + v
}
fn inline_u(n: u32) -> u32 {
    assert!(n <= 64);
    0x80 + n
}
const NEG_ONE: u32 = 0xC1;
const LIT: u32 = 0xFF;

fn sop1(dst: u32, op: u32, src0: u32) -> u32 {
    0xBE80_0000 | (dst << 16) | (op << 8) | src0
}
fn sop2(dst: u32, op: u32, src0: u32, src1: u32) -> u32 {
    0x8000_0000 | (op << 23) | (dst << 16) | (src1 << 8) | src0
}
fn sopc(op: u32, src0: u32, src1: u32) -> u32 {
    0xBF00_0000 | (op << 16) | (src1 << 8) | src0
}
fn sopp(op: u32, imm: i16) -> u32 {
    0xBF80_0000 | (op << 16) | u32::from(imm as u16)
}
fn vop1(dst: u32, op: u32, src0: u32) -> u32 {
    0x7E00_0000 | (dst << 17) | (op << 9) | src0
}
fn vop2(op: u32, dst: u32, src0: u32, vsrc1: u32) -> u32 {
    (op << 25) | (dst << 17) | (vsrc1 << 9) | src0
}
fn vopc(op: u32, src0: u32, vsrc1: u32) -> u32 {
    0x7C00_0000 | (op << 17) | (vsrc1 << 9) | src0
}
fn vop3(op: u32, dst: u32, src0: u32, src1: u32, src2: u32) -> [u32; 2] {
    [
        0xD400_0000 | (op << 16) | dst,
        src0 | (src1 << 9) | (src2 << 18),
    ]
}
fn ds_op(op: u32, off0: u8, off1: u8, addr: u32, data0: u32, vdst: u32) -> [u32; 2] {
    [
        0xD800_0000 | (op << 18) | (u32::from(off1) << 8) | u32::from(off0),
        addr | (data0 << 8) | (vdst << 24),
    ]
}
fn dpp_row_shr(
    vop2_op: u32,
    dst: u32,
    vsrc1: u32,
    vsrc0: u32,
    ctrl: u32,
    row_mask: u32,
) -> [u32; 2] {
    [
        vop2(vop2_op, dst, 0xFA, vsrc1),
        vsrc0 | (ctrl << 8) | (1 << 19) | (0xF << 24) | (row_mask << 28),
    ]
}

fn s_mov_b32(dst: u32, src0: u32) -> u32 {
    sop1(dst, 0x03, src0)
}
fn s_mov_b64(dst: u32, src0: u32) -> u32 {
    sop1(dst, 0x04, src0)
}
fn s_bcnt1_i32_b64(dst: u32, src0: u32) -> u32 {
    sop1(dst, 0x10, src0)
}
fn s_ff1_i32_b64(dst: u32, src0: u32) -> u32 {
    sop1(dst, 0x14, src0)
}
fn s_and_saveexec_b64(dst: u32, src0: u32) -> u32 {
    sop1(dst, 0x24, src0)
}
fn s_and_saveexec_b32(dst: u32, src0: u32) -> u32 {
    sop1(dst, 0x3C, src0)
}
fn s_or_saveexec_b64(dst: u32, src0: u32) -> u32 {
    sop1(dst, 0x25, src0)
}
fn s_andn2_saveexec_b64(dst: u32, src0: u32) -> u32 {
    sop1(dst, 0x27, src0)
}
fn s_add_i32(dst: u32, src0: u32, src1: u32) -> u32 {
    sop2(dst, 0x02, src0, src1)
}
fn s_or_b32(dst: u32, src0: u32, src1: u32) -> u32 {
    sop2(dst, 0x10, src0, src1)
}
fn s_or_b64(dst: u32, src0: u32, src1: u32) -> u32 {
    sop2(dst, 0x11, src0, src1)
}
fn s_lshl_b32(dst: u32, src0: u32, src1: u32) -> u32 {
    sop2(dst, 0x1E, src0, src1)
}
fn s_cmp_lt_u32(src0: u32, src1: u32) -> u32 {
    sopc(0x0A, src0, src1)
}
fn s_cbranch_scc1(delta_words: i16) -> u32 {
    sopp(0x05, delta_words)
}
fn s_cbranch_execz(delta_words: i16) -> u32 {
    sopp(0x08, delta_words)
}

fn v_mov_b32(dst: u32, src0: u32) -> u32 {
    vop1(dst, 0x01, src0)
}
fn v_cvt_f32_u32(dst: u32, src0: u32) -> u32 {
    vop1(dst, 0x06, src0)
}
fn v_readfirstlane_b32(sdst: u32, vsrc: u32) -> u32 {
    vop1(sdst, 0x02, vgpr(vsrc))
}
fn v_add_f32(dst: u32, src0: u32, vsrc1: u32) -> u32 {
    vop2(0x03, dst, src0, vsrc1)
}
fn v_sub_f32(dst: u32, src0: u32, vsrc1: u32) -> u32 {
    vop2(0x04, dst, src0, vsrc1)
}
fn v_lshrrev_b32(dst: u32, shift: u32, vsrc1: u32) -> u32 {
    vop2(0x16, dst, shift, vsrc1)
}
fn v_lshlrev_b32(dst: u32, shift: u32, vsrc1: u32) -> u32 {
    vop2(0x1A, dst, shift, vsrc1)
}
fn v_and_b32(dst: u32, src0: u32, vsrc1: u32) -> u32 {
    vop2(0x1B, dst, src0, vsrc1)
}
fn v_or_b32(dst: u32, src0: u32, vsrc1: u32) -> u32 {
    vop2(0x1C, dst, src0, vsrc1)
}
fn v_add_nc_u32(dst: u32, src0: u32, vsrc1: u32) -> u32 {
    vop2(0x25, dst, src0, vsrc1)
}
fn v_sub_nc_u32(dst: u32, src0: u32, vsrc1: u32) -> u32 {
    vop2(0x26, dst, src0, vsrc1)
}
fn v_subrev_nc_u32(dst: u32, src0: u32, vsrc1: u32) -> u32 {
    vop2(0x27, dst, src0, vsrc1)
}
fn v_cmp_le_f32(src0: u32, vsrc1: u32) -> u32 {
    vopc(0x03, src0, vsrc1)
}
fn v_cmp_lt_u32(src0: u32, vsrc1: u32) -> u32 {
    vopc(0xC1, src0, vsrc1)
}
fn v_cmp_eq_u32(src0: u32, vsrc1: u32) -> u32 {
    vopc(0xC2, src0, vsrc1)
}
fn v_cmp_gt_u32(src0: u32, vsrc1: u32) -> u32 {
    vopc(0xC4, src0, vsrc1)
}
fn v_interp_mov_f32(dst: u32, attr: u32, chan: u32, vertex: u32) -> u32 {
    0xC802_0000 | ((dst & 0xFF) << 18) | ((attr & 0x3F) << 10) | ((chan & 3) << 8) | (vertex & 0xFF)
}
#[allow(clippy::too_many_arguments)]
fn exp(target: u8, en: u8, done: bool, vm: bool, v0: u8, v1: u8, v2: u8, v3: u8) -> [u32; 2] {
    [
        0xF800_0000
            | (u32::from(vm) << 12)
            | (u32::from(done) << 11)
            | (u32::from(target) << 4)
            | u32::from(en & 0xF),
        u32::from(v0) | (u32::from(v1) << 8) | (u32::from(v2) << 16) | (u32::from(v3) << 24),
    ]
}
fn v_cndmask_b32(dst: u32, src0: u32, src1: u32, mask_s: u32) -> [u32; 2] {
    vop3(0x101, dst, src0, src1, mask_s)
}
fn v_readlane_b32(sdst: u32, vsrc: u32, lane_op: u32) -> [u32; 2] {
    vop3(0x360, sdst, vgpr(vsrc), lane_op, 0)
}
fn v_writelane_b32(vdst: u32, src: u32, lane_op: u32) -> [u32; 2] {
    vop3(0x361, vdst, src, lane_op, 0)
}
fn v_mbcnt_lo_u32_b32(dst: u32, mask: u32, b: u32) -> [u32; 2] {
    vop3(0x365, dst, mask, b, 0)
}
fn v_mbcnt_hi_u32_b32(dst: u32, mask: u32, b: u32) -> [u32; 2] {
    vop3(0x366, dst, mask, b, 0)
}
fn v_permlanex16_b32(dst: u32, vsrc0: u32, src1: u32, src2: u32) -> [u32; 2] {
    vop3(0x378, dst, vgpr(vsrc0), src1, src2)
}

fn s_load_dwordx4(sdst: u32, sbase: u32) -> [u32; 2] {
    [0xF408_0000 | (sdst << 6) | (sbase >> 1), NULL_OP << 25]
}

fn buffer_store_dword(vdata: u32, vaddr: u32, srsrc: u32) -> [u32; 2] {
    [
        0xE070_1000, // buffer_store_dword, offen=1
        vaddr | (vdata << 8) | ((srsrc >> 2) << 16) | (inline_u(0) << 24),
    ]
}

fn buffer_store_dwordx4(vdata: u32, vaddr: u32, srsrc: u32) -> [u32; 2] {
    [
        0xE078_1000, // buffer_store_dwordx4, offen=1
        vaddr | (vdata << 8) | ((srsrc >> 2) << 16) | (inline_u(0) << 24),
    ]
}

fn emit_uav_store(code: &mut Vec<u32>, data_vreg: u32, tid_vreg: u32) {
    // s0 holds heap base 0 (user param 0 = ResourceHeap); load view 0 into s[8:11].
    code.push(s_mov_b32(1, inline_u(0)));
    code.extend_from_slice(&s_load_dwordx4(8, 0));
    code.push(v_lshlrev_b32(15, inline_u(2), tid_vreg));
    code.extend_from_slice(&buffer_store_dword(data_vreg, 15, 8));
}

fn empty_dxbc() -> Vec<u8> {
    let mut b = Vec::with_capacity(32);
    b.extend_from_slice(b"DXBC");
    b.extend_from_slice(&[0u8; 16]);
    b.extend_from_slice(&1u32.to_le_bytes()); // major/minor
    b.extend_from_slice(&32u32.to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes());
    b
}

fn make_cs_program(
    words: &[u32],
    threads: [u32; 3],
    lds_bytes: u32,
    group_id_x: bool,
) -> NativeProgram {
    let lds_granules = lds_bytes.div_ceil(512);
    // 2 user SGPRs (s[0:1] heap pointer); optional TGID.x in bit 7.
    let rsrc2 = (2 << 1) | (u32::from(group_id_x) << 7) | (lds_granules << 15);
    NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 32,
        layout: vec![RootParam {
            root: RootType::ResourceHeap,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: Vec::new(),
        io: StageIo::Cs(ComputeIo { rsrc2, threads }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

fn lower_cs(
    native: &NativeProgram,
    mode: Wave64Mode,
) -> Result<super::lower::Lowered, ShaderError> {
    let p = ir::build(&native.patched_code(), native.wave64, native.vgprs)?;
    let st = structure::structure(&p);
    let res = resource::track(&p, &native.layout, native.stage, st.as_ref().ok())?;
    let io_res = io::raise(&p, native)?;
    let st = st?;
    super::lower::lower(&super::lower::Parts {
        native,
        p: &p,
        st: &st,
        io: &io_res,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        options: Options {
            wave64: mode,
            ..Default::default()
        },
    })
}

fn emit_cs(native: &NativeProgram, mode: Wave64Mode) -> Result<HostShader, ShaderError> {
    emit_program(&empty_dxbc(), native, mode)
}

fn emit_program(
    dxbc: &[u8],
    native: &NativeProgram,
    mode: Wave64Mode,
) -> Result<HostShader, ShaderError> {
    let container = Container::parse(dxbc)?;
    let p = ir::build(&native.patched_code(), native.wave64, native.vgprs)?;
    let st = structure::structure(&p);
    let res = resource::track(&p, &native.layout, native.stage, st.as_ref().ok())?;
    let io_res = io::raise(&p, native)?;
    let st = st?;
    emit_with(
        &container,
        native,
        &p,
        &st,
        &io_res,
        &res,
        Options {
            wave64: mode,
            ..Default::default()
        },
    )
}

#[derive(Clone, Copy)]
enum Lane0Guard {
    TidEqZeroBranch,
    ReadFirstLaneBranch,
    MbcntEqZeroBranch,
    TidEqZeroInline,
}

/// 64-thread (or custom thread-count) synthetic GFX10 float whole-wave reduction (`v_add_f32`)
/// + virtual lane-0 16B UAV write (`buffer_store_dwordx4` `[sum, 1.0, 2.0, 3.0]`).
///
/// When `half1_nonzero` is false: lanes `0..31 = float(tid + 1)`, lanes `32..63 = 0.0`.
/// When `half1_nonzero` is true: lanes `0..31 = 0.0`, lanes `32..63 = float(tid - 31)`.
/// Both sum to `528.0f32` (`1 + 2 + ... + 32`).
fn whole_wave_float_sum_uav_x4_program(
    threads: [u32; 3],
    lds_bytes: u32,
    half1_nonzero: bool,
    guard: Lane0Guard,
) -> NativeProgram {
    let mut code = Vec::new();
    if half1_nonzero {
        code.push(v_cmp_lt_u32(inline_u(31), 0)); // vcc = (31 < v0) i.e. tid >= 32
        code.push(v_subrev_nc_u32(3, inline_u(31), 0)); // v3 = tid - 31
    } else {
        code.push(v_cmp_gt_u32(inline_u(32), 0)); // vcc = (32 > v0) i.e. tid < 32
        code.push(v_add_nc_u32(3, inline_u(1), 0)); // v3 = tid + 1
    }
    code.push(v_cvt_f32_u32(3, vgpr(3)));
    code.extend_from_slice(&v_cndmask_b32(2, inline_u(0), vgpr(3), VCC));

    // Whole-wave float reduction section (`v2` -> `v10`):
    // 1. intra-row prefix sum via DPP row_shr:{1,2,4,8}
    // 2. cross-row merge within each 32-lane half via `v_permlanex16_b32` + `v_add_f32`
    // 3. cross-half merge via `v_readlane_b32` at lanes 15 and 63 + `v_add_f32`
    code.push(s_or_saveexec_b64(4, NEG_ONE));
    code.extend_from_slice(&v_cndmask_b32(10, inline_u(0), vgpr(2), 4));
    for shift in [0x111u32, 0x112, 0x114, 0x118] {
        code.extend_from_slice(&dpp_row_shr(0x03, 10, 10, 10, shift, 0xF));
    }
    code.extend_from_slice(&v_permlanex16_b32(11, 10, NEG_ONE, NEG_ONE));
    code.push(v_add_f32(10, vgpr(10), 11));
    code.extend_from_slice(&v_readlane_b32(12, 10, inline_u(15)));
    code.extend_from_slice(&v_readlane_b32(13, 10, inline_u(63)));
    code.push(v_mov_b32(10, 12));
    code.push(v_add_f32(10, 13, 10));
    code.push(s_mov_b64(EXEC_LO, 4));

    // Load UAV 0 descriptor (`s[8:11]`) and prepare 16-byte payload `[sum, 1.0, 2.0, 3.0]`.
    code.push(s_mov_b32(1, inline_u(0)));
    code.extend_from_slice(&s_load_dwordx4(8, 0));
    code.push(v_mov_b32(16, vgpr(10)));
    for (reg, f) in [(17u32, 1.0f32), (18, 2.0), (19, 3.0)] {
        code.push(v_mov_b32(reg, LIT));
        code.push(f.to_bits());
    }
    code.push(v_mov_b32(15, inline_u(0)));

    match guard {
        Lane0Guard::TidEqZeroBranch => {
            code.push(v_cmp_eq_u32(inline_u(0), 0));
            code.push(s_and_saveexec_b64(6, VCC));
            let br_idx = code.len();
            code.push(0);
            code.extend_from_slice(&buffer_store_dwordx4(16, 15, 8));
            let after = code.len();
            code[br_idx] = s_cbranch_execz((after - (br_idx + 1)) as i16);
            code.push(s_or_b64(EXEC_LO, EXEC_LO, 6));
        }
        Lane0Guard::ReadFirstLaneBranch => {
            code.push(v_readfirstlane_b32(14, 0));
            code.push(v_cmp_eq_u32(14, 0));
            code.push(s_and_saveexec_b64(6, VCC));
            let br_idx = code.len();
            code.push(0);
            code.extend_from_slice(&buffer_store_dwordx4(16, 15, 8));
            let after = code.len();
            code[br_idx] = s_cbranch_execz((after - (br_idx + 1)) as i16);
            code.push(s_or_b64(EXEC_LO, EXEC_LO, 6));
        }
        Lane0Guard::MbcntEqZeroBranch => {
            code.extend_from_slice(&v_mbcnt_lo_u32_b32(14, NEG_ONE, inline_u(0)));
            code.extend_from_slice(&v_mbcnt_hi_u32_b32(14, NEG_ONE, vgpr(14)));
            code.push(v_cmp_eq_u32(inline_u(0), 14));
            code.push(s_and_saveexec_b64(6, VCC));
            let br_idx = code.len();
            code.push(0);
            code.extend_from_slice(&buffer_store_dwordx4(16, 15, 8));
            let after = code.len();
            code[br_idx] = s_cbranch_execz((after - (br_idx + 1)) as i16);
            code.push(s_or_b64(EXEC_LO, EXEC_LO, 6));
        }
        Lane0Guard::TidEqZeroInline => {
            code.push(v_cmp_eq_u32(inline_u(0), 0));
            code.push(s_and_saveexec_b64(6, VCC));
            code.extend_from_slice(&buffer_store_dwordx4(16, 15, 8));
            code.push(s_or_b64(EXEC_LO, EXEC_LO, 6));
        }
    }
    code.push(S_ENDPGM);
    make_cs_program(&code, threads, lds_bytes, false)
}

fn psv_wave_range(shader: &HostShader) -> (u32, u32) {
    let c = Container::parse(&shader.dxil).unwrap();
    let psv = c.part(FourCC::PSV0).unwrap().data;
    let min = u32::from_le_bytes(psv[20..24].try_into().unwrap());
    let max = u32::from_le_bytes(psv[24..28].try_into().unwrap());
    (min, max)
}

fn sfi0_flags(shader: &HostShader) -> u64 {
    let c = Container::parse(&shader.dxil).unwrap();
    match c.part(FourCC::SFI0) {
        Some(p) => u64::from_le_bytes(p.data[0..8].try_into().unwrap()),
        None => 0,
    }
}

#[test]
fn wave_mode_psv_and_flags() {
    // 128 threads reading 64-lane ballot + mbcnt.
    let mut code = Vec::new();
    code.push(v_cmp_lt_u32(inline_u(40), 0)); // vcc = (40 < v0) i.e. tid & 63 >= 41
    code.push(s_bcnt1_i32_b64(4, VCC));
    code.extend_from_slice(&v_mbcnt_lo_u32_b32(1, NEG_ONE, inline_u(0)));
    code.extend_from_slice(&v_mbcnt_hi_u32_b32(1, NEG_ONE, vgpr(1)));
    code.push(v_add_nc_u32(1, 4, 1));
    emit_uav_store(&mut code, 1, 0);
    code.push(S_ENDPGM);

    let prog = make_cs_program(&code, [128, 1, 1], 0, false);

    let native = emit_cs(&prog, Wave64Mode::Native).unwrap();
    assert_eq!(psv_wave_range(&native), (64, 64));
    assert_ne!(sfi0_flags(&native) & 0x4000, 0);

    let wave32 = emit_cs(&prog, Wave64Mode::Wave32).unwrap();
    assert_eq!(psv_wave_range(&wave32), (32, 32));

    let emu = emit_cs(&prog, Wave64Mode::Emulate).unwrap();
    assert_eq!(psv_wave_range(&emu), (0, u32::MAX));
    assert_eq!(sfi0_flags(&emu) & 0x4000, 0);
}

#[test]
fn readfirstlane_psv_wave_size_without_mbcnt() {
    // Non-uniform v_readfirstlane_b32 without v_mbcnt/v_readlane/dpp must still enforce
    // WaveSize(64) in Wave64Mode::Native so 64-lane waves are not split into Wave32.
    let mut non_uniform_code = vec![v_readfirstlane_b32(4, 0), v_mov_b32(1, 4)];
    emit_uav_store(&mut non_uniform_code, 1, 0);
    non_uniform_code.push(S_ENDPGM);
    let non_uniform_prog = make_cs_program(&non_uniform_code, [64, 1, 1], 0, false);

    let native = emit_cs(&non_uniform_prog, Wave64Mode::Native).unwrap();
    assert_eq!(psv_wave_range(&native), (64, 64));
    assert_ne!(sfi0_flags(&native) & 0x4000, 0);

    let wave32 = emit_cs(&non_uniform_prog, Wave64Mode::Wave32).unwrap();
    assert_eq!(psv_wave_range(&wave32), (32, 32));

    let emu = emit_cs(&non_uniform_prog, Wave64Mode::Emulate).unwrap();
    assert_eq!(psv_wave_range(&emu), (0, u32::MAX));
    assert_eq!(sfi0_flags(&emu) & 0x4000, 0);

    // Uniform v_readfirstlane_b32 without other cross-lane ops does not need WaveSize(64)
    // in Native or ballot scratch in Wave32/Emulate.
    let mut uniform_code = vec![
        s_mov_b32(8, inline_u(42)),
        v_mov_b32(2, 8),
        v_readfirstlane_b32(4, 2),
        v_mov_b32(1, 4),
    ];
    emit_uav_store(&mut uniform_code, 1, 0);
    uniform_code.push(S_ENDPGM);
    let uniform_prog = make_cs_program(&uniform_code, [64, 1, 1], 0, false);

    let uniform_native = emit_cs(&uniform_prog, Wave64Mode::Native).unwrap();
    assert_eq!(psv_wave_range(&uniform_native), (0, u32::MAX));
    assert_eq!(
        lower_cs(&uniform_prog, Wave64Mode::Wave32)
            .unwrap()
            .lds_dwords,
        0
    );
    assert_eq!(
        lower_cs(&uniform_prog, Wave64Mode::Emulate)
            .unwrap()
            .lds_dwords,
        0
    );
}

#[test]
fn lds_overflow_fails_explicitly() {
    // Title uses full 32 KiB (8192 dwords) of LDS and reads a dynamic lane.
    let mut code = Vec::new();
    code.extend_from_slice(&v_readlane_b32(4, 0, vgpr(0)));
    code.push(v_mov_b32(1, 4));
    emit_uav_store(&mut code, 1, 0);
    code.push(S_ENDPGM);

    let prog = make_cs_program(&code, [1024, 1, 1], 32768, false);
    assert!(emit_cs(&prog, Wave64Mode::Native).is_ok());

    for mode in [Wave64Mode::Wave32, Wave64Mode::Emulate] {
        let err = emit_cs(&prog, mode).unwrap_err();
        assert!(
            matches!(err, ShaderError::Unsupported { what, .. } if what.contains("32 KiB")),
            "expected explicit LDS overflow error in {mode:?}, got {err:?}"
        );
    }
}

#[test]
fn divergent_cross_half_ballot_is_rejected_instead_of_hanging() {
    // if (v0 < 32) { s4 = bcnt1(v0 < 10); v1 = s4; } else { v1 = 0; } store(v1)
    let mut code = Vec::new();
    code.push(v_mov_b32(1, inline_u(0)));
    code.push(v_cmp_gt_u32(inline_u(32), 0)); // 32 > v0
    code.push(s_and_saveexec_b64(6, VCC)); // s[6:7]
    let br_idx = code.len();
    code.push(0); // placeholder for s_cbranch_execz -> after_then
    code.push(v_cmp_gt_u32(inline_u(10), 0)); // 10 > v0
    code.push(s_bcnt1_i32_b64(4, VCC));
    code.push(v_mov_b32(1, 4));
    let after_then = code.len();
    code[br_idx] = s_cbranch_execz((after_then - (br_idx + 1)) as i16);
    code.push(s_or_b64(EXEC_LO, EXEC_LO, 6));
    emit_uav_store(&mut code, 1, 0);
    code.push(S_ENDPGM);

    let prog = make_cs_program(&code, [128, 1, 1], 0, false);
    assert!(emit_cs(&prog, Wave64Mode::Native).is_ok());
    let err = emit_cs(&prog, Wave64Mode::Wave32).unwrap_err();
    assert!(
        matches!(err, ShaderError::Unsupported { .. }),
        "expected safe error for divergent ballot in Wave32, got {err:?}"
    );
    // Emulate runs the branch in every thread, so every thread reaches the
    // exchange's barriers.
    let shader = emit_cs(&prog, Wave64Mode::Emulate).expect("predicated ballot");
    check_dxil(&shader, "divergent ballot");
}

#[test]
fn divergent_readfirstlane_reads_its_halfs_first_lane_on_wave32() {
    // if (v0 < 32) { s4 = readfirstlane(v0); v1 = s4; } else { v1 = 0; } store(v1)
    // Inside divergent flow the halves cannot meet at a barrier: Wave32 reads
    // each half's own first active lane; Emulate runs every thread through
    // every block (predicated flow) so that the barriers are reached by all.
    let mut code = Vec::new();
    code.push(v_mov_b32(1, inline_u(0)));
    code.push(v_cmp_gt_u32(inline_u(32), 0)); // 32 > v0
    code.push(s_and_saveexec_b64(6, VCC)); // s[6:7]
    let br_idx = code.len();
    code.push(0); // placeholder for s_cbranch_execz -> after_then
    code.push(v_readfirstlane_b32(4, 0));
    code.push(v_mov_b32(1, 4));
    let after_then = code.len();
    code[br_idx] = s_cbranch_execz((after_then - (br_idx + 1)) as i16);
    code.push(s_or_b64(EXEC_LO, EXEC_LO, 6));
    emit_uav_store(&mut code, 1, 0);
    code.push(S_ENDPGM);

    let prog = make_cs_program(&code, [128, 1, 1], 0, false);
    assert!(emit_cs(&prog, Wave64Mode::Native).is_ok());
    let wave32 = emit_cs(&prog, Wave64Mode::Wave32).unwrap();
    assert_eq!(psv_wave_range(&wave32), (32, 32));
    let emulate = emit_cs(&prog, Wave64Mode::Emulate).unwrap();
    check_dxil(&emulate, "divergent readfirstlane (Emulate)");
}

/// Runs dxil.dll's validator on `shader` where it is found.
fn check_dxil(shader: &HostShader, name: &str) {
    if let Ok(mut v) = super::validator::Validator::load(
        super::validator::Validator::default_path().as_deref(),
    ) {
        v.validate(&mut shader.dxil.clone())
            .unwrap_or_else(|e| panic!("{name} validate: {e}"));
    }
}

/// `v1 = 1000; if (lane > 5 + (wave << shift)) { s10 = readfirstlane(lane * 8
/// + wave); v1 = s10 + lane; } store(v1)` over `threads` threads (lane: the
/// thread's place in its console wave of 64, wave: the console wave).
fn divergent_readfirstlane_if_program(threads: u32, shift: u32) -> NativeProgram {
    let mut code = vec![
        v_and_b32(2, inline_u(63), 0),       // v2 = lane
        v_lshrrev_b32(3, inline_u(6), 0),    // v3 = wave
        v_lshlrev_b32(4, inline_u(3), 2),    // v4 = lane * 8
        v_add_nc_u32(4, vgpr(3), 4),         // v4 = lane * 8 + wave
        v_lshlrev_b32(5, inline_u(shift), 3), // v5 = wave << shift
        v_add_nc_u32(5, inline_u(5), 5),     // v5 = threshold
        v_mov_b32(1, LIT),
        1000,
        v_cmp_lt_u32(vgpr(5), 2), // threshold < lane
        s_and_saveexec_b64(6, VCC),
    ];
    let br = code.len();
    code.push(0);
    code.push(v_readfirstlane_b32(10, 4));
    code.push(v_add_nc_u32(1, 10, 2));
    let after = code.len();
    code[br] = s_cbranch_execz((after - (br + 1)) as i16);
    code.push(s_or_b64(EXEC_LO, EXEC_LO, 6));
    emit_uav_store(&mut code, 1, 0);
    code.push(S_ENDPGM);
    make_cs_program(&code, [threads, 1, 1], 0, false)
}

fn divergent_readfirstlane_if_expected(threads: u32, shift: u32) -> Vec<u32> {
    (0..threads)
        .map(|t| {
            let (wave, lane) = (t >> 6, t & 63);
            let threshold = 5 + (wave << shift);
            if lane > threshold {
                // The first lane above the threshold is the first active one.
                (threshold + 1) * 8 + wave + lane
            } else {
                1000
            }
        })
        .collect()
}

/// A carry out of `v_add_co_u32` (a lane mask, here in s[12:13]) and a carry
/// in (`v_add_co_ci_u32`) inside a branch that also reads `v_readfirstlane`:
/// `if (lane > 5 + (wave << 2)) { s10 = readfirstlane(lane * 8 + wave);
/// (v8, c) = lane + 0xFFFFFFE0; v9 = c ? 20 : 10; v10 = v8 + v8 + c;
/// v1 = s10 + lane + v9 + v10 } else { v1 = 1000 }`. A mask defined by an
/// instruction the per-thread form refuses must report that reason (the
/// predicated form takes the program), not "a lane mask from a carry".
fn carry_mask_in_divergent_readfirstlane_program(threads: u32) -> NativeProgram {
    // v_add_co_u32 / v_add_co_ci_u32 (VOP3B): the carry-out SGPR is bits 14:8.
    let with_sdst = |mut w: [u32; 2], sdst: u32| {
        w[0] |= sdst << 8;
        w
    };
    let mut code = vec![
        v_and_b32(2, inline_u(63), 0),       // v2 = lane
        v_lshrrev_b32(3, inline_u(6), 0),    // v3 = wave
        v_lshlrev_b32(4, inline_u(3), 2),    // v4 = lane * 8
        v_add_nc_u32(4, vgpr(3), 4),         // v4 = lane * 8 + wave
        v_lshlrev_b32(5, inline_u(2), 3),    // v5 = wave << 2
        v_add_nc_u32(5, inline_u(5), 5),     // v5 = threshold
        v_mov_b32(1, LIT),
        1000,
        v_mov_b32(7, LIT),
        0xFFFF_FFE0,
        v_cmp_lt_u32(vgpr(5), 2), // threshold < lane
        s_and_saveexec_b64(6, VCC),
    ];
    let br = code.len();
    code.push(0);
    code.push(v_readfirstlane_b32(10, 4));
    code.extend_from_slice(&with_sdst(vop3(0x30F, 8, vgpr(2), vgpr(7), 0), 12));
    code.extend_from_slice(&v_cndmask_b32(9, inline_u(10), inline_u(20), 12));
    code.extend_from_slice(&with_sdst(vop3(0x128, 11, vgpr(8), vgpr(8), 12), 14));
    code.push(v_add_nc_u32(1, 10, 2)); // v1 = s10 + lane
    code.push(v_add_nc_u32(1, vgpr(9), 1));
    code.push(v_add_nc_u32(1, vgpr(11), 1));
    let after = code.len();
    code[br] = s_cbranch_execz((after - (br + 1)) as i16);
    code.push(s_or_b64(EXEC_LO, EXEC_LO, 6));
    emit_uav_store(&mut code, 1, 0);
    code.push(S_ENDPGM);
    make_cs_program(&code, [threads, 1, 1], 0, false)
}

fn carry_mask_in_divergent_readfirstlane_expected(threads: u32) -> Vec<u32> {
    (0..threads)
        .map(|t| {
            let (wave, lane) = (t >> 6, t & 63);
            let threshold = 5 + (wave << 2);
            if lane > threshold {
                let v8 = lane.wrapping_add(0xFFFF_FFE0);
                let carry = u32::from(lane >= 32);
                let v9 = if carry != 0 { 20 } else { 10 };
                let v11 = v8.wrapping_add(v8).wrapping_add(carry);
                ((threshold + 1) * 8 + wave + lane)
                    .wrapping_add(v9)
                    .wrapping_add(v11)
            } else {
                1000
            }
        })
        .collect()
}

#[test]
fn a_carry_mask_beside_a_divergent_readfirstlane_lowers_in_emulate() {
    for threads in [64, 128] {
        let prog = carry_mask_in_divergent_readfirstlane_program(threads);
        let shader = emit_cs(&prog, Wave64Mode::Emulate)
            .unwrap_or_else(|e| panic!("{threads} threads: {e}"));
        check_dxil(&shader, "carry mask beside a divergent readfirstlane");
    }
}

// ---- 16-bit and 64-bit float wave sums, and 64-bit float instructions ----

fn vop1_dpp(dst: u32, op: u32, src: u32, ctrl: u32) -> [u32; 2] {
    [
        vop1(dst, op, 0xFA),
        src | (ctrl << 8) | (1 << 19) | (0xF << 24) | (0xF << 28),
    ]
}

/// An f16 as the console's `v_add_f16` makes it: nearest even.
fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xFF) as i32 - 127 + 15;
    let m = b & 0x7F_FFFF;
    if (b & 0x7FFF_FFFF) == 0 {
        return sign;
    }
    if e >= 31 {
        return sign | 0x7C00;
    }
    if e <= 0 {
        // Subnormal halves (not reached by the values used here).
        if e < -10 {
            return sign;
        }
        let m = m | 0x80_0000;
        let shift = (14 - e) as u32;
        let (q, r, half) = (m >> shift, m & ((1 << shift) - 1), 1u32 << (shift - 1));
        let q = q + u32::from(r > half || (r == half && q & 1 == 1));
        return sign | q as u16;
    }
    let (q, r) = (m >> 13, m & 0x1FFF);
    let mut h = ((e as u32) << 10) | q;
    if r > 0x1000 || (r == 0x1000 && q & 1 == 1) {
        h += 1;
    }
    sign | h as u16
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let (s, e, m) = (u32::from(h >> 15), u32::from(h >> 10) & 31, u32::from(h) & 0x3FF);
    let v = if e == 0 {
        m as f32 * 2f32.powi(-24)
    } else {
        (1.0 + m as f32 / 1024.0) * 2f32.powi(e as i32 - 15)
    };
    if s == 1 { -v } else { v }
}

fn add_f16_bits(a: u16, b: u16) -> u16 {
    f32_to_f16_bits(f16_bits_to_f32(a) + f16_bits_to_f32(b))
}

/// `WaveActiveSum(half)` as the console expands it (the instructions of
/// `CS_TransposeVecMatMulAlignedOperation`): lanes below `active` run, each
/// holding `800 + tid` as an f16 in the low half of a register whose high
/// half is 0xABCD; the sum is read from lane 0 (whose high half the result
/// keeps) and stored by every lane.
fn half_wave_sum_program(threads: u32, active: u32) -> NativeProgram {
    let mut code = vec![
        v_mov_b32(11, LIT),
        0xABCD_0000,
        v_cvt_f32_u32(3, vgpr(0)),
        v_add_f32(3, LIT, 3),
        800.0f32.to_bits(),
        vop1(11, 0x0A, vgpr(3)), // v_cvt_f16_f32 v11 (low half)
        v_and_b32(2, inline_u(63), 0),
        v_cmp_gt_u32(inline_u(active), 2),
        s_and_saveexec_b64(6, VCC),
        s_or_saveexec_b64(22, NEG_ONE),
    ];
    code.extend_from_slice(&v_cndmask_b32(22, inline_u(0), vgpr(11), 22));
    for ctrl in [0xB1u32, 0x4E, 0x141, 0x140] {
        code.extend_from_slice(&dpp_row_shr(0x32, 22, 22, 22, ctrl, 0xF));
    }
    code.extend_from_slice(&v_permlanex16_b32(23, 22, inline_u(0), inline_u(0)));
    code.push(vop2(0x32, 22, vgpr(22), 23));
    code.extend_from_slice(&v_readlane_b32(13, 22, inline_u(32)));
    code.push(vop2(0x32, 22, 13, 22));
    code.extend_from_slice(&v_readlane_b32(13, 22, inline_u(0)));
    code.push(s_mov_b64(EXEC_LO, 22));
    code.push(s_or_b64(EXEC_LO, EXEC_LO, 6));
    code.push(v_mov_b32(1, 13));
    emit_uav_store(&mut code, 1, 0);
    code.push(S_ENDPGM);
    make_cs_program(&code, [threads, 1, 1], 0, false)
}

/// The console's order: adjacent lanes first (blocks of 2, 4, ... 64), one
/// f16 rounding per addition.
fn half_wave_sum_expected(threads: u32, active: u32) -> Vec<u32> {
    (0..threads)
        .map(|t| {
            let wave = t >> 6;
            let x = |l: u32| -> u32 {
                if l < active {
                    0xABCD_0000 | u32::from(f32_to_f16_bits(800.0 + (wave * 64 + l) as f32))
                } else {
                    0
                }
            };
            let mut v: Vec<u16> = (0..64).map(|l| x(l) as u16).collect();
            let mut stride = 1;
            while stride < 64 {
                for l in (0..64).step_by(2 * stride) {
                    v[l] = add_f16_bits(v[l], v[l + stride]);
                }
                stride *= 2;
            }
            (x(0) & 0xFFFF_0000) | u32::from(v[0])
        })
        .collect()
}

#[test]
fn a_half_wave_sum_lowers_in_emulate_in_the_consoles_order() {
    // (The sum is order-dependent: the tree is checked on the GPU below.)
    for (threads, active) in [(64, 64), (64, 50), (128, 64)] {
        let shader = emit_cs(&half_wave_sum_program(threads, active), Wave64Mode::Emulate)
            .unwrap_or_else(|e| panic!("{threads} threads, {active} active: {e}"));
        check_dxil(&shader, "half wave sum");
    }
}

#[test]
fn a_half_wave_sum_over_a_partial_wave_is_refused() {
    // 32 threads: the console runs the section on all 64 lanes, the
    // missing ones holding whatever their registers hold.
    let r = emit_cs(&half_wave_sum_program(32, 32), Wave64Mode::Emulate);
    assert!(
        matches!(&r, Err(ShaderError::Unsupported { what, .. })
            if what.contains("16-bit wave reduction over a partial")),
        "{:?}",
        r.err()
    );
}

/// `WaveActiveSum(double)` of `lane < active ? K : 0.0` (K's high dword
/// `k_hi`, its low dword 0) as the console expands it, moved to SGPRs and
/// converted: every lane stores the f32 of the sum.
fn double_wave_sum_program(threads: u32, active: u32, k_hi: u32) -> NativeProgram {
    let mut code = vec![
        v_mov_b32(19, inline_u(0)),
        v_mov_b32(20, LIT),
        k_hi,
        v_and_b32(6, inline_u(63), 0),
        v_cmp_gt_u32(inline_u(active), 6),
        s_and_saveexec_b64(24, VCC),
        s_or_saveexec_b64(26, NEG_ONE),
        v_mov_b32(2, inline_u(0)),
        v_mov_b32(3, inline_u(0)),
    ];
    code.extend_from_slice(&v_cndmask_b32(2, vgpr(2), vgpr(19), 26));
    code.extend_from_slice(&v_cndmask_b32(3, vgpr(3), vgpr(20), 26));
    for ctrl in [0xB1u32, 0x4E, 0x141, 0x140] {
        code.extend_from_slice(&vop1_dpp(4, 1, 2, ctrl));
        code.extend_from_slice(&vop1_dpp(5, 1, 3, ctrl));
        code.extend_from_slice(&vop3(0x164, 2, vgpr(4), vgpr(2), 0));
    }
    code.extend_from_slice(&v_permlanex16_b32(4, 2, inline_u(0), inline_u(0)));
    code.extend_from_slice(&v_permlanex16_b32(5, 3, inline_u(0), inline_u(0)));
    code.extend_from_slice(&vop3(0x164, 2, vgpr(4), vgpr(2), 0));
    code.extend_from_slice(&v_readlane_b32(8, 2, inline_u(32)));
    code.extend_from_slice(&v_readlane_b32(9, 3, inline_u(32)));
    code.extend_from_slice(&vop3(0x164, 2, 8, vgpr(2), 0));
    code.extend_from_slice(&v_readlane_b32(8, 2, inline_u(0)));
    code.extend_from_slice(&v_readlane_b32(9, 3, inline_u(0)));
    code.push(s_mov_b32(12, 8));
    code.push(s_mov_b32(13, 9));
    code.push(s_mov_b64(EXEC_LO, 26));
    code.push(s_or_b64(EXEC_LO, EXEC_LO, 24));
    code.push(vop1(1, 0x0F, 12)); // v_cvt_f32_f64 v1, s[12:13]
    emit_uav_store(&mut code, 1, 0);
    code.push(S_ENDPGM);
    make_cs_program(&code, [threads, 1, 1], 0, false)
}

fn double_wave_sum_expected(threads: u32, active: u32, k_hi: u32) -> Vec<u32> {
    let k = f64::from_bits(u64::from(k_hi) << 32);
    let r = (f64::from(active.min(64)) * k) as f32;
    // (Lanes of a wave beyond 64 threads are in another wave.)
    vec![r.to_bits(); threads as usize]
}

#[test]
fn a_double_wave_sum_of_exact_terms_lowers_in_emulate() {
    for (threads, active, k_hi) in [(64, 64, 0x3FF0_0000), (64, 50, 0x3FD8_0000), (128, 0, 0x3FF0_0000)] {
        let shader = emit_cs(&double_wave_sum_program(threads, active, k_hi), Wave64Mode::Emulate)
            .unwrap_or_else(|e| panic!("{threads} threads, {active} active: {e}"));
        check_dxil(&shader, "double wave sum");
        assert_eq!(sfi0_flags(&shader) & 0x21, 0x21, "Doubles, and extensions for the integer conversion");
    }
}

#[test]
fn a_double_wave_sum_of_terms_that_round_is_refused() {
    // 0.1 (0x3FB99999 9999999A has low mantissa bits set; here the high
    // dword alone with a low dword of 0 stays exact, so use a literal low
    // dword of 1 to make n * K inexact).
    let mut prog = double_wave_sum_program(64, 64, 0x3FF0_0000);
    // v19 (K's low dword) = 1: `v_mov_b32 v19, 1` instead of 0.
    let mut words: Vec<u32> = prog
        .code
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    words[0] = v_mov_b32(19, inline_u(1));
    prog.code = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let r = emit_cs(&prog, Wave64Mode::Emulate);
    assert!(
        matches!(&r, Err(ShaderError::Unsupported { what, .. })
            if what.contains("double wave sum of terms that round")),
        "{:?}",
        r.err()
    );
}

/// 64-bit float instructions outside a wave section: `v[4:5] = double(tid)`,
/// `* 0.5`, `+ 2^30`, `- 2^30` (exact in f64, not in f32), to f32, widened
/// again, `+ 1.0`, narrowed: stores `tid * 0.5 + 1.0`.
fn f64_valu_program(threads: u32) -> NativeProgram {
    let mut code = vec![vop1(4, 0x16, vgpr(0))]; // v_cvt_f64_u32 v[4:5], v0
    code.extend_from_slice(&vop3(0x165, 6, vgpr(4), 240, 0)); // v_mul_f64 v[6:7], v[4:5], 0.5
    code.push(v_mov_b32(10, inline_u(0)));
    code.push(v_mov_b32(11, LIT));
    code.push(0x41D0_0000); // 2^30
    code.extend_from_slice(&vop3(0x164, 8, vgpr(6), vgpr(10), 0)); // + 2^30
    let mut sub = vop3(0x164, 8, vgpr(8), vgpr(10), 0); // - 2^30
    sub[1] |= 1 << 30; // neg src1
    code.extend_from_slice(&sub);
    code.push(vop1(1, 0x0F, vgpr(8))); // v_cvt_f32_f64 v1, v[8:9]
    code.push(vop1(12, 0x10, vgpr(1))); // v_cvt_f64_f32 v[12:13], v1
    code.extend_from_slice(&vop3(0x164, 14, vgpr(12), 242, 0)); // + 1.0
    code.push(vop1(1, 0x0F, vgpr(14))); // v_cvt_f32_f64 v1, v[14:15]
    emit_uav_store(&mut code, 1, 0);
    code.push(S_ENDPGM);
    make_cs_program(&code, [threads, 1, 1], 0, false)
}

fn f64_valu_expected(threads: u32) -> Vec<u32> {
    (0..threads).map(|t| (t as f32 * 0.5 + 1.0).to_bits()).collect()
}

#[test]
fn f64_instructions_lower_and_set_the_double_flags() {
    for mode in [Wave64Mode::Native, Wave64Mode::Emulate] {
        let shader = emit_cs(&f64_valu_program(64), mode).unwrap();
        check_dxil(&shader, "f64 instructions");
        assert_eq!(sfi0_flags(&shader) & 0x21, 0x21, "Doubles, and extensions for v_cvt_f64_u32");
    }
}

/// `if (lane > 2) { s10 = readfirstlane(lane * 8 + wave); v1 = s10 + lane;
/// store(v1) }`: the lanes that return early must not leave the group
/// barriers of those that stay.
fn divergent_readfirstlane_early_return_program(threads: u32) -> NativeProgram {
    let mut code = vec![
        v_and_b32(2, inline_u(63), 0),    // v2 = lane
        v_lshrrev_b32(3, inline_u(6), 0), // v3 = wave
        v_lshlrev_b32(4, inline_u(3), 2), // v4 = lane * 8
        v_add_nc_u32(4, vgpr(3), 4),      // v4 = lane * 8 + wave
        v_cmp_lt_u32(inline_u(2), 2),     // 2 < lane
        s_and_saveexec_b64(6, VCC),
    ];
    let br = code.len();
    code.push(0);
    code.push(v_readfirstlane_b32(10, 4));
    code.push(v_add_nc_u32(1, 10, 2));
    emit_uav_store(&mut code, 1, 0);
    let end = code.len();
    code[br] = s_cbranch_execz((end - (br + 1)) as i16);
    code.push(S_ENDPGM);
    make_cs_program(&code, [threads, 1, 1], 0, false)
}

fn divergent_readfirstlane_early_return_expected(threads: u32) -> Vec<u32> {
    (0..threads)
        .map(|t| {
            let (wave, lane) = (t >> 6, t & 63);
            // Lanes 0..=2 return before storing (the buffer starts zeroed).
            if lane > 2 { 3 * 8 + wave + lane } else { 0 }
        })
        .collect()
}

/// A divergent `if` nested in a divergent `if`, each with a
/// `v_readfirstlane`, joined between them:
/// `v1 = 1000; if (lane > 4) { if (lane & 1) { v1 = rfl(lane * 8 + wave) + lane }
/// v1 += rfl(lane * 3 + wave) } store(v1)`.
fn divergent_readfirstlane_nested_program(threads: u32) -> NativeProgram {
    let mut code = vec![
        v_and_b32(2, inline_u(63), 0),    // v2 = lane
        v_lshrrev_b32(3, inline_u(6), 0), // v3 = wave
        v_lshlrev_b32(4, inline_u(3), 2), // v4 = lane * 8
        v_add_nc_u32(4, vgpr(3), 4),      // v4 = lane * 8 + wave
        v_lshlrev_b32(7, inline_u(1), 2), // v7 = lane * 2
        v_add_nc_u32(7, vgpr(2), 7),      // v7 = lane * 3
        v_add_nc_u32(7, vgpr(3), 7),      // v7 = lane * 3 + wave
        v_and_b32(6, inline_u(1), 2),     // v6 = lane & 1
        v_mov_b32(1, LIT),
        1000,
        v_cmp_lt_u32(inline_u(4), 2), // 4 < lane
        s_and_saveexec_b64(6, VCC),
    ];
    let outer = code.len();
    code.push(0);
    code.push(v_cmp_eq_u32(inline_u(1), 6)); // 1 == lane & 1
    code.push(s_and_saveexec_b64(8, VCC));
    let inner = code.len();
    code.push(0);
    code.push(v_readfirstlane_b32(10, 4));
    code.push(v_add_nc_u32(1, 10, 2));
    let inner_end = code.len();
    code[inner] = s_cbranch_execz((inner_end - (inner + 1)) as i16);
    code.push(s_or_b64(EXEC_LO, EXEC_LO, 8));
    code.push(v_readfirstlane_b32(11, 7));
    code.push(v_add_nc_u32(1, 11, 1));
    let outer_end = code.len();
    code[outer] = s_cbranch_execz((outer_end - (outer + 1)) as i16);
    code.push(s_or_b64(EXEC_LO, EXEC_LO, 6));
    emit_uav_store(&mut code, 1, 0);
    code.push(S_ENDPGM);
    make_cs_program(&code, [threads, 1, 1], 0, false)
}

fn divergent_readfirstlane_nested_expected(threads: u32) -> Vec<u32> {
    (0..threads)
        .map(|t| {
            let (wave, lane) = (t >> 6, t & 63);
            if lane <= 4 {
                return 1000;
            }
            // Lane 5 is the first of the outer lanes, and the first odd one.
            let outer = 5 * 3 + wave;
            if lane & 1 == 1 {
                (5 * 8 + wave) + lane + outer
            } else {
                1000 + outer
            }
        })
        .collect()
}

#[test]
fn divergent_readfirstlane_nested_and_early_return_lower_in_emulate() {
    for threads in [64, 128] {
        for prog in [
            divergent_readfirstlane_if_program(threads, 2),
            divergent_readfirstlane_early_return_program(threads),
            divergent_readfirstlane_nested_program(threads),
            divergent_readfirstlane_uniform_loop_program(threads),
        ] {
            let shader = emit_cs(&prog, Wave64Mode::Emulate)
                .unwrap_or_else(|e| panic!("{threads} threads: {e}"));
            check_dxil(&shader, "divergent readfirstlane");
        }
    }
}

/// `v6 = (lane & 3) + 1; do { v1 += readfirstlane(lane * 8 + wave); v6 -= 1;
/// exec &= (v6 > 0) } while (exec != 0)`: each lane leaves the loop after its
/// own number of rounds (a loop of divergent trip count).
fn divergent_trip_loop_program(threads: u32) -> NativeProgram {
    let mut code = vec![
        v_and_b32(2, inline_u(63), 0),    // v2 = lane
        v_lshrrev_b32(3, inline_u(6), 0), // v3 = wave
        v_lshlrev_b32(4, inline_u(3), 2), // v4 = lane * 8
        v_add_nc_u32(4, vgpr(3), 4),      // v4 = lane * 8 + wave
        v_and_b32(6, inline_u(3), 2),     // v6 = lane & 3
        v_add_nc_u32(6, inline_u(1), 6),  // v6 = rounds
        v_mov_b32(1, inline_u(0)),
        s_mov_b64(8, EXEC_LO),
    ];
    let top = code.len();
    code.push(v_readfirstlane_b32(10, 4));
    code.push(v_add_nc_u32(1, 10, 1));
    code.push(v_subrev_nc_u32(6, inline_u(1), 6));
    code.push(v_cmp_lt_u32(inline_u(0), 6));
    code.push(sop2(EXEC_LO, 0x0F, EXEC_LO, VCC)); // s_and_b64 exec, exec, vcc
    let back = code.len();
    code.push(sopp(0x09, (top as i64 - (back as i64 + 1)) as i16)); // s_cbranch_execnz
    code.push(s_mov_b64(EXEC_LO, 8));
    emit_uav_store(&mut code, 1, 0);
    code.push(S_ENDPGM);
    make_cs_program(&code, [threads, 1, 1], 0, false)
}

#[test]
fn divergent_readfirstlane_in_a_divergent_loop_lowers_in_emulate() {
    // The lanes leave the loop in different rounds: one real loop that every
    // thread of the group runs while any thread is in (the exchange's
    // barriers are reached by all), each thread guarded by its own flag.
    for threads in [64, 128] {
        let prog = divergent_trip_loop_program(threads);
        let shader = emit_cs(&prog, Wave64Mode::Emulate)
            .unwrap_or_else(|e| panic!("{threads} threads: {e}"));
        check_dxil(&shader, "divergent trip loop");
    }
}

/// `v1 = 0; for (s12 = 0; s12 < 3; s12++) { if (lane > 5 + s12) { v1 +=
/// readfirstlane(lane * 8 + wave); } }`: a loop all threads go round alike,
/// with a divergent read in it, whose first active lane moves each round.
fn divergent_readfirstlane_uniform_loop_program(threads: u32) -> NativeProgram {
    let mut code = vec![
        v_and_b32(2, inline_u(63), 0),    // v2 = lane
        v_lshrrev_b32(3, inline_u(6), 0), // v3 = wave
        v_lshlrev_b32(4, inline_u(3), 2), // v4 = lane * 8
        v_add_nc_u32(4, vgpr(3), 4),      // v4 = lane * 8 + wave
        v_mov_b32(1, inline_u(0)),
        s_mov_b32(12, inline_u(0)),
    ];
    let top = code.len();
    code.push(s_add_i32(13, 12, inline_u(5)));
    code.push(v_cmp_lt_u32(13, 2)); // 5 + round < lane
    code.push(s_and_saveexec_b64(6, VCC));
    let br = code.len();
    code.push(0);
    code.push(v_readfirstlane_b32(10, 4));
    code.push(v_add_nc_u32(1, 10, 1));
    let after = code.len();
    code[br] = s_cbranch_execz((after - (br + 1)) as i16);
    code.push(s_or_b64(EXEC_LO, EXEC_LO, 6));
    code.push(s_add_i32(12, 12, inline_u(1)));
    code.push(s_cmp_lt_u32(12, inline_u(3)));
    let back = code.len();
    code.push(s_cbranch_scc1((top as i64 - (back as i64 + 1)) as i16));
    emit_uav_store(&mut code, 1, 0);
    code.push(S_ENDPGM);
    make_cs_program(&code, [threads, 1, 1], 0, false)
}

fn divergent_readfirstlane_uniform_loop_expected(threads: u32) -> Vec<u32> {
    (0..threads)
        .map(|t| {
            let (wave, lane) = (t >> 6, t & 63);
            (0..3u32)
                .filter(|round| lane > 5 + round)
                // The first lane above the threshold is the first active one.
                .map(|round| (5 + round + 1) * 8 + wave)
                .sum()
        })
        .collect()
}

#[test]
fn divergent_uniform_readfirstlane_succeeds_without_barrier() {
    // s8 = 42; v2 = s8 (uniform across the group); v_mbcnt forces console_lanes;
    // if (tid < 40) s4 = readfirstlane(v2); must succeed on Wave32 and Emulate.
    let mut code = Vec::new();
    code.extend_from_slice(&v_mbcnt_lo_u32_b32(3, NEG_ONE, inline_u(0)));
    code.extend_from_slice(&v_mbcnt_hi_u32_b32(3, NEG_ONE, vgpr(3)));
    code.push(s_mov_b32(8, inline_u(42)));
    code.push(v_mov_b32(2, 8));
    code.push(v_mov_b32(1, 3));
    code.push(v_cmp_gt_u32(inline_u(40), 0));
    code.push(s_and_saveexec_b64(6, VCC));
    let br_idx = code.len();
    code.push(0);
    code.push(v_readfirstlane_b32(4, 2));
    code.push(v_add_nc_u32(1, 4, 3));
    let after_then = code.len();
    code[br_idx] = s_cbranch_execz((after_then - (br_idx + 1)) as i16);
    code.push(s_or_b64(EXEC_LO, EXEC_LO, 6));
    emit_uav_store(&mut code, 1, 0);
    code.push(S_ENDPGM);

    let prog = make_cs_program(&code, [64, 1, 1], 0, false);
    for mode in [Wave64Mode::Native, Wave64Mode::Wave32, Wave64Mode::Emulate] {
        emit_cs(&prog, mode).unwrap_or_else(|e| panic!("{mode:?} failed: {e}"));
    }
}

#[test]
fn divergent_early_return_taints_subsequent_blocks() {
    // if (v0 < 32) return; // after_ret is non-uniform across the group!
    // bcnt1 in after_ret must fail instead of inserting a hanging barrier.
    let mut code = Vec::new();
    code.push(v_cmp_gt_u32(inline_u(32), 0));
    code.push(s_and_saveexec_b64(6, VCC));
    let br_idx = code.len();
    code.push(0);
    code.push(S_ENDPGM);
    let rest = code.len();
    code[br_idx] = s_cbranch_execz((rest - (br_idx + 1)) as i16);
    code.push(v_cmp_gt_u32(LIT, 0));
    code.push(96);
    code.push(s_bcnt1_i32_b64(4, VCC));
    code.push(v_mov_b32(1, 4));
    emit_uav_store(&mut code, 1, 0);
    code.push(S_ENDPGM);

    let prog = make_cs_program(&code, [128, 1, 1], 0, false);
    assert!(emit_cs(&prog, Wave64Mode::Native).is_ok());
    assert!(matches!(
        emit_cs(&prog, Wave64Mode::Wave32),
        Err(ShaderError::Unsupported { .. })
    ));
    // Emulate: the lanes that returned keep passing the barriers (they run
    // every block, off their path).
    let shader = emit_cs(&prog, Wave64Mode::Emulate).expect("predicated flow");
    check_dxil(&shader, "early return then ballot");
}

// Synthetic LDS ordering patterns. With a join, the writer and reader are
// separate EXEC-masked regions; without one, both accesses stay divergent.
fn lds_region_program(divergent_write: bool, join_between: bool, delay: bool) -> NativeProgram {
    let mut code = vec![
        v_mov_b32(3, inline_u(7)),
        v_lshlrev_b32(2, inline_u(2), 0),
        v_add_nc_u32(4, inline_u(7), 0),
    ];
    let mut writer_branch = None;
    if divergent_write {
        code.push(v_cmp_lt_u32(inline_u(31), 0)); // lanes 32..63
        code.push(s_and_saveexec_b64(6, VCC));
        writer_branch = Some(code.len());
        code.push(0);
        if delay {
            code.push(s_mov_b32(10, inline_u(0)));
            let top = code.len();
            code.extend([vop2(0x0B, 4, LIT, 4), 40503]);
            code.push(v_add_nc_u32(4, inline_u(1), 4));
            code.push(s_add_i32(10, 10, inline_u(1)));
            code.extend([s_cmp_lt_u32(10, LIT), 4096]);
            let at = code.len();
            code.push(s_cbranch_scc1((top as isize - (at + 1) as isize) as i16));
        }
    }
    code.extend(ds_op(0x0D, 0, 0, 2, 4, 0)); // ds_write_b32 v2, v4
    if join_between {
        if let Some(at) = writer_branch.take() {
            code[at] = s_cbranch_execz((code.len() - (at + 1)) as i16);
            code.push(s_or_b64(EXEC_LO, EXEC_LO, 6));
        }
        code.push(v_cmp_gt_u32(inline_u(32), 0)); // lanes 0..31
        code.push(s_and_saveexec_b64(6, VCC));
        let at = code.len();
        code.push(0);
        code.extend(ds_op(0x36, 128, 0, 2, 0, 3)); // ds_read_b32 v3, v2 offset:128
        code[at] = s_cbranch_execz((code.len() - (at + 1)) as i16);
    } else {
        code.extend(ds_op(0x36, 0, 0, 2, 0, 3)); // read in the writer's divergent block
        let at = writer_branch.expect("divergent writer");
        code[at] = s_cbranch_execz((code.len() - (at + 1)) as i16);
    }
    code.push(s_or_b64(EXEC_LO, EXEC_LO, 6));
    emit_uav_store(&mut code, 3, 0);
    code.push(S_ENDPGM);
    make_cs_program(&code, [64, 1, 1], 512, false)
}

#[test]
fn divergent_lds_without_a_uniform_fence_is_rejected() {
    let prog = lds_region_program(true, false, false);
    assert!(emit_cs(&prog, Wave64Mode::Native).is_ok());
    let err = match emit_cs(&prog, Wave64Mode::Wave32) {
        Err(err) => err,
        Ok(_) => panic!("expected an unfenceable LDS access to be refused in Wave32"),
    };
    assert!(
        matches!(err, ShaderError::Unsupported { what, .. }
        if what.contains("uniform LDS fence")),
        "Wave32: {err:?}"
    );
    // Emulate runs the divergent block in every thread, with a barrier in
    // front of every access.
    let shader = emit_cs(&prog, Wave64Mode::Emulate).expect("predicated LDS");
    check_dxil(&shader, "divergent write then read");
}

#[test]
fn lds_regions_fence_at_uniform_joins_and_exits() {
    for (divergent_write, delay) in [(false, false), (true, false), (true, true)] {
        let prog = lds_region_program(divergent_write, true, delay);
        let p = ir::build(&prog.patched_code(), prog.wave64, prog.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &prog.layout, prog.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &prog).unwrap();
        let (uniform, _) = super::lower::compute_uniformity(&p, &st, &res, &io.inputs);
        let fences = super::lower::plan_lds_fences(&p, &uniform).unwrap();
        let writer = p
            .blocks
            .iter()
            .position(|b| b.insts.iter().any(|i| i.name() == "ds_write_b32"))
            .unwrap();
        let reader = p
            .blocks
            .iter()
            .position(|b| b.insts.iter().any(|i| i.name() == "ds_read_b32"))
            .unwrap();
        assert!(!uniform[reader]);
        assert!(
            fences
                .enter
                .iter()
                .chain(&fences.leave)
                .all(|&b| uniform[b])
        );
        if divergent_write {
            assert!(!uniform[writer]);
            let crate::exact64::ir::Exit::Jump(join) = p.blocks[writer].exit else {
                panic!("writer rejoins")
            };
            assert!(
                fences.enter.contains(&join),
                "the join has no LDS instruction but needs a fence"
            );
        } else {
            assert!(uniform[writer]);
            assert!(
                fences.leave.contains(&writer),
                "the write must finish before the divergent read"
            );
        }
        for mode in [Wave64Mode::Wave32, Wave64Mode::Emulate] {
            let mut shader = emit_cs(&prog, mode).unwrap();
            #[cfg(windows)]
            if let Ok(mut validator) = super::validator::Validator::load(
                super::validator::Validator::default_path().as_deref(),
            ) {
                validator.validate(&mut shader.dxil).unwrap();
            }
        }
    }
}

fn guarded_lds_program() -> NativeProgram {
    let mut code = vec![
        v_mov_b32(3, inline_u(7)),
        v_lshlrev_b32(2, inline_u(2), 0),
        v_add_nc_u32(4, inline_u(7), 0),
        v_cmp_lt_u32(inline_u(31), 0),
        s_and_saveexec_b64(6, VCC),
    ];
    code.extend(ds_op(0x0D, 0, 0, 2, 4, 0));
    code.extend([
        s_or_b64(EXEC_LO, EXEC_LO, 6),
        v_cmp_gt_u32(inline_u(32), 0),
        s_and_saveexec_b64(6, VCC),
    ]);
    code.extend(ds_op(0x36, 128, 0, 2, 0, 3));
    code.push(s_or_b64(EXEC_LO, EXEC_LO, 6));
    emit_uav_store(&mut code, 3, 0);
    code.push(S_ENDPGM);
    make_cs_program(&code, [64, 1, 1], 512, false)
}

#[test]
fn guarded_lds_in_a_uniform_block_stays_fenceable() {
    let prog = guarded_lds_program();
    let p = ir::build(&prog.patched_code(), prog.wave64, prog.vgprs).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &prog.layout, prog.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &prog).unwrap();
    assert_eq!(p.blocks.len(), 1);
    for (i, inst) in p.blocks[0].insts.iter().enumerate() {
        if inst.name() == "ds_write_b32" || inst.name() == "ds_read_b32" {
            assert!(st.guards[0][i].is_some());
        }
    }
    let (uniform, _) = super::lower::compute_uniformity(&p, &st, &res, &io.inputs);
    assert_eq!(uniform, [true]);
    let fences = super::lower::plan_lds_fences(&p, &uniform).unwrap();
    assert!(fences.leave.contains(&0));
    for mode in [Wave64Mode::Wave32, Wave64Mode::Emulate] {
        assert!(emit_cs(&prog, mode).is_ok());
    }
}

#[test]
fn lds_in_opposite_divergent_arms_cannot_share_a_fence() {
    // Model the per-lane CFG directly: a writer and a reader in sibling
    // arms have no path between them, but the host halves run concurrently.
    let prog = guarded_lds_program();
    let mut p = ir::build(&prog.patched_code(), prog.wave64, prog.vgprs).unwrap();
    let writer = p.blocks[0]
        .insts
        .iter()
        .find(|i| i.name() == "ds_write_b32")
        .unwrap()
        .clone();
    let reader = p.blocks[0]
        .insts
        .iter()
        .find(|i| i.name() == "ds_read_b32")
        .unwrap()
        .clone();
    let block = |preds, insts, exit| crate::exact64::ir::Block {
        offset: 0,
        preds,
        phis: vec![],
        insts,
        exit,
    };
    p.blocks = vec![
        block(
            vec![],
            vec![],
            crate::exact64::ir::Exit::Branch {
                taken: 1,
                not_taken: 2,
            },
        ),
        block(vec![0], vec![writer], crate::exact64::ir::Exit::Jump(3)),
        block(vec![0], vec![reader], crate::exact64::ir::Exit::Jump(3)),
        block(vec![1, 2], vec![], crate::exact64::ir::Exit::End),
    ];
    assert!(super::lower::plan_lds_fences(&p, &[true, false, false, true]).is_err());
    // If the writer returns, the reader's suffix is no longer reached by
    // the whole group; it must not get a group-sync barrier there either.
    p.blocks[1].exit = crate::exact64::ir::Exit::End;
    p.blocks[3].preds = vec![2];
    assert!(super::lower::plan_lds_fences(&p, &[true, false, false, false]).is_err());
}

#[test]
fn divergent_multiple_lds_reads_without_writes_succeed() {
    // Uniform write -> divergent branch containing two ds_read_b32s -> join.
    // Multiple pure reads in a divergent region cannot race with each other.
    let mut code = vec![
        v_mov_b32(3, inline_u(0)),
        v_lshlrev_b32(2, inline_u(2), 0),
        v_add_nc_u32(4, inline_u(7), 0),
    ];
    code.extend(ds_op(0x0D, 0, 0, 2, 4, 0)); // ds_write_b32
    code.push(v_cmp_gt_u32(inline_u(32), 0));
    code.push(s_and_saveexec_b64(6, VCC));
    let br = code.len();
    code.push(0);
    code.extend(ds_op(0x36, 0, 0, 2, 0, 3));
    code.extend(ds_op(0x36, 4, 0, 2, 0, 5));
    code.push(v_add_nc_u32(3, vgpr(5), 3));
    code[br] = s_cbranch_execz((code.len() - (br + 1)) as i16);
    code.push(s_or_b64(EXEC_LO, EXEC_LO, 6));
    emit_uav_store(&mut code, 3, 0);
    code.push(S_ENDPGM);

    let prog = make_cs_program(&code, [64, 1, 1], 512, false);
    for mode in [Wave64Mode::Native, Wave64Mode::Wave32, Wave64Mode::Emulate] {
        assert!(emit_cs(&prog, mode).is_ok(), "{mode:?} should succeed");
    }

    // <= 32 threads in Wave32 fits inside one native 32-lane wave, so even a
    // divergent write + read succeeds on Wave32 (and Emulate, predicated).
    let mut small = lds_region_program(true, false, false);
    small.io = StageIo::Cs(ComputeIo {
        rsrc2: match small.io {
            StageIo::Cs(c) => c.rsrc2,
            _ => unreachable!(),
        },
        threads: [32, 1, 1],
    });
    assert!(emit_cs(&small, Wave64Mode::Wave32).is_ok());
    assert!(emit_cs(&small, Wave64Mode::Emulate).is_ok());
}

#[test]
fn translate_with_handles_synthetic_wave64_container() {
    // Build genuine DXBC + XDAT(XBCH(FSSM)) Wave64 compute container with ResourceHeap param.
    let mut code = Vec::new();
    code.extend_from_slice(&v_mbcnt_lo_u32_b32(1, NEG_ONE, inline_u(0)));
    code.extend_from_slice(&v_mbcnt_hi_u32_b32(1, NEG_ONE, vgpr(1)));
    emit_uav_store(&mut code, 1, 0);
    code.push(S_ENDPGM);
    let code_bytes: Vec<u8> = code.iter().flat_map(|w| w.to_le_bytes()).collect();

    const ABSENT: u64 = 0xDEAD_BEEF_DEAD_BEEF;
    const RECORD_SIZE: usize = 0x54;
    const STAGE_SIZE: usize = 0x34;
    const ENTRY_SIZE: usize = 20;

    let mut payload = 6u64.to_le_bytes().to_vec(); // GRAPHICS_SLOT_COUNT
    payload.resize(0x68, 0);
    for slot_off in [0x08usize, 0x10, 0x18, 0x20, 0x28, 0x30, 0x48, 0x60] {
        payload[slot_off..slot_off + 8].copy_from_slice(&ABSENT.to_le_bytes());
    }
    let rec = payload.len() as u64;
    payload[0x60..0x68].copy_from_slice(&rec.to_le_bytes());
    payload.resize(rec as usize + RECORD_SIZE + STAGE_SIZE, 0);

    let r = rec as usize;
    let code_at = (rec as usize + RECORD_SIZE + STAGE_SIZE) as u64;
    let layout_at = code_at + code_bytes.len() as u64;
    payload[r..r + 8].copy_from_slice(&code_at.to_le_bytes());
    payload[r + 8..r + 16].copy_from_slice(&(code_bytes.len() as u64).to_le_bytes());
    payload[r + 0x28..r + 0x30].copy_from_slice(&layout_at.to_le_bytes()); // RECORD_LAYOUT
    payload[r + 0x30..r + 0x38].copy_from_slice(&layout_at.to_le_bytes()); // RECORD_USED
    payload[r + 0x38..r + 0x3C].copy_from_slice(&1u32.to_le_bytes()); // 1 root param
    payload[r + 0x3C..r + 0x40].copy_from_slice(&0u32.to_le_bytes()); // 0 used entries
    payload[r + 0x50..r + 0x54].copy_from_slice(&(32u32 | (31 << 20)).to_le_bytes()); // RECORD_REGS

    let sd = r + RECORD_SIZE;
    payload[sd + 0x24..sd + 0x28].copy_from_slice(&(2u32 << 1).to_le_bytes()); // CS_RSRC2: 2 user SGPRs
    payload[sd + 0x28..sd + 0x2C].copy_from_slice(&64u32.to_le_bytes()); // CS_THREADS[0]
    payload[sd + 0x2C..sd + 0x30].copy_from_slice(&1u32.to_le_bytes());
    payload[sd + 0x30..sd + 0x34].copy_from_slice(&1u32.to_le_bytes());

    payload.extend_from_slice(&code_bytes);
    // RootParam: ResourceHeap (5), table_off=0, sgprs=1, base_reg=0, user_data=0, slot=0
    let mut entry = [0u8; ENTRY_SIZE];
    entry[0..4].copy_from_slice(&5u32.to_le_bytes());
    entry[8..12].copy_from_slice(&1u32.to_le_bytes());
    payload.extend_from_slice(&entry);

    let mut fssm = b"FSSM".to_vec();
    fssm.extend_from_slice(&0x101u32.to_le_bytes()); // FSSM_VERSION
    fssm.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    fssm.extend_from_slice(&payload);

    let compressed = {
        let mut buf = vec![0u8; zlib_rs::compress_bound(fssm.len())];
        let (done, rc) = zlib_rs::compress_slice(&mut buf, &fssm, zlib_rs::DeflateConfig::new(6));
        assert_eq!(rc, zlib_rs::ReturnCode::Ok);
        done.to_vec()
    };
    let mut xbch = b"XBCH".to_vec();
    xbch.extend_from_slice(&3u32.to_le_bytes()); // XBCH_VERSION
    xbch.extend_from_slice(&(compressed.len() as u64).to_le_bytes());
    xbch.extend_from_slice(&(fssm.len() as u64).to_le_bytes());
    xbch.extend_from_slice(&compressed);

    let mut dxil_part = vec![0u8; 0x18];
    dxil_part.extend_from_slice(b"XDAT");
    dxil_part.extend_from_slice(&1u32.to_le_bytes());
    dxil_part.extend_from_slice(&0x24u32.to_le_bytes());
    let blob_off = (dxil_part.len() + 12) as u32;
    for v in [0x0Du32, blob_off, xbch.len() as u32] {
        dxil_part.extend_from_slice(&v.to_le_bytes());
    }
    dxil_part.extend_from_slice(&xbch);

    let part_off = 0x24u32;
    let total = part_off as usize + 8 + dxil_part.len();
    let mut dxbc = b"DXBC".to_vec();
    dxbc.extend_from_slice(&[0u8; 16]);
    dxbc.extend_from_slice(&1u32.to_le_bytes());
    dxbc.extend_from_slice(&(total as u32).to_le_bytes());
    dxbc.extend_from_slice(&1u32.to_le_bytes());
    dxbc.extend_from_slice(&part_off.to_le_bytes());
    dxbc.extend_from_slice(b"DXIL");
    dxbc.extend_from_slice(&(dxil_part.len() as u32).to_le_bytes());
    dxbc.extend_from_slice(&dxil_part);

    let sh_n = translate_with(
        &dxbc,
        Options {
            wave64: Wave64Mode::Native,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(psv_wave_range(&sh_n), (64, 64));
    let sh_w32 = translate_with(
        &dxbc,
        Options {
            wave64: Wave64Mode::Wave32,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(psv_wave_range(&sh_w32), (32, 32));
    let sh_emu = translate_with(
        &dxbc,
        Options {
            wave64: Wave64Mode::Emulate,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(psv_wave_range(&sh_emu), (0, u32::MAX));
}

#[test]
fn pure_wave_sections_lds_allocation_and_emulate_contract() {
    // Pure wave::sections shader: DPP + v_permlanex16_b32 + v_readlane_b32 all lie inside the
    // recognized section, so Wave64Mode::Wave32 must allocate only 2 * virtual_waves LDS dwords
    // (NOT 66 * virtual_waves), while Wave64Mode::Emulate keeps zero WAVE_OPS and PSV (0, u32::MAX).
    let p64 =
        whole_wave_float_sum_uav_x4_program([64, 1, 1], 0, false, Lane0Guard::TidEqZeroBranch);
    let w32_64 = lower_cs(&p64, Wave64Mode::Wave32).unwrap();
    assert_eq!(w32_64.lds_dwords, 2);
    assert!(w32_64.wave_ops);

    let p64_title_lds =
        whole_wave_float_sum_uav_x4_program([64, 1, 1], 512, false, Lane0Guard::TidEqZeroBranch);
    let w32_64_title = lower_cs(&p64_title_lds, Wave64Mode::Wave32).unwrap();
    assert_eq!(w32_64_title.lds_dwords, 128 + 2);

    let p128 =
        whole_wave_float_sum_uav_x4_program([128, 1, 1], 0, false, Lane0Guard::TidEqZeroBranch);
    let w32_128 = lower_cs(&p128, Wave64Mode::Wave32).unwrap();
    assert_eq!(w32_128.lds_dwords, 4);

    // 1024 threads (16 virtual waves) + 32256 bytes (63 * 512B granules = 8064 dwords) title LDS:
    // 8064 + 2*16 = 8096 <= 8192 succeeds in Wave32 (would overflow if 64*16 were wasted).
    let p1024_near_limit = whole_wave_float_sum_uav_x4_program(
        [1024, 1, 1],
        32256,
        false,
        Lane0Guard::TidEqZeroBranch,
    );
    let w32_1024 = lower_cs(&p1024_near_limit, Wave64Mode::Wave32).unwrap();
    assert_eq!(w32_1024.lds_dwords, 8064 + 32);

    let emu_64 = lower_cs(&p64, Wave64Mode::Emulate).unwrap();
    assert!(!emu_64.wave_ops);
    assert_eq!(emu_64.lds_dwords, 66);
    let emu_shader = emit_cs(&p64, Wave64Mode::Emulate).unwrap();
    assert_eq!(psv_wave_range(&emu_shader), (0, u32::MAX));
    assert_eq!(sfi0_flags(&emu_shader) & 0x4000, 0);

    // An unresolved dynamic v_readlane_b32 outside wave::sections must still allocate 64 * vwaves.
    let mut external_readlane = vec![v_mov_b32(1, 0)];
    external_readlane.extend_from_slice(&v_readlane_b32(4, 1, vgpr(0)));
    external_readlane.push(v_mov_b32(1, 4));
    emit_uav_store(&mut external_readlane, 1, 0);
    external_readlane.push(S_ENDPGM);
    let p_ext = make_cs_program(&external_readlane, [64, 1, 1], 0, false);
    assert_eq!(lower_cs(&p_ext, Wave64Mode::Wave32).unwrap().lds_dwords, 66);
}

#[test]
fn wave_section_readlane_preexisting_vgpr_is_rejected() {
    // v_readlane_b32 inside a wave::sections reading a pre-section VGPR (Sym::Own) and
    // escaping must fail explicitly with Unsupported instead of miscompiling to Shape::Value.
    let mut direct = vec![
        v_add_nc_u32(2, inline_u(1), 0),
        s_or_saveexec_b64(4, NEG_ONE),
    ];
    direct.extend_from_slice(&v_readlane_b32(12, 2, inline_u(35)));
    direct.push(s_mov_b64(EXEC_LO, 4));
    direct.push(v_mov_b32(1, 12));
    emit_uav_store(&mut direct, 1, 0);
    direct.push(S_ENDPGM);

    let prog_direct = make_cs_program(&direct, [64, 1, 1], 0, false);
    for mode in [Wave64Mode::Native, Wave64Mode::Wave32, Wave64Mode::Emulate] {
        let err = emit_cs(&prog_direct, mode).unwrap_err();
        assert!(
            matches!(err, ShaderError::Unsupported { .. }),
            "expected Unsupported for Sym::Own readlane escape in {mode:?}, got {err:?}"
        );
    }

    // Moving a pre-section VGPR across lanes via v_permlanex16_b32 inside wave::sections must
    // also refuse to miscompile as Shape::Value.
    let mut perm = vec![
        v_add_nc_u32(2, inline_u(1), 0),
        s_or_saveexec_b64(4, NEG_ONE),
    ];
    perm.extend_from_slice(&v_permlanex16_b32(10, 2, NEG_ONE, NEG_ONE));
    perm.push(s_mov_b64(EXEC_LO, 4));
    emit_uav_store(&mut perm, 10, 0);
    perm.push(S_ENDPGM);

    let prog_perm = make_cs_program(&perm, [64, 1, 1], 0, false);
    for mode in [Wave64Mode::Native, Wave64Mode::Wave32, Wave64Mode::Emulate] {
        let err = emit_cs(&prog_perm, mode).unwrap_err();
        assert!(
            matches!(err, ShaderError::Unsupported { .. }),
            "expected Unsupported for cross-lane Sym::Own escape in {mode:?}, got {err:?}"
        );
    }
}

#[cfg(windows)]
// ---- predicated flow: loops with per-lane trips, ballots, lane reads, LDS ----

fn v_xor_b32(dst: u32, src0: u32, vsrc1: u32) -> u32 {
    vop2(0x1D, dst, src0, vsrc1)
}
fn s_and_b64(dst: u32, src0: u32, src1: u32) -> u32 {
    sop2(dst, 0x0F, src0, src1)
}
fn s_cbranch_execnz(delta_words: i16) -> u32 {
    sopp(0x09, delta_words)
}
fn ds_bpermute_b32(vdst: u32, addr: u32, data: u32) -> [u32; 2] {
    ds_op(0xB3, 0, 0, addr, data, vdst)
}

/// The prologue the loop programs share: `v2 = lane`, `v3 = wave`,
/// `v4 = lane * 8 + wave`.
fn lane_prologue() -> Vec<u32> {
    vec![
        v_and_b32(2, inline_u(63), 0),    // v2 = lane
        v_lshrrev_b32(3, inline_u(6), 0), // v3 = wave
        v_lshlrev_b32(4, inline_u(3), 2), // v4 = lane * 8
        v_add_nc_u32(4, vgpr(3), 4),      // v4 = lane * 8 + wave
    ]
}

/// A mid-loop exit: `t = (lane & 3) + 1; for (r = 0; r < 6; r++) { v1 +=
/// readfirstlane(lane * 8 + wave); if (r + 1 >= t) break; v1 += 100; }`,
/// the break a per-lane exit through EXEC (`s_cbranch_execz` when no lane
/// is left), the loop's back edge a scalar test.
fn mid_exit_loop_program(threads: u32) -> NativeProgram {
    let mut code = lane_prologue();
    code.extend([
        v_and_b32(6, inline_u(3), 2),    // v6 = lane & 3
        v_add_nc_u32(6, inline_u(1), 6), // v6 = t
        v_mov_b32(1, inline_u(0)),
        s_mov_b32(12, inline_u(0)),
        s_mov_b64(8, EXEC_LO),
    ]);
    let top = code.len();
    code.push(v_readfirstlane_b32(10, 4));
    code.push(v_add_nc_u32(1, 10, 1));
    code.push(s_add_i32(13, 12, inline_u(1)));
    code.push(v_cmp_lt_u32(13, 6)); // r + 1 < t
    code.push(s_and_b64(EXEC_LO, EXEC_LO, VCC));
    let exit = code.len();
    code.push(0); // s_cbranch_execz done
    code.extend([v_add_nc_u32(1, LIT, 1), 100]);
    code.push(s_add_i32(12, 12, inline_u(1)));
    code.push(s_cmp_lt_u32(12, inline_u(6)));
    let back = code.len();
    code.push(s_cbranch_scc1((top as i64 - (back as i64 + 1)) as i16));
    let done = code.len();
    code[exit] = s_cbranch_execz((done - (exit + 1)) as i16);
    code.push(s_mov_b64(EXEC_LO, 8));
    emit_uav_store(&mut code, 1, 0);
    code.push(S_ENDPGM);
    make_cs_program(&code, [threads, 1, 1], 0, false)
}

fn mid_exit_loop_expected(threads: u32) -> Vec<u32> {
    (0..threads)
        .map(|t| {
            let (wave, lane) = (t >> 6, t & 63);
            let rounds = (lane & 3) + 1;
            (0..rounds).map(|r| r * 8 + wave).sum::<u32>() + 100 * (rounds - 1)
        })
        .collect()
}

/// A divergent loop inside a divergent loop, each lane with its own trips,
/// and a `v_readfirstlane` in the inner one.
fn nested_divergent_loop_program(threads: u32) -> NativeProgram {
    let mut code = lane_prologue();
    code.extend([
        v_and_b32(6, inline_u(1), 2),    // v6 = lane & 1
        v_add_nc_u32(6, inline_u(1), 6), // v6 = outer trips
        v_mov_b32(1, inline_u(0)),
        s_mov_b64(8, EXEC_LO),
    ]);
    let outer = code.len();
    code.push(s_mov_b64(14, EXEC_LO)); // the outer round's lanes
    code.push(v_lshrrev_b32(7, inline_u(2), 2)); // v7 = lane >> 2
    code.push(v_and_b32(7, inline_u(3), 7)); // v7 = (lane >> 2) & 3
    code.push(v_add_nc_u32(7, inline_u(1), 7)); // v7 = inner trips
    let inner = code.len();
    code.push(v_readfirstlane_b32(10, 4));
    code.push(v_add_nc_u32(1, 10, 1));
    code.push(v_subrev_nc_u32(7, inline_u(1), 7));
    code.push(v_cmp_lt_u32(inline_u(0), 7));
    code.push(s_and_b64(EXEC_LO, EXEC_LO, VCC));
    let at = code.len();
    code.push(s_cbranch_execnz((inner as i64 - (at as i64 + 1)) as i16));
    code.push(s_mov_b64(EXEC_LO, 14));
    code.push(v_subrev_nc_u32(6, inline_u(1), 6));
    code.push(v_cmp_lt_u32(inline_u(0), 6));
    code.push(s_and_b64(EXEC_LO, EXEC_LO, VCC));
    let at = code.len();
    code.push(s_cbranch_execnz((outer as i64 - (at as i64 + 1)) as i16));
    code.push(s_mov_b64(EXEC_LO, 8));
    emit_uav_store(&mut code, 1, 0);
    code.push(S_ENDPGM);
    make_cs_program(&code, [threads, 1, 1], 0, false)
}

fn nested_divergent_loop_expected(threads: u32) -> Vec<u32> {
    let mut out = Vec::new();
    for wave in 0..threads.div_ceil(64) {
        let lanes = (threads - wave * 64).min(64) as usize;
        let mut v1 = vec![0u32; lanes];
        let mut outer_trips: Vec<u32> = (0..lanes as u32).map(|l| (l & 1) + 1).collect();
        let mut outer_live: Vec<bool> = vec![true; lanes];
        loop {
            let round = outer_live.clone();
            let mut inner_trips: Vec<u32> = (0..lanes as u32).map(|l| ((l >> 2) & 3) + 1).collect();
            let mut live = round.clone();
            loop {
                let first = live.iter().position(|&x| x).expect("a lane") as u32;
                for l in 0..lanes {
                    if live[l] {
                        v1[l] += first * 8 + wave;
                        inner_trips[l] -= 1;
                        live[l] = inner_trips[l] > 0;
                    }
                }
                if !live.iter().any(|&x| x) {
                    break;
                }
            }
            for l in 0..lanes {
                if round[l] {
                    outer_trips[l] -= 1;
                    outer_live[l] = outer_trips[l] > 0;
                }
            }
            if !outer_live.iter().any(|&x| x) {
                break;
            }
        }
        out.extend(v1);
    }
    out
}

/// `if (lane even) { s4 = bcnt1(lane < 20); v1 = s4 }` (a ballot in a
/// divergent branch), then the loop of [`divergent_trip_loop_program`] with
/// `s_bcnt1_i32_b64 s10, exec` in place of the read (EXEC as data in a
/// divergent loop).
fn ballot_in_branch_and_loop_program(threads: u32) -> NativeProgram {
    let mut code = lane_prologue();
    code.extend([
        v_and_b32(5, inline_u(1), 2), // v5 = lane & 1
        v_mov_b32(1, LIT),
        1000,
        v_cmp_eq_u32(inline_u(0), 5), // lane even
        s_and_saveexec_b64(6, VCC),
    ]);
    let br = code.len();
    code.push(0);
    code.push(v_cmp_gt_u32(inline_u(20), 2)); // lane < 20
    code.push(s_bcnt1_i32_b64(4, VCC));
    code.push(v_mov_b32(1, 4));
    code[br] = s_cbranch_execz((code.len() - (br + 1)) as i16);
    code.push(s_or_b64(EXEC_LO, EXEC_LO, 6));
    // The loop: v6 = (lane & 3) + 1 rounds; each adds the lanes still in.
    code.extend([
        v_and_b32(6, inline_u(3), 2),
        v_add_nc_u32(6, inline_u(1), 6),
        v_mov_b32(8, inline_u(0)),
        s_mov_b64(8, EXEC_LO),
    ]);
    let top = code.len();
    code.push(s_bcnt1_i32_b64(10, EXEC_LO));
    code.push(v_add_nc_u32(8, 10, 8));
    code.push(v_subrev_nc_u32(6, inline_u(1), 6));
    code.push(v_cmp_lt_u32(inline_u(0), 6));
    code.push(s_and_b64(EXEC_LO, EXEC_LO, VCC));
    let at = code.len();
    code.push(s_cbranch_execnz((top as i64 - (at as i64 + 1)) as i16));
    code.push(s_mov_b64(EXEC_LO, 8));
    code.push(v_add_nc_u32(1, vgpr(8), 1)); // v1 += loop sum
    emit_uav_store(&mut code, 1, 0);
    code.push(S_ENDPGM);
    make_cs_program(&code, [threads, 1, 1], 0, false)
}

fn ballot_in_branch_and_loop_expected(threads: u32) -> Vec<u32> {
    (0..threads)
        .map(|t| {
            let lane = t & 63;
            let branch = if lane & 1 == 0 { 10 } else { 1000 };
            // Round r (1-based) has the lanes with (lane & 3) + 1 >= r.
            let rounds = (lane & 3) + 1;
            let loop_sum: u32 = (1..=rounds).map(|r| 16 * (5 - r)).sum();
            branch + loop_sum
        })
        .collect()
}

/// Lanes read other lanes in a divergent branch: `if (lane odd) { v1 =
/// bpermute(lane ^ 2, lane + 500) + readlane(lane + 500, 5) }`.
fn cross_lane_in_branch_program(threads: u32) -> NativeProgram {
    let mut code = lane_prologue();
    code.extend([
        v_add_nc_u32(4, LIT, 2), // v4 = lane + 500
        500,
        v_xor_b32(6, inline_u(2), 2),     // v6 = lane ^ 2
        v_lshlrev_b32(5, inline_u(2), 6), // v5 = (lane ^ 2) * 4
        v_mov_b32(1, LIT),
        1000,
        v_and_b32(7, inline_u(1), 2),
        v_cmp_eq_u32(inline_u(1), 7), // lane odd
        s_and_saveexec_b64(6, VCC),
    ]);
    let br = code.len();
    code.push(0);
    code.extend(ds_bpermute_b32(8, 5, 4)); // v8 = v4 of lane (lane ^ 2)
    code.extend(v_readlane_b32(10, 4, inline_u(5))); // s10 = v4 of lane 5
    code.push(v_add_nc_u32(1, 10, 8));
    code[br] = s_cbranch_execz((code.len() - (br + 1)) as i16);
    code.push(s_or_b64(EXEC_LO, EXEC_LO, 6));
    emit_uav_store(&mut code, 1, 0);
    code.push(S_ENDPGM);
    make_cs_program(&code, [threads, 1, 1], 0, false)
}

fn cross_lane_in_branch_expected(threads: u32) -> Vec<u32> {
    (0..threads)
        .map(|t| {
            let lane = t & 63;
            if lane & 1 == 1 {
                ((lane ^ 2) + 500) + 505
            } else {
                1000
            }
        })
        .collect()
}

/// Title LDS shared by lanes of one console wave inside a divergent branch:
/// `if (tid odd) { lds[tid] = tid + 1000; v1 = lds[tid ^ 32]; }`. The
/// console runs the store for all lanes before the load, so the partner's
/// store (the other 32-lane half of the wave) is seen.
fn lockstep_lds_in_branch_program(threads: u32) -> NativeProgram {
    let mut code = Vec::new();
    if threads > 64 {
        // Beyond one console wave the per-thread form leaves LDS to the
        // program's own barriers; a divergent loop (its result unused) puts
        // the program in predicated flow, which fences each access.
        code.extend([
            v_and_b32(9, inline_u(3), 0),
            v_add_nc_u32(9, inline_u(1), 9),
            v_lshlrev_b32(12, inline_u(3), 0),
            s_mov_b64(8, EXEC_LO),
        ]);
        let top = code.len();
        code.push(v_readfirstlane_b32(10, 12));
        // The read feeds the trip count (it adds 0: the value is below 2^31).
        code.push(v_mov_b32(13, 10));
        code.extend([vop2(0x16, 13, LIT, 13), 31]); // v13 = s10 >> 31
        code.push(v_subrev_nc_u32(9, vgpr(13), 9));
        code.push(v_subrev_nc_u32(9, inline_u(1), 9));
        code.push(v_cmp_lt_u32(inline_u(0), 9));
        code.push(s_and_b64(EXEC_LO, EXEC_LO, VCC));
        let at = code.len();
        code.push(s_cbranch_execnz((top as i64 - (at as i64 + 1)) as i16));
        code.push(s_mov_b64(EXEC_LO, 8));
    }
    code.extend([
        v_lshlrev_b32(2, inline_u(2), 0), // v2 = tid * 4
        v_add_nc_u32(4, LIT, 0),
        1000, // v4 = tid + 1000
        v_xor_b32(6, inline_u(32), 0),    // v6 = tid ^ 32
        v_lshlrev_b32(5, inline_u(2), 6), // v5 = (tid ^ 32) * 4
        v_mov_b32(1, inline_u(5)),
        v_and_b32(7, inline_u(1), 0),
        v_cmp_eq_u32(inline_u(1), 7), // tid odd
        s_and_saveexec_b64(6, VCC),
    ]);
    let br = code.len();
    code.push(0);
    code.extend(ds_op(0x0D, 0, 0, 2, 4, 0)); // ds_write_b32 v2, v4
    code.extend(ds_op(0x36, 0, 0, 5, 0, 1)); // ds_read_b32 v1, v5
    code[br] = s_cbranch_execz((code.len() - (br + 1)) as i16);
    code.push(s_or_b64(EXEC_LO, EXEC_LO, 6));
    emit_uav_store(&mut code, 1, 0);
    code.push(S_ENDPGM);
    make_cs_program(&code, [threads, 1, 1], threads * 4, false)
}

fn lockstep_lds_in_branch_expected(threads: u32) -> Vec<u32> {
    (0..threads)
        .map(|t| if t & 1 == 1 { (t ^ 32) + 1000 } else { 5 })
        .collect()
}

/// `if (wave == 0 && lane odd) { v1 = bcnt1(exec) + readfirstlane(lane * 8 +
/// wave) }`: with 128 threads the second wave has no lane in the branch.
fn empty_wave_in_branch_program(threads: u32) -> NativeProgram {
    let mut code = lane_prologue();
    code.extend([
        v_and_b32(7, inline_u(1), 2),     // lane & 1
        v_lshlrev_b32(9, inline_u(1), 3), // wave * 2
        v_add_nc_u32(9, vgpr(7), 9),
        v_mov_b32(1, LIT),
        1000,
        v_cmp_eq_u32(inline_u(1), 9), // wave 0, lane odd
        s_and_saveexec_b64(6, VCC),
    ]);
    let br = code.len();
    code.push(0);
    code.push(s_bcnt1_i32_b64(4, EXEC_LO));
    code.push(v_readfirstlane_b32(10, 4));
    code.push(v_mov_b32(1, 4));
    code.push(v_add_nc_u32(1, 10, 1));
    code[br] = s_cbranch_execz((code.len() - (br + 1)) as i16);
    code.push(s_or_b64(EXEC_LO, EXEC_LO, 6));
    emit_uav_store(&mut code, 1, 0);
    code.push(S_ENDPGM);
    make_cs_program(&code, [threads, 1, 1], 0, false)
}

fn empty_wave_in_branch_expected(threads: u32) -> Vec<u32> {
    (0..threads)
        .map(|t| {
            let (wave, lane) = (t >> 6, t & 63);
            if wave == 0 && lane & 1 == 1 {
                // 32 odd lanes in the wave; the first is lane 1 (8).
                32 + 8
            } else {
                1000
            }
        })
        .collect()
}

#[test]
fn divergent_loops_and_branches_with_wave_operations_lower_in_emulate() {
    for threads in [64, 128] {
        for (name, prog) in [
            ("mid exit", mid_exit_loop_program(threads)),
            ("nested", nested_divergent_loop_program(threads)),
            ("ballot", ballot_in_branch_and_loop_program(threads)),
            ("cross lane", cross_lane_in_branch_program(threads)),
            ("lockstep lds", lockstep_lds_in_branch_program(threads)),
            ("empty wave", empty_wave_in_branch_program(threads)),
        ] {
            let shader = emit_cs(&prog, Wave64Mode::Emulate)
                .unwrap_or_else(|e| panic!("{name} ({threads} threads): {e}"));
            check_dxil(&shader, name);
        }
    }
}

/// `if (lane odd) { v1 = permlanex16(lane + 500, lane n takes lane n ^ 1 of the
/// other row) }` with FI / BC as given: the source lane is even, so off.
fn permlanex16_in_branch_program(threads: u32, fi: bool, bc: bool) -> NativeProgram {
    let mut code = lane_prologue();
    let mut word0 = v_permlanex16_b32(1, 4, 20, 21);
    word0[0] |= (u32::from(fi) | u32::from(bc) << 1) << 11;
    code.extend([
        v_add_nc_u32(4, LIT, 2), // v4 = lane + 500
        500,
        s_mov_b32(20, LIT),
        0x6745_2301,
        s_mov_b32(21, LIT),
        0xEFCD_AB89,
        v_mov_b32(1, LIT),
        1000,
        v_and_b32(7, inline_u(1), 2),
        v_cmp_eq_u32(inline_u(1), 7), // lane odd
        s_and_saveexec_b64(6, VCC),
    ]);
    let br = code.len();
    code.push(0);
    code.extend(word0);
    code[br] = s_cbranch_execz((code.len() - (br + 1)) as i16);
    code.push(s_or_b64(EXEC_LO, EXEC_LO, 6));
    emit_uav_store(&mut code, 1, 0);
    code.push(S_ENDPGM);
    make_cs_program(&code, [threads, 1, 1], 0, false)
}

fn permlanex16_in_branch_expected(threads: u32, fi: bool, bc: bool) -> Vec<u32> {
    (0..threads)
        .map(|t| {
            let lane = t & 63;
            if lane & 1 == 0 {
                return 1000;
            }
            let src = (lane & !31) | ((lane & 16) ^ 16) | ((lane & 15) ^ 1);
            if fi {
                src + 500
            } else if bc {
                0
            } else {
                1000
            }
        })
        .collect()
}

#[test]
fn permlanex16_lowers_in_emulate() {
    for threads in [64, 128] {
        let prog = permlanex16_in_branch_program(threads, false, false);
        let shader = emit_cs(&prog, Wave64Mode::Emulate)
            .unwrap_or_else(|e| panic!("{threads} threads: {e}"));
        check_dxil(&shader, "permlanex16");
    }
}

mod gpu {
    use std::sync::{Mutex, OnceLock};

    use super::*;
    use crate::exact64::emit::host::{Dispatch, Draw, Host, Root, View};

    fn with_hosts(f: impl FnOnce(&[(String, Host)])) {
        static HOSTS: OnceLock<Mutex<Vec<(String, Host)>>> = OnceLock::new();
        let _one = super::super::valu_tests::ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let lock = HOSTS.get_or_init(|| Mutex::new(Host::all_adapters()));
        let guard = lock.lock().unwrap_or_else(|e| e.into_inner());
        f(&guard);
    }

    fn run_on_hosts_modes(
        name: &str,
        prog: &NativeProgram,
        groups: [u32; 3],
        uav_dwords: u32,
        expected: &[u32],
        expect_native_fail: bool,
        run_native_when_supported: bool,
    ) {
        with_hosts(|hosts| {
            assert!(!hosts.is_empty(), "expected at least one SM 6.6 adapter");
            for (adapter_name, host) in hosts {
                let mut modes = vec![Wave64Mode::Emulate];
                if host.supports_wave32() {
                    modes.push(Wave64Mode::Wave32);
                }
                if run_native_when_supported && host.supports_wave64() {
                    modes.push(Wave64Mode::Native);
                }
                if expect_native_fail && !host.supports_wave64() {
                    let native = emit_cs(prog, Wave64Mode::Native).unwrap();
                    assert!(
                        host.compute_pipeline(&native).is_err(),
                        "{name}: Native Wave64 unexpectedly built a PSO on {adapter_name}"
                    );
                }
                let (ops, min_lane, max_lane) = host.wave_lanes();
                eprintln!(
                    "[wave64_tests] {name}: adapter={adapter_name:?} wave=({ops}, {min_lane}..={max_lane}) modes={modes:?}"
                );
                for mode in modes {
                    let shader = emit_cs(prog, mode)
                        .unwrap_or_else(|e| panic!("{name} ({mode:?}, {adapter_name}): {e}"));
                    if let Ok(mut v) = super::super::validator::Validator::load(
                        super::super::validator::Validator::default_path().as_deref(),
                    ) {
                        v.validate(&mut shader.dxil.clone())
                            .unwrap_or_else(|e| panic!("{name} ({mode:?}) validate: {e}"));
                    }
                    let (root, pso) = host
                        .compute_pipeline(&shader)
                        .unwrap_or_else(|e| panic!("{name} ({mode:?}, {adapter_name}) PSO: {e}"));
                    let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
                    let mut roots = vec![(0, Root::Constants(ud))];
                    for (idx, &reg) in shader.root_srvs.iter().enumerate() {
                        let data = if reg == crate::exact64::emit::VIEW_STRIDES {
                            vec![4]
                        } else {
                            vec![0; 4]
                        };
                        roots.push((1 + idx as u32, Root::View { srv: true, data }));
                    }
                    let out = host
                        .dispatch(
                            &root,
                            &pso,
                            &Dispatch {
                                roots,
                                views: &[View::RawUav(uav_dwords)],
                                samplers: &[],
                                groups,
                            },
                        )
                        .unwrap_or_else(|e| {
                            panic!("{name} ({mode:?}, {adapter_name}) dispatch: {e}")
                        });
                    assert_eq!(
                        &out[0][..expected.len()],
                        expected,
                        "{name} ({mode:?}, {adapter_name}) output mismatch"
                    );
                }
            }
        });
    }

    fn run_on_hosts(
        name: &str,
        prog: &NativeProgram,
        groups: [u32; 3],
        uav_dwords: u32,
        expected: &[u32],
        expect_native_fail: bool,
    ) {
        run_on_hosts_modes(
            name,
            prog,
            groups,
            uav_dwords,
            expected,
            expect_native_fail,
            false,
        );
    }

    #[test]
    fn gpu_single_wave_lds_regions_are_ordered() {
        let plain: Vec<u32> = (0..64).map(|t| if t < 32 { t + 39 } else { 7 }).collect();
        let delayed: Vec<u32> = (0..64)
            .map(|t| {
                if t >= 32 {
                    return 7;
                }
                let mut x = t + 39;
                for _ in 0..4096 {
                    x = (x & 0xFF_FFFFu32).wrapping_mul(40503).wrapping_add(1);
                }
                x
            })
            .collect();
        for (name, prog, expected) in [
            (
                "delayed writer then divergent reader",
                lds_region_program(true, true, true),
                &delayed,
            ),
            (
                "divergent writer then divergent reader",
                lds_region_program(true, true, false),
                &plain,
            ),
            (
                "uniform writer then divergent reader",
                lds_region_program(false, true, false),
                &plain,
            ),
            ("same-block EXEC guards", guarded_lds_program(), &plain),
        ] {
            for _ in 0..5 {
                run_on_hosts(name, &prog, [1, 1, 1], 64, expected, true);
            }
        }
    }

    #[test]
    fn gpu_64_threads_mbcnt_ballot_and_high_ff1() {
        // v1 = mbcnt_64(-1) + bcnt1(tid > 34) * 1000 + ff1(tid > 34) * 100000
        // For tid in 0..64: active count = 29, first active bit = 35!
        let mut code = Vec::new();
        code.push(v_cmp_lt_u32(inline_u(34), 0)); // vcc = (34 < v0), bits 35..63 = 1
        code.push(s_bcnt1_i32_b64(4, VCC)); // s4 = 29
        code.push(s_ff1_i32_b64(5, VCC)); // s5 = 35
        code.extend_from_slice(&v_mbcnt_lo_u32_b32(1, NEG_ONE, inline_u(0)));
        code.extend_from_slice(&v_mbcnt_hi_u32_b32(1, NEG_ONE, vgpr(1)));
        // Pack: lane | (bcnt << 8) | (ff1 << 16)
        code.push(v_lshlrev_b32(2, inline_u(8), 4)); // wrong if vop2 takes s4 in vsrc1; put s4 in src0!
        code.pop();
        // s_mov + v_mov for shifts using SGPR in src0:
        code.push(s_mov_b32(6, LIT));
        code.push(1000);
        code.push(s_mov_b32(7, LIT));
        code.push(100_000);
        // Wait: keep integer multiply simple using shifts (s_lshl_b32) + v_or_b32!
        code.clear();
        code.push(v_cmp_lt_u32(inline_u(34), 0)); // bits 35..63 set -> bcnt=29, ff1=35
        code.push(s_bcnt1_i32_b64(4, VCC));
        code.push(s_ff1_i32_b64(5, VCC));
        code.extend_from_slice(&v_mbcnt_lo_u32_b32(1, NEG_ONE, inline_u(0)));
        code.extend_from_slice(&v_mbcnt_hi_u32_b32(1, NEG_ONE, vgpr(1)));
        // s6 = (s4 << 8) | (s5 << 16) = 29*256 + 35*65536
        code.push(sop2(4, 0x1E, 4, inline_u(8))); // s_lshl_b32 s4, s4, 8
        code.push(sop2(5, 0x1E, 5, inline_u(16))); // s_lshl_b32 s5, s5, 16
        code.push(sop2(4, 0x10, 4, 5)); // s_or_b32 s4, s4, s5
        code.push(v_or_b32(1, 4, 1));
        emit_uav_store(&mut code, 1, 0);
        code.push(S_ENDPGM);

        let prog = make_cs_program(&code, [64, 1, 1], 0, false);
        let expected: Vec<u32> = (0..64u32).map(|i| i | (29 << 8) | (35 << 16)).collect();
        run_on_hosts("mbcnt_ballot_ff1", &prog, [1, 1, 1], 64, &expected, true);
    }

    #[test]
    fn gpu_128_threads_whole_wave_active_and_prefix() {
        // Virtual wave 0 (0..63), virtual wave 1 (64..127).
        // Value = (tid & 63) + 1; mask out odd console lanes via v_cmpx / s_and_saveexec?
        // Even simpler: active lanes = (tid & 1) == 0 via v_cmp_eq_u32 + s_and_saveexec_b64!
        let mut code = vec![
            v_and_b32(2, inline_u(63), 0),   // v2 = tid & 63
            v_add_nc_u32(2, inline_u(1), 2), // v2 = (tid & 63) + 1
            v_and_b32(3, inline_u(1), 0),    // v3 = tid & 1
            v_cmp_eq_u32(inline_u(0), 3),    // vcc = even lanes
            s_and_saveexec_b64(4, VCC),      // only even lanes active! (s[4:5] = all)
            s_or_saveexec_b64(6, NEG_ONE),   // s[6:7] = saved even-lane mask
        ];
        code.extend_from_slice(&v_cndmask_b32(10, inline_u(0), vgpr(2), 6));
        for shift in [0x111u32, 0x112, 0x114, 0x118] {
            code.extend_from_slice(&dpp_row_shr(0x25, 10, 10, 10, shift, 0xF));
        }
        // Store intra-row inclusive prefix in v11 before cross-row additions:
        code.push(v_mov_b32(11, vgpr(10)));
        code.extend_from_slice(&v_readlane_b32(12, 10, inline_u(15)));
        code.extend_from_slice(&v_readlane_b32(13, 10, inline_u(31)));
        code.extend_from_slice(&v_readlane_b32(14, 10, inline_u(47)));
        code.extend_from_slice(&v_readlane_b32(15, 10, inline_u(63)));
        // Broadcast cross-row sums for active sum (v10) and prefix offsets (v4 = s12+s13, v5 = v4+s14):
        code.push(v_mov_b32(4, 12));
        code.push(v_add_nc_u32(4, 13, 4));
        code.push(v_mov_b32(5, 14));
        code.push(v_add_nc_u32(5, vgpr(4), 5)); // v5 = s12+s13+s14
        code.push(v_mov_b32(10, 15));
        code.push(v_add_nc_u32(10, vgpr(5), 10)); // v10 = full 64-lane Active sum!
        // Apply prefix offsets to rows 1, 2, 3:
        code.push(s_mov_b32(EXEC_LO, LIT));
        code.push(0xFFFF_0000);
        code.push(s_mov_b32(EXEC_HI, inline_u(0)));
        code.push(v_add_nc_u32(11, 12, 11));
        code.push(s_mov_b32(EXEC_LO, inline_u(0)));
        code.push(s_mov_b32(EXEC_HI, LIT));
        code.push(0x0000_FFFF);
        code.push(v_add_nc_u32(11, vgpr(4), 11));
        code.push(s_mov_b32(EXEC_HI, LIT));
        code.push(0xFFFF_0000);
        code.push(v_add_nc_u32(11, vgpr(5), 11));
        code.push(s_mov_b64(EXEC_LO, 6)); // restore EXEC -> end of whole-wave section!
        code.push(v_sub_nc_u32(12, vgpr(2), 11)); // Wait: v_sub_nc_u32 vdst, src0, vsrc1 = src0 - vsrc1 (Wait: gfx10 vop2 sub is vdst = src0 - vsrc1 or vsrc1 - src0? Make sure: use v11 + v12 in exclusive!)
        code.pop();
        // Safer: v_sub_nc_u32_e32 v12, v2, v11 or let's use vop3 / check lower.rs! Actually Active (v10) and Inclusive Prefix (v11) already test Shape::Active and Shape::Prefix!
        // Pack: active | (prefix_incl << 12) for active even lanes; 0xFFFF for odd lanes!
        code.push(v_lshlrev_b32(11, inline_u(12), 11));
        code.push(v_or_b32(1, vgpr(10), 11));
        code.push(s_mov_b64(EXEC_LO, 4)); // restore all lanes
        // Write 1 for inactive odd lanes via v_cndmask: actually pre-init v1 = 0 before branch, keep simple: only even lanes store, odd stay 0 in UAV!
        // Wait: just write all active lanes -> odd lanes remain 0 in zeroed RawUav!
        code.pop(); // don't restore yet so odd lanes don't overwrite with garbage
        emit_uav_store(&mut code, 1, 0);
        code.push(s_mov_b64(EXEC_LO, 4));
        code.push(S_ENDPGM);

        let prog = make_cs_program(&code, [128, 1, 1], 0, false);
        let even_sum_64: u32 = (0..64u32).filter(|l| l % 2 == 0).map(|l| l + 1).sum();
        let mut expected = Vec::with_capacity(128);
        for g in 0..128u32 {
            if g % 2 != 0 {
                expected.push(0);
            } else {
                let l = g & 63;
                let pref: u32 = (0..=l).filter(|k| k % 2 == 0).map(|k| k + 1).sum();
                expected.push(even_sum_64 | (pref << 12));
            }
        }
        run_on_hosts(
            "whole_wave_active_and_prefix",
            &prog,
            [1, 1, 1],
            128,
            &expected,
            true,
        );
    }

    #[test]
    fn gpu_96_threads_partial_virtual_wave() {
        // 96 threads: vwave 0 = 64 threads, vwave 1 = 32 threads (Half 0 only, NO Half 1!).
        // Tests both 64-lane ballot + Shape::Active across partial wave.
        let mut code = vec![
            v_and_b32(2, inline_u(63), 0),
            v_add_nc_u32(2, inline_u(1), 2), // 1..=64 in w0, 1..=32 in w1
            v_cmp_lt_u32(inline_u(15), 2),   // count lanes > 15: 49 in w0, 17 in w1
            s_bcnt1_i32_b64(4, VCC),
            s_or_saveexec_b64(6, NEG_ONE),
        ];
        code.extend_from_slice(&v_cndmask_b32(10, inline_u(0), vgpr(2), 6));
        for shift in [0x111u32, 0x112, 0x114, 0x118] {
            code.extend_from_slice(&dpp_row_shr(0x25, 10, 10, 10, shift, 0xF));
        }
        code.extend_from_slice(&v_readlane_b32(12, 10, inline_u(15)));
        code.extend_from_slice(&v_readlane_b32(13, 10, inline_u(31)));
        code.extend_from_slice(&v_readlane_b32(14, 10, inline_u(47)));
        code.extend_from_slice(&v_readlane_b32(15, 10, inline_u(63)));
        code.push(v_mov_b32(10, 12));
        code.push(v_add_nc_u32(10, 13, 10));
        code.push(v_add_nc_u32(10, 14, 10));
        code.push(v_add_nc_u32(10, 15, 10));
        code.push(s_mov_b64(EXEC_LO, 6));

        code.push(sop2(4, 0x1E, 4, inline_u(16))); // s_lshl_b32 s4, s4, 16 (bcnt << 16)
        code.push(v_or_b32(1, 4, 10));
        emit_uav_store(&mut code, 1, 0);
        code.push(S_ENDPGM);

        let prog = make_cs_program(&code, [96, 1, 1], 0, false);
        let mut expected = Vec::with_capacity(96);
        let sum0: u32 = (1..=64).sum();
        let sum1: u32 = (1..=32).sum();
        for g in 0..96u32 {
            if g < 64 {
                expected.push(sum0 | (49 << 16));
            } else {
                expected.push(sum1 | (17 << 16));
            }
        }
        run_on_hosts(
            "partial_virtual_wave_96",
            &prog,
            [1, 1, 1],
            96,
            &expected,
            true,
        );
    }

    #[test]
    fn gpu_128_threads_readlane_writelane_spills_and_firstlane() {
        // Tests:
        // 1. Dynamic cross-half v_readlane_b32 reading partner = (tid ^ 33) & 63
        // 2. Non-uniform v_readfirstlane_b32 picking lane 0 (0 in w0, 64 in w1)
        // 3. Constant-lane SGPR spill via v_writelane_b32(_, _, 35) -> v_readlane_b32(_, 35) even inside divergent if!
        let mut code = Vec::new();
        // Spill s12 = 77 into v3 lane 35; keep in v1 when tid < 48 (divergent branch!)
        code.push(s_mov_b32(12, LIT));
        code.push(77);
        code.extend_from_slice(&v_writelane_b32(3, 12, inline_u(35)));
        code.push(v_mov_b32(4, inline_u(0)));
        code.push(v_cmp_gt_u32(inline_u(48), 0)); // 48 > v0 i.e. v0 < 48
        code.push(s_and_saveexec_b64(6, VCC));
        let br_idx = code.len();
        code.push(0);
        code.extend_from_slice(&v_readlane_b32(13, 3, inline_u(35)));
        code.push(v_mov_b32(4, 13));
        let after = code.len();
        code[br_idx] = s_cbranch_execz((after - (br_idx + 1)) as i16);
        code.push(s_or_b64(EXEC_LO, EXEC_LO, 6));

        // Non-uniform v_readfirstlane_b32 of tid (`v0`): 0 for w0, 64 for w1!
        code.push(v_readfirstlane_b32(14, 0));

        // Dynamic v_readlane_b32 reading (tid ^ 33) & 63 across halves:
        code.push(v_mov_b32(5, vgpr(0)));
        code.push(vop2(0x1D, 5, inline_u(33), 5)); // v_xor_b32 v5, 33, v5
        code.push(v_and_b32(5, inline_u(63), 5));
        code.extend_from_slice(&v_readlane_b32(15, 0, vgpr(5)));

        // Result = spill (4) + readfirstlane (s14 << 8) + dynamic_readlane (s15 << 16)
        code.push(sop2(14, 0x1E, 14, inline_u(8)));
        code.push(v_add_nc_u32(1, 14, 4));
        code.push(sop2(15, 0x1E, 15, inline_u(16)));
        code.push(v_add_nc_u32(1, 15, 1));
        emit_uav_store(&mut code, 1, 0);
        code.push(S_ENDPGM);

        let prog = make_cs_program(&code, [128, 1, 1], 0, false);
        let expected: Vec<u32> = (0..128u32)
            .map(|g| {
                let wbase = g & !63;
                let spill = if g < 48 { 77 } else { 0 };
                let first = wbase;
                let partner = wbase | ((g ^ 33) & 63);
                spill + (first << 8) + (partner << 16)
            })
            .collect();
        run_on_hosts(
            "readlane_spills_firstlane",
            &prog,
            [1, 1, 1],
            128,
            &expected,
            true,
        );
    }

    #[test]
    fn gpu_128_threads_ds_swizzle_and_dpp_row_mask() {
        // 1. ds_swizzle_b32 XOR 2 within 32-lane halves (offset = 0x1F | (2 << 10) = 0x81F)
        // 2. v_add_nc_u32_dpp row_shr:1 bound_ctrl:1 with row_mask = 0xA (rows 1 & 3 enabled, rows 0 & 2 masked!)
        let mut code = Vec::new();
        code.extend_from_slice(&ds_op(0x35, 0x1F, 0x08, 0, 0, 2)); // v2 = swizzle(v0, xor 2)
        code.push(v_mov_b32(3, inline_u(1))); // fallback 1 on masked rows 0 and 2!
        // src1=v0, src0=v0 (dpp row_shr:1 -> prev_tid), row_mask=0xA (bits 1 and 3!), bank_mask=0xF, bound_ctrl=1
        code.extend_from_slice(&dpp_row_shr(0x25, 3, 0, 0, 0x111, 0xA));
        code.push(v_lshlrev_b32(3, inline_u(16), 3));
        code.push(v_or_b32(1, vgpr(2), 3));
        emit_uav_store(&mut code, 1, 0);
        code.push(S_ENDPGM);

        let prog = make_cs_program(&code, [128, 1, 1], 0, false);
        let expected: Vec<u32> = (0..128u32)
            .map(|g| {
                let swz = g ^ 2;
                let row = (g & 63) >> 4;
                let dpp = if row == 1 || row == 3 {
                    if (g & 15) == 0 { g } else { g + (g - 1) }
                } else {
                    1
                };
                swz | (dpp << 16)
            })
            .collect();
        run_on_hosts(
            "ds_swizzle_dpp_rowmask",
            &prog,
            [1, 1, 1],
            128,
            &expected,
            true,
        );
    }

    #[test]
    fn gpu_64_threads_lockstep_lds_without_s_barrier() {
        // 64 threads sharing LDS without `s_barrier` (assumes single Wave64 wave on console).
        let mut code = Vec::new();
        code.push(v_lshlrev_b32(2, inline_u(2), 0)); // write addr = tid * 4
        code.push(v_add_nc_u32(3, inline_u(7), 0)); // val = tid + 7
        code.extend_from_slice(&ds_op(0x0D, 0, 0, 2, 3, 0)); // ds_write_b32 addr=v2, data=v3
        code.push(vop2(0x1D, 4, inline_u(32), 0)); // partner = tid ^ 32 (crosses Wave32 half!)
        code.push(v_lshlrev_b32(4, inline_u(2), 4)); // read addr = partner * 4
        code.extend_from_slice(&ds_op(0x36, 0, 0, 4, 0, 1)); // ds_read_b32 v1, addr=v4
        emit_uav_store(&mut code, 1, 0);
        code.push(S_ENDPGM);

        let prog = make_cs_program(&code, [64, 1, 1], 512, false);
        let expected: Vec<u32> = (0..64u32).map(|i| (i ^ 32) + 7).collect();
        run_on_hosts(
            "lockstep_lds_no_barrier",
            &prog,
            [1, 1, 1],
            64,
            &expected,
            false,
        );
    }

    #[test]
    fn gpu_128_threads_uniform_loop_cross_half_ballot() {
        // Uniform scalar loop for s10 in 0..4 performing 64-lane ballot every iteration.
        let mut code = Vec::new();
        code.push(s_mov_b32(10, inline_u(0))); // loop counter
        code.push(v_mov_b32(1, inline_u(0))); // accumulator
        let loop_top = code.len();
        code.push(v_and_b32(2, inline_u(63), 0));
        code.push(v_add_nc_u32(2, 10, 2)); // (tid & 63) + iter
        code.push(v_cmp_lt_u32(inline_u(40), 2)); // count increases by 1 each iteration!
        code.push(s_bcnt1_i32_b64(4, VCC));
        code.push(v_add_nc_u32(1, 4, 1));
        code.push(s_add_i32(10, 10, inline_u(1)));
        code.push(s_cmp_lt_u32(10, inline_u(4)));
        let br = code.len();
        code.push(s_cbranch_scc1(
            (loop_top as isize - (br + 1) as isize) as i16,
        ));
        emit_uav_store(&mut code, 1, 0);
        code.push(S_ENDPGM);

        let prog = make_cs_program(&code, [128, 1, 1], 0, false);
        // iter 0: >40 in 0..=63 => 23; iter 1: 24; iter 2: 25; iter 3: 26 => sum = 98
        let expected = vec![23 + 24 + 25 + 26; 128];
        run_on_hosts(
            "uniform_loop_ballots",
            &prog,
            [1, 1, 1],
            128,
            &expected,
            true,
        );
    }

    #[test]
    fn gpu_32_threads_wave32_fast_path_with_high_lane_spill() {
        // 32-thread Wave64 compute shader: Wave32 uses single wave + zero LDS scratch,
        // while still resolving constant-lane SGPR spill at lane 45 (>= 32).
        let mut code = Vec::new();
        code.push(s_mov_b32(12, LIT));
        code.push(123);
        code.extend_from_slice(&v_writelane_b32(3, 12, inline_u(45)));
        code.extend_from_slice(&v_readlane_b32(13, 3, inline_u(45)));
        code.push(v_add_nc_u32(1, 13, 0));
        emit_uav_store(&mut code, 1, 0);
        code.push(S_ENDPGM);

        let prog = make_cs_program(&code, [32, 1, 1], 0, false);
        let expected: Vec<u32> = (0..32u32).map(|i| 123 + i).collect();
        run_on_hosts(
            "wave32_32_threads_spill",
            &prog,
            [1, 1, 1],
            32,
            &expected,
            true,
        );
    }

    #[test]
    fn gpu_128_threads_cross_block_ballot_and_constant_mask_guard() {
        // Block 0 defines `vcc` (`tid & 63 >= 40`), branches uniformly via s_branch (block split)
        // to block 1 where `s_bcnt1` consumes cross-block `vcc`, plus `Atom::Constant` branch mask (`-16` -> lanes 4..63).
        let mut code = vec![
            v_and_b32(2, inline_u(63), 0),
            v_cmp_lt_u32(inline_u(39), 2), // vcc = (tid & 63) >= 40 (24 bits)
            sopp(0x02, 0),                 // s_branch +0 -> block boundary!
            // Block 1:
            s_bcnt1_i32_b64(4, VCC),    // s4 = 24
            s_and_saveexec_b64(8, 208), // 208 = inline -16 (bits 4..63 active in Wave64)
        ];
        let br_idx = code.len();
        code.push(0); // s_cbranch_execz -> after
        code.push(v_add_nc_u32(1, 4, 0));
        emit_uav_store(&mut code, 1, 0);
        let after = code.len();
        code[br_idx] = s_cbranch_execz((after - (br_idx + 1)) as i16);
        code.push(s_or_b64(EXEC_LO, EXEC_LO, 8));
        code.push(S_ENDPGM);

        let prog = make_cs_program(&code, [128, 1, 1], 0, false);
        let expected: Vec<u32> = (0..128u32)
            .map(|g| if (g & 63) >= 4 { 24 + g } else { 0 })
            .collect();
        run_on_hosts(
            "cross_block_and_const_mask",
            &prog,
            [1, 1, 1],
            128,
            &expected,
            true,
        );
    }

    #[test]
    fn gpu_64_threads_empty_exec_readfirstlane_returns_lane0() {
        // GFX10 v_readfirstlane_b32 with EXEC == 0 returns lane 0's value (`v2[0] = 7`).
        let mut code = vec![
            v_add_nc_u32(2, inline_u(7), 0), // lane 0 = 7, lane 32 = 39
            s_mov_b64(6, EXEC_LO),
            s_mov_b64(EXEC_LO, inline_u(0)),
            v_readfirstlane_b32(4, 2),
            s_mov_b64(EXEC_LO, 6),
            v_mov_b32(1, 4),
        ];
        emit_uav_store(&mut code, 1, 0);
        code.push(S_ENDPGM);

        let prog = make_cs_program(&code, [64, 1, 1], 0, false);
        let expected = vec![7u32; 64];
        run_on_hosts(
            "empty_exec_readfirstlane",
            &prog,
            [1, 1, 1],
            64,
            &expected,
            false,
        );
    }

    #[test]
    fn gpu_64_threads_float_whole_wave_sum_lane0_uav_x4() {
        let expected = [
            528.0f32.to_bits(),
            1.0f32.to_bits(),
            2.0f32.to_bits(),
            3.0f32.to_bits(),
        ];
        for (label, half1_nonzero, guard) in [
            ("h0_tid_eq_0_branch", false, Lane0Guard::TidEqZeroBranch),
            ("h1_tid_eq_0_branch", true, Lane0Guard::TidEqZeroBranch),
            ("h0_tid_eq_0_inline", false, Lane0Guard::TidEqZeroInline),
            ("h1_tid_eq_0_inline", true, Lane0Guard::TidEqZeroInline),
        ] {
            let prog = whole_wave_float_sum_uav_x4_program([64, 1, 1], 0, half1_nonzero, guard);
            run_on_hosts_modes(label, &prog, [1, 1, 1], 4, &expected, true, true);
        }
    }

    #[test]
    fn gpu_64_threads_float_whole_wave_sum_readfirstlane_and_mbcnt_lane0_guards() {
        let expected = [
            528.0f32.to_bits(),
            1.0f32.to_bits(),
            2.0f32.to_bits(),
            3.0f32.to_bits(),
        ];
        for (label, half1_nonzero, guard) in [
            (
                "h0_readfirstlane_guard",
                false,
                Lane0Guard::ReadFirstLaneBranch,
            ),
            (
                "h1_readfirstlane_guard",
                true,
                Lane0Guard::ReadFirstLaneBranch,
            ),
            ("h0_mbcnt_guard", false, Lane0Guard::MbcntEqZeroBranch),
            ("h1_mbcnt_guard", true, Lane0Guard::MbcntEqZeroBranch),
        ] {
            let prog = whole_wave_float_sum_uav_x4_program([64, 1, 1], 0, half1_nonzero, guard);
            run_on_hosts_modes(label, &prog, [1, 1, 1], 4, &expected, true, true);
        }
    }

    #[test]
    fn gpu_uniform_guard_readfirstlane_masked_lane0_uav_x4() {
        // Uniform-block s_and_saveexec_b64 (without s_cbranch_execz) masking off lane 0:
        // tests v_readfirstlane_b32 guard handling on !emulating_wave64() (Wave32 <=32 threads
        // and AMD Native 64-lane waves including hi ballot extraction) as well as Emulate/Wave32.
        let build_guard_prog =
            |threads: [u32; 3], min_active: u32, mask_63: bool, force_wave64: bool| {
                let mut code = Vec::new();
                if force_wave64 {
                    // Use v_mbcnt so console_lanes = true and Native enforces WaveSize(64).
                    code.extend_from_slice(&v_mbcnt_lo_u32_b32(14, NEG_ONE, inline_u(0)));
                    code.extend_from_slice(&v_mbcnt_hi_u32_b32(14, NEG_ONE, vgpr(14)));
                }
                code.push(s_mov_b32(1, inline_u(0)));
                code.extend_from_slice(&s_load_dwordx4(8, 0));
                if mask_63 {
                    code.push(v_and_b32(2, inline_u(63), 0));
                } else {
                    code.push(v_mov_b32(2, vgpr(0)));
                }
                code.push(v_cvt_f32_u32(16, vgpr(0)));
                for (reg, f) in [(17u32, 1.0f32), (18, 2.0), (19, 3.0)] {
                    code.push(v_mov_b32(reg, LIT));
                    code.push(f.to_bits());
                }
                // Write each virtual 64-lane wave's result to dword offset (tid >> 6) * 4.
                code.push(vop2(0x16, 15, inline_u(6), 0)); // v_lshrrev_b32 v15, 6, v0
                code.push(v_lshlrev_b32(15, inline_u(4), 15));
                code.push(v_cmp_lt_u32(inline_u(min_active - 1), 2));
                code.push(s_and_saveexec_b64(4, VCC));
                code.push(v_readfirstlane_b32(14, 0));
                code.push(v_cmp_eq_u32(14, 0));
                code.push(s_and_saveexec_b64(6, VCC));
                code.extend_from_slice(&buffer_store_dwordx4(16, 15, 8));
                code.push(s_or_b64(EXEC_LO, EXEC_LO, 6));
                code.push(s_or_b64(EXEC_LO, EXEC_LO, 4));
                code.push(S_ENDPGM);
                make_cs_program(&code, threads, 0, false)
            };

        for (label, threads, min_active, mask_63, force_wave64, expected) in [
            (
                "32_guard_ge16",
                [32, 1, 1],
                16,
                false,
                false,
                vec![
                    16.0f32.to_bits(),
                    1.0f32.to_bits(),
                    2.0f32.to_bits(),
                    3.0f32.to_bits(),
                ],
            ),
            (
                "64_guard_ge32",
                [64, 1, 1],
                32,
                false,
                true,
                vec![
                    32.0f32.to_bits(),
                    1.0f32.to_bits(),
                    2.0f32.to_bits(),
                    3.0f32.to_bits(),
                ],
            ),
            (
                "64_guard_ge33",
                [64, 1, 1],
                33,
                false,
                true,
                vec![
                    33.0f32.to_bits(),
                    1.0f32.to_bits(),
                    2.0f32.to_bits(),
                    3.0f32.to_bits(),
                ],
            ),
            (
                "128_guard_lane_ge32",
                [128, 1, 1],
                32,
                true,
                true,
                vec![
                    32.0f32.to_bits(),
                    1.0f32.to_bits(),
                    2.0f32.to_bits(),
                    3.0f32.to_bits(),
                    96.0f32.to_bits(),
                    1.0f32.to_bits(),
                    2.0f32.to_bits(),
                    3.0f32.to_bits(),
                ],
            ),
        ] {
            let prog = build_guard_prog(threads, min_active, mask_63, force_wave64);
            let uav_dwords = expected.len() as u32;
            run_on_hosts_modes(
                label,
                &prog,
                [1, 1, 1],
                uav_dwords,
                &expected,
                force_wave64,
                true,
            );
        }

        // Combined whole-wave float reduction + uniform-block s_and_saveexec_b64 (tid >= 32):
        // Half 0 is masked off before the whole-wave section, so sum(33..=64) = 1552.0 and
        // v_readfirstlane_b32 under the same guard picks lane 32 to write [1552.0, 1.0, 2.0, 3.0].
        let mut red_code = vec![
            v_add_nc_u32(2, inline_u(1), 0),
            v_cvt_f32_u32(2, vgpr(2)),
            v_cmp_lt_u32(inline_u(31), 0),
            s_and_saveexec_b64(4, VCC),
            s_or_saveexec_b64(6, NEG_ONE),
        ];
        red_code.extend_from_slice(&v_cndmask_b32(10, inline_u(0), vgpr(2), 6));
        for shift in [0x111u32, 0x112, 0x114, 0x118] {
            red_code.extend_from_slice(&dpp_row_shr(0x03, 10, 10, 10, shift, 0xF));
        }
        red_code.extend_from_slice(&v_permlanex16_b32(11, 10, NEG_ONE, NEG_ONE));
        red_code.push(v_add_f32(10, vgpr(10), 11));
        red_code.extend_from_slice(&v_readlane_b32(12, 10, inline_u(15)));
        red_code.extend_from_slice(&v_readlane_b32(13, 10, inline_u(63)));
        red_code.push(v_mov_b32(10, 12));
        red_code.push(v_add_f32(10, 13, 10));
        red_code.push(s_mov_b64(EXEC_LO, 6));
        red_code.push(s_mov_b32(1, inline_u(0)));
        red_code.extend_from_slice(&s_load_dwordx4(8, 0));
        red_code.push(v_mov_b32(16, vgpr(10)));
        for (reg, f) in [(17u32, 1.0f32), (18, 2.0), (19, 3.0)] {
            red_code.push(v_mov_b32(reg, LIT));
            red_code.push(f.to_bits());
        }
        red_code.push(v_mov_b32(15, inline_u(0)));
        red_code.push(v_readfirstlane_b32(14, 0));
        red_code.push(v_cmp_eq_u32(14, 0));
        red_code.push(s_and_saveexec_b64(6, VCC));
        red_code.extend_from_slice(&buffer_store_dwordx4(16, 15, 8));
        red_code.push(s_or_b64(EXEC_LO, EXEC_LO, 6));
        red_code.push(s_or_b64(EXEC_LO, EXEC_LO, 4));
        red_code.push(S_ENDPGM);
        let red_prog = make_cs_program(&red_code, [64, 1, 1], 0, false);
        let red_expected = [
            1552.0f32.to_bits(),
            1.0f32.to_bits(),
            2.0f32.to_bits(),
            3.0f32.to_bits(),
        ];
        run_on_hosts_modes(
            "64_guard_ge32_whole_wave_sum",
            &red_prog,
            [1, 1, 1],
            4,
            &red_expected,
            true,
            true,
        );

        // Guard with no active lanes (tid >= 64 on [32,1,1] and [64,1,1]) must fall back to lane 0.
        for (threads, force_wave64) in [([32, 1, 1], false), ([64, 1, 1], true)] {
            let mut empty_guard = Vec::new();
            if force_wave64 {
                empty_guard.extend_from_slice(&v_mbcnt_lo_u32_b32(3, NEG_ONE, inline_u(0)));
                empty_guard.extend_from_slice(&v_mbcnt_hi_u32_b32(3, NEG_ONE, vgpr(3)));
            }
            empty_guard.extend([
                v_add_nc_u32(2, inline_u(7), 0),
                v_cmp_lt_u32(inline_u(63), 0),
                s_and_saveexec_b64(4, VCC),
                v_readfirstlane_b32(14, 2),
                s_or_b64(EXEC_LO, EXEC_LO, 4),
                v_mov_b32(1, 14),
            ]);
            emit_uav_store(&mut empty_guard, 1, 0);
            empty_guard.push(S_ENDPGM);
            let prog = make_cs_program(&empty_guard, threads, 0, false);
            let expected = vec![7u32; threads[0] as usize];
            run_on_hosts_modes(
                "uniform_guard_empty_fallback_lane0",
                &prog,
                [1, 1, 1],
                threads[0],
                &expected,
                force_wave64,
                true,
            );
        }
    }

    #[test]
    fn gpu_wave_section_does_not_swallow_side_effecting_store() {
        // A buffer_store_dword between s_or_saveexec_b64(-1) and s_mov_b64(exec, saved) must not
        // be recognized as a pure whole-wave section and dropped; all 64 lanes must write 1..=64.
        let mut code = vec![
            s_or_saveexec_b64(4, NEG_ONE),
            v_add_nc_u32(1, inline_u(1), 0),
        ];
        emit_uav_store(&mut code, 1, 0);
        code.push(s_mov_b64(EXEC_LO, 4));
        code.push(S_ENDPGM);

        let prog = make_cs_program(&code, [64, 1, 1], 0, false);
        let expected: Vec<u32> = (1..=64).collect();
        run_on_hosts_modes(
            "section_with_uav_store",
            &prog,
            [1, 1, 1],
            64,
            &expected,
            false,
            true,
        );
    }

    #[test]
    fn gpu_readfirstlane_native_wave64_without_mbcnt() {
        // Without v_mbcnt/v_readlane/dpp, non-uniform or guarded v_readfirstlane_b32 on a
        // [64,1,1] CS must still enforce WaveSize(64) in Wave64Mode::Native so AMD Native
        // does not split the 64-thread wave into two Wave32 waves.
        // 1. tid >= 16 uniform-block guard + v_readfirstlane_b32 + 16B UAV write -> [16.0, 1.0, 2.0, 3.0].
        let mut guard_code = vec![s_mov_b32(1, inline_u(0))];
        guard_code.extend_from_slice(&s_load_dwordx4(8, 0));
        guard_code.push(v_cvt_f32_u32(16, vgpr(0)));
        for (reg, f) in [(17u32, 1.0f32), (18, 2.0), (19, 3.0)] {
            guard_code.push(v_mov_b32(reg, LIT));
            guard_code.push(f.to_bits());
        }
        guard_code.extend([
            v_mov_b32(15, inline_u(0)),
            v_cmp_lt_u32(inline_u(15), 0),
            s_and_saveexec_b64(4, VCC),
            v_readfirstlane_b32(14, 0),
            v_cmp_eq_u32(14, 0),
            s_and_saveexec_b64(6, VCC),
        ]);
        guard_code.extend_from_slice(&buffer_store_dwordx4(16, 15, 8));
        guard_code.extend([
            s_or_b64(EXEC_LO, EXEC_LO, 6),
            s_or_b64(EXEC_LO, EXEC_LO, 4),
            S_ENDPGM,
        ]);
        let guard_prog = make_cs_program(&guard_code, [64, 1, 1], 0, false);
        let guard_expected = [
            16.0f32.to_bits(),
            1.0f32.to_bits(),
            2.0f32.to_bits(),
            3.0f32.to_bits(),
        ];
        run_on_hosts_modes(
            "64_guard_ge16_no_mbcnt",
            &guard_prog,
            [1, 1, 1],
            4,
            &guard_expected,
            true,
            true,
        );

        // 2. Conditional write under tid >= 32 guard + v_readfirstlane_b32 readback across all
        // 64 lanes without v_mbcnt -> [99; 64] (Wave32 split would yield [11; 32] + [99; 32]).
        let mut cond_code = vec![
            v_mov_b32(2, inline_u(11)),
            v_cmp_lt_u32(inline_u(31), 0),
            s_and_saveexec_b64(4, VCC),
            v_mov_b32(2, LIT),
            99,
            v_readfirstlane_b32(14, 2),
            s_or_b64(EXEC_LO, EXEC_LO, 4),
            v_mov_b32(1, 14),
        ];
        emit_uav_store(&mut cond_code, 1, 0);
        cond_code.push(S_ENDPGM);
        let cond_prog = make_cs_program(&cond_code, [64, 1, 1], 0, false);
        let cond_expected = [99u32; 64];
        run_on_hosts_modes(
            "64_cond_write_readfirstlane_no_mbcnt",
            &cond_prog,
            [1, 1, 1],
            64,
            &cond_expected,
            true,
            true,
        );

        // 3. Unguarded non-uniform v_readfirstlane_b32(v0 + 10) across 64 lanes without v_mbcnt -> [10; 64].
        let mut unguarded_code = vec![
            v_add_nc_u32(2, inline_u(10), 0),
            v_readfirstlane_b32(14, 2),
            v_mov_b32(1, 14),
        ];
        emit_uav_store(&mut unguarded_code, 1, 0);
        unguarded_code.push(S_ENDPGM);
        let unguarded_prog = make_cs_program(&unguarded_code, [64, 1, 1], 0, false);
        let unguarded_expected = [10u32; 64];
        run_on_hosts_modes(
            "64_unguarded_readfirstlane_no_mbcnt",
            &unguarded_prog,
            [1, 1, 1],
            64,
            &unguarded_expected,
            true,
            true,
        );
    }

    fn validate_dxil(shader: &HostShader) {
        if let Ok(mut v) = super::super::validator::Validator::load(
            super::super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut shader.dxil.clone())
                .unwrap_or_else(|e| panic!("DXIL validation failed ({:?}): {e}", shader.stage));
        }
    }

    fn sig_element(
        name: &str,
        index: u32,
        kind: u8,
        comp: u8,
        row: Option<u32>,
        cols: u8,
        usage: u8,
    ) -> SigElement {
        SigElement {
            name: name.into(),
            index,
            kind,
            comp,
            interp: 0,
            row,
            col: 0,
            cols,
            usage,
            half: false,
        }
    }

    fn fullscreen_vs_readfirstlane(wave64: bool) -> (Vec<u8>, NativeProgram) {
        let mut code = vec![
            v_and_b32(1, inline_u(1), 0),
            v_lshlrev_b32(1, inline_u(2), 1),
            v_cvt_f32_u32(1, vgpr(1)),
            v_add_f32(1, LIT, 1),
            (-1.0f32).to_bits(),
            v_lshrrev_b32(2, inline_u(1), 0),
            v_lshlrev_b32(2, inline_u(2), 2),
            v_cvt_f32_u32(2, vgpr(2)),
            v_sub_f32(2, LIT, 2),
            1.0f32.to_bits(),
            v_mov_b32(3, inline_u(0)),
            v_mov_b32(4, LIT),
            1.0f32.to_bits(),
            v_cmp_lt_u32(inline_u(0), 0),
            if wave64 {
                s_and_saveexec_b64(4, VCC)
            } else {
                s_and_saveexec_b32(4, VCC)
            },
            v_readfirstlane_b32(14, 0),
            if wave64 {
                s_or_b64(EXEC_LO, EXEC_LO, 4)
            } else {
                s_or_b32(EXEC_LO, EXEC_LO, 4)
            },
            v_cvt_f32_u32(5, 14),
            v_mov_b32(6, LIT),
            1.0f32.to_bits(),
            v_mov_b32(7, LIT),
            2.0f32.to_bits(),
            v_mov_b32(8, LIT),
            3.0f32.to_bits(),
        ];
        code.extend_from_slice(&exp(12, 0xF, false, false, 1, 2, 3, 4));
        code.extend_from_slice(&exp(32, 0xF, true, false, 5, 6, 7, 8));
        code.push(S_ENDPGM);

        let osg1 = container::signature(
            &[
                sig_element("SV_Position", 0, 3, 9, Some(0), 4, 15),
                sig_element("TEXCOORD", 0, 0, 9, Some(1), 4, 15),
            ],
            false,
        );
        let dxbc = crate::exact64::dxbc::build(&[(FourCC::OSG1, &osg1)]);
        let native = NativeProgram {
            stage: HwStage::Vs,
            wave64,
            vgprs: 16,
            layout: Vec::new(),
            used: Vec::new(),
            io: StageIo::Vs {
                inputs: Vec::new(),
                outputs: vec![VsOutput {
                    usage: Usage::Generic,
                    row: 1,
                    param: 0,
                    fp16: false,
                    pack2: false,
                }],
            },
            code: code.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        (dxbc, native)
    }

    fn texcoord0_passthrough_ps() -> (Vec<u8>, NativeProgram) {
        let mut code = vec![
            v_interp_mov_f32(0, 0, 0, 2),
            v_interp_mov_f32(1, 0, 1, 2),
            v_interp_mov_f32(2, 0, 2, 2),
            v_interp_mov_f32(3, 0, 3, 2),
        ];
        code.extend_from_slice(&exp(0, 0xF, true, true, 0, 1, 2, 3));
        code.push(S_ENDPGM);

        let isg1 = container::signature(&[sig_element("TEXCOORD", 0, 0, 9, Some(1), 4, 15)], true);
        let osg1 =
            container::signature(&[sig_element("SV_Target", 0, 16, 9, Some(0), 4, 15)], false);
        let dxbc = crate::exact64::dxbc::build(&[(FourCC::ISG1, &isg1), (FourCC::OSG1, &osg1)]);
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 4,
            layout: Vec::new(),
            used: Vec::new(),
            io: StageIo::Ps {
                inputs: vec![PsInput {
                    usage: Usage::Generic,
                    row: 1,
                    mask: 0xF,
                    attr: 0,
                    flat: true,
                    fp16: false,
                    pack2: false,
                    row2: 1,
                }],
                input_addr: 0,
                db_shader_control: 0,
                patches: Vec::new(),
            },
            code: code.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        (dxbc, native)
    }

    fn ps_pos_x_readfirstlane(wave64: bool) -> (Vec<u8>, NativeProgram) {
        let mut code = vec![
            v_mov_b32(1, vgpr(0)),
            v_cmp_le_f32(LIT, 0),
            2.0f32.to_bits(),
            if wave64 {
                s_and_saveexec_b64(4, VCC)
            } else {
                s_and_saveexec_b32(4, VCC)
            },
            v_readfirstlane_b32(6, 0),
            v_mov_b32(1, 6),
            if wave64 {
                s_or_b64(EXEC_LO, EXEC_LO, 4)
            } else {
                s_or_b32(EXEC_LO, EXEC_LO, 4)
            },
            v_mov_b32(2, LIT),
            1.0f32.to_bits(),
        ];
        code.extend_from_slice(&exp(0, 0xF, true, true, 1, 0, 2, 2));
        code.push(S_ENDPGM);

        let isg1 =
            container::signature(&[sig_element("SV_Position", 0, 3, 9, Some(0), 4, 1)], true);
        let osg1 =
            container::signature(&[sig_element("SV_Target", 0, 16, 9, Some(0), 4, 15)], false);
        let dxbc = crate::exact64::dxbc::build(&[(FourCC::ISG1, &isg1), (FourCC::OSG1, &osg1)]);
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64,
            vgprs: 8,
            layout: Vec::new(),
            used: Vec::new(),
            io: StageIo::Ps {
                inputs: Vec::new(),
                input_addr: 1 << 8,
                db_shader_control: 0,
                patches: Vec::new(),
            },
            code: code.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        (dxbc, native)
    }

    #[test]
    fn vs_ps_readfirstlane_host_wave_contract() {
        let (ps_pass_dxbc, ps_pass_native) = texcoord0_passthrough_ps();
        let ps_pass = emit_program(&ps_pass_dxbc, &ps_pass_native, Wave64Mode::Native).unwrap();
        validate_dxil(&ps_pass);

        for wave64 in [false, true] {
            let (vs_dxbc, vs_native) = fullscreen_vs_readfirstlane(wave64);
            let vs = emit_program(&vs_dxbc, &vs_native, Wave64Mode::Native).unwrap();
            validate_dxil(&vs);
            assert_eq!(psv_wave_range(&vs), (0, u32::MAX));
            assert_ne!(sfi0_flags(&vs) & 0x4000, 0);

            let (ps_dxbc, ps_native) = ps_pos_x_readfirstlane(wave64);
            let ps = emit_program(&ps_dxbc, &ps_native, Wave64Mode::Native).unwrap();
            validate_dxil(&ps);
            assert_eq!(psv_wave_range(&ps), (0, u32::MAX));
            assert_ne!(sfi0_flags(&ps) & 0x4000, 0);

            with_hosts(|hosts| {
                assert!(!hosts.is_empty(), "expected at least one SM 6.6 adapter");
                for (adapter_name, host) in hosts {
                    let (root, pso) = host
                        .pipeline(&vs, &ps_pass)
                        .unwrap_or_else(|e| panic!("VS wave64={wave64} ({adapter_name}) PSO: {e}"));
                    let vs_out = host
                        .draw(
                            &root,
                            &pso,
                            &Draw {
                                roots: Vec::new(),
                                textures: &[],
                                samplers: &[],
                                vertices: &[],
                                stride: 0,
                                size: (4, 1),
                                mesh_groups: 0,
                            },
                        )
                        .unwrap_or_else(|e| {
                            panic!("VS wave64={wave64} ({adapter_name}) draw: {e}")
                        });
                    assert_eq!(
                        vs_out,
                        vec![[1.0, 1.0, 2.0, 3.0]; 4],
                        "VS wave64={wave64} ({adapter_name}) pixel mismatch"
                    );

                    let (root, pso) = host
                        .pipeline(&vs, &ps)
                        .unwrap_or_else(|e| panic!("PS wave64={wave64} ({adapter_name}) PSO: {e}"));
                    let ps_out = host
                        .draw(
                            &root,
                            &pso,
                            &Draw {
                                roots: Vec::new(),
                                textures: &[],
                                samplers: &[],
                                vertices: &[],
                                stride: 0,
                                size: (8, 1),
                                mesh_groups: 0,
                            },
                        )
                        .unwrap_or_else(|e| {
                            panic!("PS wave64={wave64} ({adapter_name}) draw: {e}")
                        });
                    let xs: Vec<f32> = ps_out.iter().map(|px| px[0]).collect();
                    let (_, _, max_lane) = host.wave_lanes();
                    if max_lane >= 8 {
                        assert_eq!(
                            xs,
                            vec![0.5, 1.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5],
                            "PS wave64={wave64} ({adapter_name}) host-wave mismatch"
                        );
                    } else {
                        assert_eq!(max_lane, 4, "{adapter_name}");
                        assert_eq!(
                            xs,
                            vec![0.5, 1.5, 2.5, 2.5, 4.5, 4.5, 6.5, 6.5],
                            "PS wave64={wave64} ({adapter_name}) 4-lane quad-wave mismatch"
                        );
                    }
                }
            });
        }

        // StageIo::Ms: Input::MeshGroup (v5) with v_readfirstlane_b32 succeeds without regression.
        let mut ms_code = vec![v_readfirstlane_b32(2, 5)];
        ms_code.extend_from_slice(&exp(20, 0x1, true, false, 9, 0, 0, 0));
        ms_code.push(S_ENDPGM);
        let ms_native = NativeProgram {
            stage: HwStage::Ms,
            wave64: true,
            vgprs: 16,
            layout: Vec::new(),
            used: Vec::new(),
            io: StageIo::Ms(MeshIo {
                outputs: Vec::new(),
                clip_cntl: 0,
                threads: 64,
                max_vertices: 3,
                max_primitives: 1,
                topology: 2,
                lds_bytes: 0,
            }),
            code: ms_code.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let ms_dxbc = empty_dxbc();
        for mode in [Wave64Mode::Native, Wave64Mode::Wave32, Wave64Mode::Emulate] {
            let ms = emit_program(&ms_dxbc, &ms_native, mode)
                .unwrap_or_else(|e| panic!("MS MeshGroup readfirstlane failed in {mode:?}: {e}"));
            validate_dxil(&ms);
            assert_eq!(psv_wave_range(&ms), (0, u32::MAX));
        }
    }

    fn whole_wave_float_sum_multi_group_program(
        threads: [u32; 3],
        half1_nonzero: bool,
        guard: Lane0Guard,
    ) -> NativeProgram {
        assert!(threads[0] == 64 || threads[0] == 128);
        let vwaves = threads[0] / 64;
        let mut code = vec![
            v_and_b32(4, inline_u(63), 0),
            v_lshrrev_b32(5, inline_u(6), 0),
            s_lshl_b32(3, 2, inline_u(7)),
            v_lshlrev_b32(6, inline_u(4), 5),
            v_add_nc_u32(6, 3, 6),
        ];
        if half1_nonzero {
            code.push(v_cmp_lt_u32(inline_u(31), 4));
            code.push(v_subrev_nc_u32(3, inline_u(31), 4));
        } else {
            code.push(v_cmp_gt_u32(inline_u(32), 4));
            code.push(v_add_nc_u32(3, inline_u(1), 4));
        }
        code.push(v_add_nc_u32(3, vgpr(6), 3));
        code.push(v_cvt_f32_u32(3, vgpr(3)));
        code.extend_from_slice(&v_cndmask_b32(2, inline_u(0), vgpr(3), VCC));

        code.push(s_or_saveexec_b64(4, NEG_ONE));
        code.extend_from_slice(&v_cndmask_b32(10, inline_u(0), vgpr(2), 4));
        for shift in [0x111u32, 0x112, 0x114, 0x118] {
            code.extend_from_slice(&dpp_row_shr(0x03, 10, 10, 10, shift, 0xF));
        }
        code.extend_from_slice(&v_permlanex16_b32(11, 10, NEG_ONE, NEG_ONE));
        code.push(v_add_f32(10, vgpr(10), 11));
        code.extend_from_slice(&v_readlane_b32(12, 10, inline_u(15)));
        code.extend_from_slice(&v_readlane_b32(13, 10, inline_u(63)));
        code.push(v_mov_b32(10, 12));
        code.push(v_add_f32(10, 13, 10));
        code.push(s_mov_b64(EXEC_LO, 4));

        code.push(s_mov_b32(1, inline_u(0)));
        code.extend_from_slice(&s_load_dwordx4(8, 0));
        code.push(v_mov_b32(16, vgpr(10)));
        for (reg, f) in [(17u32, 1.0f32), (18, 2.0), (19, 3.0)] {
            code.push(v_mov_b32(reg, LIT));
            code.push(f.to_bits());
        }
        if vwaves == 2 {
            code.push(s_lshl_b32(3, 2, inline_u(1)));
        } else {
            code.push(s_mov_b32(3, 2));
        }
        code.push(v_add_nc_u32(15, 3, 5));
        code.push(v_lshlrev_b32(15, inline_u(4), 15));

        match guard {
            Lane0Guard::TidEqZeroBranch => {
                code.push(v_cmp_eq_u32(inline_u(0), 4));
                code.push(s_and_saveexec_b64(6, VCC));
                let br_idx = code.len();
                code.push(0);
                code.extend_from_slice(&buffer_store_dwordx4(16, 15, 8));
                let after = code.len();
                code[br_idx] = s_cbranch_execz((after - (br_idx + 1)) as i16);
                code.push(s_or_b64(EXEC_LO, EXEC_LO, 6));
            }
            Lane0Guard::ReadFirstLaneBranch => {
                code.push(v_readfirstlane_b32(14, 0));
                code.push(v_cmp_eq_u32(14, 0));
                code.push(s_and_saveexec_b64(6, VCC));
                let br_idx = code.len();
                code.push(0);
                code.extend_from_slice(&buffer_store_dwordx4(16, 15, 8));
                let after = code.len();
                code[br_idx] = s_cbranch_execz((after - (br_idx + 1)) as i16);
                code.push(s_or_b64(EXEC_LO, EXEC_LO, 6));
            }
            Lane0Guard::MbcntEqZeroBranch | Lane0Guard::TidEqZeroInline => unreachable!(),
        }
        code.push(S_ENDPGM);
        make_cs_program(&code, threads, 0, true)
    }

    #[test]
    fn gpu_multi_workgroup_4x1x1_wave_sections_float_sum() {
        for (threads, vwaves) in [([64, 1, 1], 1u32), ([128, 1, 1], 2u32)] {
            let mut expected = Vec::with_capacity((4 * vwaves * 4) as usize);
            for gid in 0..4u32 {
                for vwave in 0..vwaves {
                    let sum = 528.0f32 + 32.0 * (gid * 128 + vwave * 16) as f32;
                    expected.extend_from_slice(&[
                        sum.to_bits(),
                        1.0f32.to_bits(),
                        2.0f32.to_bits(),
                        3.0f32.to_bits(),
                    ]);
                }
            }
            for (label, half1_nonzero, guard) in [
                ("mg_h0_tid0", false, Lane0Guard::TidEqZeroBranch),
                ("mg_h1_tid0", true, Lane0Guard::TidEqZeroBranch),
                ("mg_h0_firstlane", false, Lane0Guard::ReadFirstLaneBranch),
                ("mg_h1_firstlane", true, Lane0Guard::ReadFirstLaneBranch),
            ] {
                let prog = whole_wave_float_sum_multi_group_program(threads, half1_nonzero, guard);
                let expected_lds = match guard {
                    Lane0Guard::ReadFirstLaneBranch => 4 * vwaves,
                    _ => 2 * vwaves,
                };
                assert_eq!(
                    lower_cs(&prog, Wave64Mode::Wave32).unwrap().lds_dwords,
                    expected_lds
                );
                let name = format!("{label}_{}", threads[0]);
                run_on_hosts_modes(
                    &name,
                    &prog,
                    [4, 1, 1],
                    expected.len() as u32,
                    &expected,
                    true,
                    true,
                );
            }
        }
    }

    #[test]
    fn gpu_nested_3level_saveexec_andn2_readfirstlane() {
        // 1. Uniform-block 3-level s_and_saveexec_b64 + s_andn2_saveexec_b64 nesting.
        let mut uniform_code = vec![
            v_and_b32(2, inline_u(63), 0),
            v_mov_b32(3, inline_u(0)),
            v_cmp_lt_u32(inline_u(7), 2),
            s_and_saveexec_b64(4, VCC),
            v_cmp_gt_u32(inline_u(48), 2),
            s_and_saveexec_b64(6, VCC),
            v_and_b32(4, inline_u(1), 2),
            v_cmp_eq_u32(inline_u(0), 4),
            s_and_saveexec_b64(8, VCC),
            v_readfirstlane_b32(10, 2),
            v_mov_b32(3, 10),
            v_add_nc_u32(3, inline_u(1), 3),
            s_andn2_saveexec_b64(8, 8),
            v_readfirstlane_b32(10, 2),
            v_mov_b32(3, 10),
            v_add_nc_u32(3, inline_u(1), 3),
            s_or_b64(EXEC_LO, EXEC_LO, 8),
            s_andn2_saveexec_b64(6, 6),
            v_readfirstlane_b32(10, 2),
            v_mov_b32(3, 10),
            v_add_nc_u32(3, inline_u(1), 3),
            s_or_b64(EXEC_LO, EXEC_LO, 6),
            s_andn2_saveexec_b64(4, 4),
            v_readfirstlane_b32(10, 2),
            v_mov_b32(3, 10),
            v_add_nc_u32(3, inline_u(1), 3),
            s_or_b64(EXEC_LO, EXEC_LO, 4),
        ];
        emit_uav_store(&mut uniform_code, 3, 0);
        uniform_code.push(S_ENDPGM);

        let uniform_prog = make_cs_program(&uniform_code, [128, 1, 1], 0, false);
        let uniform_expected: Vec<u32> = (0..128u32)
            .map(|g| {
                let l = g & 63;
                if l < 8 {
                    1
                } else if l < 48 {
                    if l % 2 == 0 { 9 } else { 10 }
                } else {
                    49
                }
            })
            .collect();
        run_on_hosts_modes(
            "nested_3level_uniform_saveexec_andn2",
            &uniform_prog,
            [1, 1, 1],
            128,
            &uniform_expected,
            true,
            true,
        );

        // 2. Divergent-branch 3-level nesting (s_cbranch_execz + s_andn2_saveexec_b64)
        // merging into phi + outer uniform-join and cross-half guarded v_readfirstlane_b32.
        let mut div_code = vec![
            v_and_b32(2, inline_u(63), 0),
            v_mov_b32(3, inline_u(0)),
            v_cmp_gt_u32(inline_u(48), 2),
            s_and_saveexec_b64(4, VCC),
        ];
        let br_l1_else = div_code.len();
        div_code.push(0);
        div_code.push(v_cmp_gt_u32(inline_u(32), 2));
        div_code.push(s_and_saveexec_b64(6, VCC));
        let br_l2_else = div_code.len();
        div_code.push(0);
        div_code.push(v_and_b32(4, inline_u(1), 2));
        div_code.push(v_cmp_eq_u32(inline_u(0), 4));
        div_code.push(s_and_saveexec_b64(8, VCC));
        let br_l3_else = div_code.len();
        div_code.push(0);
        div_code.push(v_mov_b32(3, inline_u(31)));
        let l3_else = div_code.len();
        div_code[br_l3_else] = s_cbranch_execz((l3_else - (br_l3_else + 1)) as i16);
        div_code.push(s_andn2_saveexec_b64(8, 8));
        let br_l3_join = div_code.len();
        div_code.push(0);
        div_code.push(v_mov_b32(3, inline_u(32)));
        let l3_join = div_code.len();
        div_code[br_l3_join] = s_cbranch_execz((l3_join - (br_l3_join + 1)) as i16);
        div_code.push(s_or_b64(EXEC_LO, EXEC_LO, 8));
        let l2_else = div_code.len();
        div_code[br_l2_else] = s_cbranch_execz((l2_else - (br_l2_else + 1)) as i16);
        div_code.push(s_andn2_saveexec_b64(6, 6));
        let br_l2_join = div_code.len();
        div_code.push(0);
        div_code.push(v_mov_b32(3, inline_u(22)));
        let l2_join = div_code.len();
        div_code[br_l2_join] = s_cbranch_execz((l2_join - (br_l2_join + 1)) as i16);
        div_code.push(s_or_b64(EXEC_LO, EXEC_LO, 6));
        let l1_else = div_code.len();
        div_code[br_l1_else] = s_cbranch_execz((l1_else - (br_l1_else + 1)) as i16);
        div_code.push(s_andn2_saveexec_b64(4, 4));
        let br_l1_join = div_code.len();
        div_code.push(0);
        div_code.push(v_mov_b32(3, inline_u(12)));
        let l1_join = div_code.len();
        div_code[br_l1_join] = s_cbranch_execz((l1_join - (br_l1_join + 1)) as i16);
        div_code.push(s_or_b64(EXEC_LO, EXEC_LO, 4));

        div_code.extend([
            v_readfirstlane_b32(14, 3),
            v_cmp_lt_u32(inline_u(31), 2),
            s_and_saveexec_b64(4, VCC),
            v_readfirstlane_b32(15, 3),
            s_or_b64(EXEC_LO, EXEC_LO, 4),
            s_lshl_b32(14, 14, inline_u(8)),
            s_lshl_b32(15, 15, inline_u(16)),
            s_or_b32(14, 14, 15),
            v_or_b32(1, 14, 3),
        ]);
        emit_uav_store(&mut div_code, 1, 0);
        div_code.push(S_ENDPGM);

        let div_prog = make_cs_program(&div_code, [128, 1, 1], 0, false);
        let div_expected: Vec<u32> = (0..128u32)
            .map(|g| {
                let l = g & 63;
                let leaf = if l < 32 {
                    if l % 2 == 0 { 31 } else { 32 }
                } else if l < 48 {
                    22
                } else {
                    12
                };
                leaf | (31 << 8) | (22 << 16)
            })
            .collect();
        run_on_hosts_modes(
            "nested_3level_divergent_phi_readfirstlane",
            &div_prog,
            [1, 1, 1],
            128,
            &div_expected,
            true,
            true,
        );
    }

    /// Runs `prog` lowered as `Wave64Mode::Emulate` on every adapter and
    /// compares the first `expected.len()` dwords of its UAV.
    fn run_emulate(name: &str, prog: &NativeProgram, uav_dwords: u32, expected: &[u32]) {
        with_hosts(|hosts| {
            assert!(!hosts.is_empty(), "expected at least one SM 6.6 adapter");
            for (adapter_name, host) in hosts {
                eprintln!("[wave64_tests] {name}: adapter={adapter_name:?} mode=Emulate");
                let shader = emit_cs(prog, Wave64Mode::Emulate)
                    .unwrap_or_else(|e| panic!("{name} ({adapter_name}): {e}"));
                check_dxil(&shader, name);
                let (root, pso) = host
                    .compute_pipeline(&shader)
                    .unwrap_or_else(|e| panic!("{name} ({adapter_name}) PSO: {e}"));
                let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
                let mut roots = vec![(0, Root::Constants(ud))];
                for (idx, &reg) in shader.root_srvs.iter().enumerate() {
                    let data = if reg == crate::exact64::emit::VIEW_STRIDES {
                        vec![4]
                    } else {
                        vec![0; 4]
                    };
                    roots.push((1 + idx as u32, Root::View { srv: true, data }));
                }
                let out = host
                    .dispatch(
                        &root,
                        &pso,
                        &Dispatch {
                            roots,
                            views: &[View::RawUav(uav_dwords)],
                            samplers: &[],
                            groups: [1, 1, 1],
                        },
                    )
                    .unwrap_or_else(|e| panic!("{name} ({adapter_name}) dispatch: {e}"));
                assert_eq!(
                    &out[0][..expected.len()],
                    expected,
                    "{name} ({adapter_name}) output mismatch"
                );
            }
        });
    }

    #[test]
    fn gpu_divergent_readfirstlane_in_an_if_reads_each_waves_first_active_lane() {
        for threads in [64, 128] {
            // Wave 1 (128 threads) starts at lane 10; with shift 6 its lanes
            // are all below the threshold: no lane there runs the read.
            for shift in [2, 6] {
                run_emulate(
                    &format!("divergent_readfirstlane_if_{threads}_{shift}"),
                    &divergent_readfirstlane_if_program(threads, shift),
                    threads,
                    &divergent_readfirstlane_if_expected(threads, shift),
                );
            }
        }
    }

    #[test]
    fn gpu_carry_mask_beside_a_divergent_readfirstlane() {
        for threads in [64, 128] {
            run_emulate(
                &format!("carry_mask_divergent_readfirstlane_{threads}"),
                &carry_mask_in_divergent_readfirstlane_program(threads),
                threads,
                &carry_mask_in_divergent_readfirstlane_expected(threads),
            );
        }
    }

    #[test]
    fn gpu_half_wave_sum_in_the_consoles_order() {
        for (threads, active) in [(64, 64), (64, 50), (128, 64)] {
            run_emulate(
                &format!("half_wave_sum_{threads}_{active}"),
                &half_wave_sum_program(threads, active),
                threads,
                &half_wave_sum_expected(threads, active),
            );
        }
    }

    #[test]
    fn gpu_double_wave_sum_of_exact_terms() {
        for (threads, active, k_hi) in [
            (64, 64, 0x3FF0_0000),
            (64, 50, 0x3FD8_0000),
            (128, 0, 0x3FF0_0000),
            (128, 64, 0x3FD8_0000),
        ] {
            run_emulate(
                &format!("double_wave_sum_{threads}_{active}"),
                &double_wave_sum_program(threads, active, k_hi),
                threads,
                &double_wave_sum_expected(threads, active, k_hi),
            );
        }
    }

    #[test]
    fn gpu_f64_instructions() {
        run_emulate("f64_valu", &f64_valu_program(64), 64, &f64_valu_expected(64));
    }

    #[test]
    fn gpu_divergent_readfirstlane_after_an_early_return() {
        for threads in [64, 128] {
            run_emulate(
                &format!("divergent_readfirstlane_early_return_{threads}"),
                &divergent_readfirstlane_early_return_program(threads),
                threads,
                &divergent_readfirstlane_early_return_expected(threads),
            );
        }
    }

    #[test]
    fn gpu_divergent_readfirstlane_in_a_uniform_loop() {
        for threads in [64, 128] {
            run_emulate(
                &format!("divergent_readfirstlane_uniform_loop_{threads}"),
                &divergent_readfirstlane_uniform_loop_program(threads),
                threads,
                &divergent_readfirstlane_uniform_loop_expected(threads),
            );
        }
    }

    /// The console's result of [`divergent_trip_loop_program`] per thread.
    fn divergent_trip_loop_expected(threads: u32) -> Vec<u32> {
        (0..threads)
            .map(|t| {
                let (wave, lane) = (t >> 6, t & 63);
                // Round r (1-based) runs the lanes whose count is at least r;
                // the first of them is lane r - 1 of the lanes below 4.
                (1..=(lane & 3) + 1).map(|r| (r - 1) * 8 + wave).sum()
            })
            .collect()
    }

    #[test]
    fn gpu_divergent_readfirstlane_in_a_divergent_trip_loop() {
        for threads in [64, 128] {
            run_emulate(
                &format!("divergent_trip_loop_{threads}"),
                &divergent_trip_loop_program(threads),
                threads,
                &divergent_trip_loop_expected(threads),
            );
        }
    }

    #[test]
    fn gpu_divergent_mid_loop_exit() {
        for threads in [64, 128] {
            run_emulate(
                &format!("mid_exit_loop_{threads}"),
                &mid_exit_loop_program(threads),
                threads,
                &mid_exit_loop_expected(threads),
            );
        }
    }

    #[test]
    fn gpu_nested_divergent_loops_with_readfirstlane() {
        for threads in [64, 128] {
            run_emulate(
                &format!("nested_divergent_loop_{threads}"),
                &nested_divergent_loop_program(threads),
                threads,
                &nested_divergent_loop_expected(threads),
            );
        }
    }

    #[test]
    fn gpu_ballot_in_a_divergent_branch_and_loop() {
        for threads in [64, 128] {
            run_emulate(
                &format!("ballot_branch_loop_{threads}"),
                &ballot_in_branch_and_loop_program(threads),
                threads,
                &ballot_in_branch_and_loop_expected(threads),
            );
        }
    }

    #[test]
    fn gpu_cross_lane_reads_in_a_divergent_branch() {
        for threads in [64, 128] {
            run_emulate(
                &format!("cross_lane_branch_{threads}"),
                &cross_lane_in_branch_program(threads),
                threads,
                &cross_lane_in_branch_expected(threads),
            );
        }
    }

    #[test]
    fn gpu_lockstep_lds_store_and_load_in_a_divergent_branch() {
        // One console wave: beyond it the per-thread form leaves lanes of one
        // wave to the program's own barriers (see `LDS_ALL_OVER_64`).
        for threads in [64] {
            run_emulate(
                &format!("lockstep_lds_branch_{threads}"),
                &lockstep_lds_in_branch_program(threads),
                threads,
                &lockstep_lds_in_branch_expected(threads),
            );
        }
    }

    #[test]
    fn gpu_a_wave_with_no_active_lane_in_a_divergent_branch() {
        for threads in [64, 128] {
            run_emulate(
                &format!("empty_wave_branch_{threads}"),
                &empty_wave_in_branch_program(threads),
                threads,
                &empty_wave_in_branch_expected(threads),
            );
        }
    }

    /// The BVH builder programs of `XboxBvhBuilder.dll` (`VICTORY_BVH_DLL`,
    /// read only) as `hash:offset:size,...` in `VICTORY_BVH_PROGRAMS`, run
    /// on the hardware adapter over title memory of `VICTORY_BVH_FILL`
    /// (default 0) dwords with root constants from `VICTORY_BVH_CONSTANTS`
    /// (four comma separated numbers), `VICTORY_BVH_GROUPS` groups: the time
    /// each takes and whether the device survives. Run under the GPU lock,
    /// with `VICTORY_SHADER_LOOP_LIMIT` (feature `diagnostics`) low enough
    /// that a loop that does not end cannot hold the device.
    #[test]
    #[ignore = "harness: needs VICTORY_BVH_DLL and VICTORY_BVH_PROGRAMS"]
    fn gpu_bvh_builder_programs_terminate() {
        let (Some(dll), Ok(list)) = (
            std::env::var_os("VICTORY_BVH_DLL"),
            std::env::var("VICTORY_BVH_PROGRAMS"),
        ) else {
            return;
        };
        let fill: u32 = std::env::var("VICTORY_BVH_FILL")
            .ok()
            .and_then(|x| x.parse().ok())
            .unwrap_or(0);
        let words: Vec<u32> = std::env::var("VICTORY_BVH_CONSTANTS")
            .unwrap_or_default()
            .split(',')
            .filter_map(|x| x.parse().ok())
            .collect();
        let groups: u32 = std::env::var("VICTORY_BVH_GROUPS")
            .ok()
            .and_then(|x| x.parse().ok())
            .unwrap_or(1);
        const TITLE: u64 = 0x12_3450_0000;
        let page_table = |view: u32, null: u32| -> Vec<u32> {
            let (hi, lo) = ((TITLE >> 32) as u32, TITLE as u32);
            let mut t = vec![0u32; 4096];
            t[(hi << 4 | lo >> 28) as usize] = 1;
            for _ in 0..2 * 4096 {
                t.extend([null, 0]);
            }
            let entry = 4096 + 2 * (4096 + (lo >> 16 & 0xFFF) as usize);
            t[entry..entry + 2].copy_from_slice(&[view, lo]);
            t
        };
        let data = std::fs::read(dll).unwrap();
        let _one = super::super::valu_tests::ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let Ok(host) = Host::new(false) else {
            eprintln!("no hardware device");
            return;
        };
        for item in list.split(',') {
            let mut f = item.split(':');
            let (name, off, size) = (f.next().unwrap(), f.next().unwrap(), f.next().unwrap());
            let off = usize::from_str_radix(off.trim_start_matches("0x"), 16).unwrap();
            let size: usize = size.parse().unwrap();
            let shader = translate_with(
                &data[off..off + size],
                Options {
                    wave64: Wave64Mode::Emulate,
                    ..Default::default()
                },
            )
            .unwrap_or_else(|e| panic!("{name}: {e}"));
            let (root, pso) = host
                .compute_pipeline(&shader)
                .unwrap_or_else(|e| panic!("{name}: PSO: {e}"));
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match u.root {
                    RootType::Constants if std::env::var_os("VICTORY_BVH_BOX").is_some() => {
                        match u.dword {
                            0 => TITLE as u32,
                            1 => (TITLE >> 32) as u32,
                            _ => 0,
                        }
                    }
                    RootType::Constants => words.get(u.dword as usize).copied().unwrap_or(0),
                    RootType::Descriptor => (TITLE >> (32 * u.dword)) as u32,
                    _ => 0,
                };
            }
            let fill_value: u32 = std::env::var("VICTORY_BVH_FILLVALUE")
                .ok()
                .and_then(|x| x.parse().ok())
                .unwrap_or(0);
            let mut memory = vec![fill_value; fill as usize];
            // `VICTORY_BVH_BOX=n`: the layout `Box32Rehydration` reads (the user
            // pointer is the title address; at +0x4c the two buffers it reads,
            // at +0 the byte offset of its first group): per-thread dwords whose
            // top byte is a child count `1 + lane % n`, and a node table of
            // `fill_value` dwords.
            if let Some(n) = std::env::var("VICTORY_BVH_BOX").ok().and_then(|x| x.parse::<u32>().ok()) {
                memory.iter_mut().for_each(|w| *w = 0);
                let (r1, r2) = (TITLE + 0x1000, TITLE + 0x2000);
                memory[0x4c / 4] = r1 as u32;
                memory[0x4c / 4 + 1] = (r1 >> 32) as u32;
                memory[0x4c / 4 + 2] = r2 as u32;
                memory[0x4c / 4 + 3] = (r2 >> 32) as u32;
                for lane in 0..1024u32 {
                    memory[0x1000 / 4 + lane as usize] = (1 + lane % n) << 24 | (lane + 1);
                }
                for w in memory[0x2000 / 4..].iter_mut() {
                    *w = fill_value;
                }
            }
            let roots = vec![
                (0, Root::Constants(ud)),
                (1, Root::View { srv: true, data: page_table(1, 3) }),
                (2, Root::Uav(2 * 4096 / 32)),
            ];
            let views = [
                View::RawUav(4),
                View::Memory(&memory),
                View::Pair,
                View::NullPair,
                View::Pair,
            ];
            let start = std::time::Instant::now();
            let out = host.dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots,
                    views: &views,
                    samplers: &[],
                    groups: [groups, 1, 1],
                },
            );
            eprintln!(
                "BVHGPU {name}: groups {groups} fill {fill}x{fill_value} constants {words:?}: {:?} in {:?}",
                out.as_ref().map(|_| "completed").map_err(|e| e.clone()),
                start.elapsed()
            );
        }
    }
    #[test]
    fn gpu_permlanex16_in_a_divergent_branch() {
        for threads in [64, 128] {
            for (fi, bc) in [(true, false), (false, false), (false, true)] {
                run_emulate(
                    &format!("permlanex16_{threads}_{fi}_{bc}"),
                    &permlanex16_in_branch_program(threads, fi, bc),
                    threads,
                    &permlanex16_in_branch_expected(threads, fi, bc),
                );
            }
        }
    }

    #[test]
    fn gpu_divergent_readfirstlane_in_nested_ifs() {
        for threads in [64, 128] {
            run_emulate(
                &format!("divergent_readfirstlane_nested_{threads}"),
                &divergent_readfirstlane_nested_program(threads),
                threads,
                &divergent_readfirstlane_nested_expected(threads),
            );
        }
    }
}

/// Corpus harness (not part of the normal run): translates every compute
/// container of a shader library directory (`VICTORY_CORPUS`) with
/// [`Wave64Mode::Emulate`] and writes one line per container to the file
/// `VICTORY_CORPUS_OUT`: `file<TAB>status<TAB>detail<TAB>predicated retry<TAB>dxil hash`.
/// Run: `cargo test --release -p victory-shader corpus_wave64 -- --ignored --nocapture`.
#[test]
#[ignore = "harness: needs VICTORY_CORPUS (a library directory) and VICTORY_CORPUS_OUT"]
fn corpus_wave64_emulate() {
    use std::hash::{Hash, Hasher};
    use std::io::Write;
    let (Some(dir), Some(out)) = (
        std::env::var_os("VICTORY_CORPUS"),
        std::env::var_os("VICTORY_CORPUS_OUT"),
    ) else {
        return;
    };
    let only = std::env::var("VICTORY_CORPUS_ONLY").ok();
    let mut validator = std::env::var_os("VICTORY_CORPUS_VALIDATE")
        .and_then(|_| super::validator::Validator::load(super::validator::Validator::default_path().as_deref()).ok());
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "dxbc"))
        .collect();
    files.sort();
    let mut report = std::fs::File::create(out).unwrap();
    for path in files {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        if only.as_ref().is_some_and(|o| !o.split(',').any(|x| x == name)) {
            continue;
        }
        let bytes = std::fs::read(&path).unwrap();
        let wave64 = (|| {
            let c = Container::parse(&bytes).ok()?;
            let dxil = c.part(FourCC::DXIL).or(c.part(FourCC::ACCC))?;
            let progs = crate::exact64::native_programs(dxil.data).ok()?;
            progs.iter().find(|p| p.stage == HwStage::Cs).map(|p| p.wave64)
        })()
        .unwrap_or(false);
        if !wave64 {
            continue;
        }
        let result = std::panic::catch_unwind(|| {
            translate_with(
                &bytes,
                Options {
                    wave64: Wave64Mode::Emulate,
                    ..Default::default()
                },
            )
        });
        let retry = super::lower::PREDICATED_ERROR
            .with(|e| e.borrow().clone())
            .unwrap_or_default();
        let (status, detail, hash) = match result {
            Ok(Ok(shader))
                if validator
                    .as_mut()
                    .is_some_and(|v| v.validate(&mut shader.dxil.clone()).is_err()) =>
            {
                let why = validator
                    .as_mut()
                    .and_then(|v| v.validate(&mut shader.dxil.clone()).err())
                    .map(|e| e.to_string().replace(['\n', '\t'], " "))
                    .unwrap_or_default();
                ("invalid".to_string(), why, String::new())
            }
            Ok(Ok(shader)) => {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                shader.dxil.hash(&mut h);
                ("ok".to_string(), format!("{} bytes", shader.dxil.len()), format!("{:016x}", h.finish()))
            }
            Ok(Err(ShaderError::Unsupported { offset, what })) => {
                ("refused".to_string(), format!("{offset:#x}: {what}"), String::new())
            }
            Ok(Err(e)) => ("error".to_string(), e.to_string(), String::new()),
            Err(_) => ("panic".to_string(), String::new(), String::new()),
        };
        writeln!(report, "{name}	{status}	{detail}	{retry}	{hash}").unwrap();
    }
}

/// The BVH builder programs of `XboxBvhBuilder.dll` (`VICTORY_BVH_DLL`, read
/// only), as `hash:offset:size,...` in `VICTORY_BVH_PROGRAMS`: translates
/// each in Emulate and prints what the emulation costs (group barriers and
/// real loops in the translation).
#[test]
#[ignore = "harness: needs VICTORY_BVH_DLL and VICTORY_BVH_PROGRAMS"]
fn bvh_builder_programs_translation_cost() {
    let (Some(dll), Ok(list)) = (
        std::env::var_os("VICTORY_BVH_DLL"),
        std::env::var("VICTORY_BVH_PROGRAMS"),
    ) else {
        return;
    };
    let data = std::fs::read(dll).unwrap();
    for item in list.split(',') {
        let mut f = item.split(':');
        let (name, off, size) = (f.next().unwrap(), f.next().unwrap(), f.next().unwrap());
        let off = usize::from_str_radix(off.trim_start_matches("0x"), 16).unwrap();
        let size: usize = size.parse().unwrap();
        if let Some((threads, lds)) = (|| {
            let c = Container::parse(&data[off..off + size]).ok()?;
            let dxil = c.part(FourCC::DXIL).or(c.part(FourCC::ACCC))?;
            let progs = crate::exact64::native_programs(dxil.data).ok()?;
            let p = progs.iter().find(|p| p.stage == HwStage::Cs)?;
            match p.io {
                StageIo::Cs(cs) => Some((cs.threads, cs.lds_bytes())),
                _ => None,
            }
        })() {
            eprintln!("BVH {name}: threads {threads:?}, LDS {lds} bytes");
        }
        let r = translate_with(
            &data[off..off + size],
            Options {
                wave64: Wave64Mode::Emulate,
                ..Default::default()
            },
        );
        let (barriers, loops) = super::lower::BARRIER_COUNT.with(|c| c.get());
        match r {
            Ok(sh) => eprintln!(
                "BVH {name}: translated, {} bytes of DXIL, {barriers} group barriers, {loops} real loops; layout cbv {:?} srv {:?} uav {:?} user_data {:?} bindings {:?} heap {} heap_writes {} memory {} memory_writes {}",
                sh.dxil.len(), sh.root_cbvs, sh.root_srvs, sh.root_uavs, sh.user_data, sh.bindings, sh.heap, sh.heap_writes, sh.memory, sh.memory_writes
            ),
            Err(e) => eprintln!("BVH {name}: {e}"),
        }
    }
}

#[test]
fn lockstep_lds_programs_have_a_barrier_before_each_access() {
    for threads in [64, 128] {
        let prog = lockstep_lds_in_branch_program(threads);
        emit_cs(&prog, Wave64Mode::Emulate).unwrap();
        let (barriers, loops) = super::lower::BARRIER_COUNT.with(|c| c.get());
        eprintln!("LOCKSTEP {threads}: {barriers} barriers, {loops} loops");
        // One for each of the two accesses at least, plus the loop's.
        assert!(barriers >= 2, "{threads}: {barriers}");
    }
}

/// Layout of library programs (`VICTORY_CORPUS`, `VICTORY_CORPUS_ONLY`):
/// threads, LDS, roots, bindings.
#[test]
#[ignore = "harness: needs VICTORY_CORPUS and VICTORY_CORPUS_ONLY"]
fn corpus_layout() {
    let (Some(dir), Ok(only)) = (
        std::env::var_os("VICTORY_CORPUS"),
        std::env::var("VICTORY_CORPUS_ONLY"),
    ) else {
        return;
    };
    for name in only.split(',') {
        let bytes = std::fs::read(std::path::Path::new(&dir).join(name)).unwrap();
        let c = Container::parse(&bytes).unwrap();
        let dxil = c.part(FourCC::DXIL).or(c.part(FourCC::ACCC)).unwrap();
        let progs = crate::exact64::native_programs(dxil.data).unwrap();
        let p = progs.iter().find(|p| p.stage == HwStage::Cs).unwrap();
        let StageIo::Cs(cs) = p.io else { continue };
        eprintln!("LAYOUT {name}: threads {:?} lds {} layout {:?} used {:?}", cs.threads, cs.lds_bytes(), p.layout, p.used);
        let sh = translate_with(&bytes, Options { wave64: Wave64Mode::Emulate, ..Default::default() }).unwrap();
        eprintln!("LAYOUT {name}: cbv {:?} srv {:?} uav {:?} user_data {:?} bindings {:?} heap {}/{}", sh.root_cbvs, sh.root_srvs, sh.root_uavs, sh.user_data, sh.bindings, sh.heap, sh.heap_writes);
    }
}
