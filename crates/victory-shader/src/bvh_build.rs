//! A CPU builder for the console's ray-tracing structures: what
//! `XboxBvhBuilder.dll` makes with GPU programs for 64-lane waves, made on
//! the host (`VICTORY_RT=cpu`, `victory-hle`'s `bvhbuilder`), in the node
//! formats [`crate::bvh`] reads. It follows the layout the FiveStar project
//! (GPL-3.0, `FiveStar.Shaders/Bvh`) found for the same title: a bottom-level
//! structure is a 64-byte header (dword 0 `0x8000_0006`, the root node
//! pointer at 0x30), box32 nodes (128 bytes, four children, f32 bounds) in
//! preorder, then one 64-byte triangle node per triangle; a top-level one is
//! a header, box32 nodes over the instances' world bounds, then 128-byte
//! instance nodes. The node formats are AMD's RT IP 1.1 ones (public in
//! Mesa's RADV, MIT).
//!
//! VERIFIED here (unit tests): that [`crate::bvh::intersect`], walked as the
//! title walks, finds the same closest triangle as a brute-force test over a
//! built structure. ASSUMED (not checked against the console's own output):
//! header dwords other than 0, 1, 3, 4-9, 10, 12, 13, 14, 15; the instance
//! node fields the shadow shader does not read (0x00-0x2f, 0x70); what the
//! walkers do with instance-node flag bits.
//!
//! Tree: Morton order of the box centres (63 bits), a binary split where
//! the codes' highest differing bit changes, collapsed to four children by
//! opening the child of the largest surface area until four are there.
//! Updates ([`refit`]) keep the tree and recompute bounds bottom-up.

use crate::bvh;

/// Header, box32 node, triangle node, instance node, in bytes.
pub const HEADER: usize = 0x40;
pub const BOX32: usize = 128;
pub const TRIANGLE: usize = 64;
pub const INSTANCE: usize = 128;
/// Node kinds (the low three bits of a node pointer).
pub const KIND_BOX32: u32 = 5;
pub const KIND_BOX16: u32 = 4;
/// A 16-bit-bounds box node.
pub const BOX16: usize = 64;
pub const KIND_INSTANCE: u32 = 6;
/// A root or child that points nowhere.
pub const NO_NODE: u32 = 0xffff_ffff;
/// Dword 0 of a bottom-level structure and of a top-level one.
pub const BLAS_FLAGS: u32 = 0x8000_0006;
pub const TLAS_FLAGS: u32 = 0x0000_0006;
/// Dword 15 with no vertex transform (every scale exponent 127).
const NO_TRANSFORM: u32 = 0x007f_7f7f;
/// `triangle_id` of a lone triangle: barycentric I from vertex slot 1, J from 2.
const TRIANGLE_ID: u32 = 0x9;

/// Bounds as min x y z, max x y z.
pub type Bounds = [f32; 6];

const EMPTY: Bounds = [f32::INFINITY, f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];

fn union(a: &Bounds, b: &Bounds) -> Bounds {
    [
        a[0].min(b[0]),
        a[1].min(b[1]),
        a[2].min(b[2]),
        a[3].max(b[3]),
        a[4].max(b[4]),
        a[5].max(b[5]),
    ]
}

fn area(b: &Bounds) -> f32 {
    let (x, y, z) = (b[3] - b[0], b[4] - b[1], b[5] - b[2]);
    if x < 0.0 || y < 0.0 || z < 0.0 {
        return 0.0;
    }
    x * y + y * z + z * x
}

/// Widens finite bounds by a few units in the last place, so the slab test's
/// rounding cannot lose a triangle's own edge.
fn pad(b: &Bounds) -> Bounds {
    let lo = |x: f32| x - x.abs() * 2.4e-7 - 1e-30;
    let hi = |x: f32| x + x.abs() * 2.4e-7 + 1e-30;
    [lo(b[0]), lo(b[1]), lo(b[2]), hi(b[3]), hi(b[4]), hi(b[5])]
}

fn finite(b: &Bounds) -> bool {
    b.iter().all(|x| x.is_finite()) && b[0] <= b[3] && b[1] <= b[4] && b[2] <= b[5]
}

/// One triangle: vertices (already transformed), the geometry it came from
/// and its index in it, and the geometry's flags byte.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tri {
    pub v: [[f32; 3]; 3],
    pub geometry: u32,
    pub primitive: u32,
    pub flags: u8,
}

impl Tri {
    pub fn bounds(&self) -> Bounds {
        let mut b = EMPTY;
        for v in &self.v {
            b = union(&b, &[v[0], v[1], v[2], v[0], v[1], v[2]]);
        }
        b
    }
}

/// A top-level instance (`D3D12_RAYTRACING_INSTANCE_DESC`, plus what the
/// bottom-level structure's header says).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Instance {
    /// Object to world, row-major 3x4.
    pub transform: [f32; 12],
    pub id: u32,
    pub mask: u8,
    pub contribution: u32,
    pub flags: u8,
    /// The bottom-level structure's address (256-byte aligned).
    pub blas: u64,
    /// Its root node pointer (header dword 12) and bounds (dwords 4-9).
    pub root: u32,
    pub bounds: Bounds,
}

// ---- the tree ------------------------------------------------------------

#[derive(Clone, Copy)]
enum Child {
    None,
    /// A leaf by position in the leaf order.
    Leaf(u32, Bounds),
    Box(u32, Bounds),
}

struct Binary {
    /// Per node: children (u32::MAX for a leaf), range in the order, bounds.
    left: Vec<u32>,
    right: Vec<u32>,
    lo: Vec<u32>,
    bounds: Vec<Bounds>,
}

struct Tree {
    /// Leaf indices (into the caller's list) in leaf order.
    order: Vec<u32>,
    boxes: Vec<[Child; 4]>,
}

fn spread(mut x: u64) -> u64 {
    x &= 0x1f_ffff;
    x = (x | x << 32) & 0x1f00000000ffff;
    x = (x | x << 16) & 0x1f0000ff0000ff;
    x = (x | x << 8) & 0x100f00f00f00f00f;
    x = (x | x << 4) & 0x10c30c30c30c30c3;
    x = (x | x << 2) & 0x1249249249249249;
    x
}

fn morton(b: &Bounds, low: [f32; 3], scale: [f32; 3]) -> u64 {
    let mut code = 0u64;
    for a in 0..3 {
        let c = 0.5 * (b[a] + b[3 + a]);
        let q = ((c - low[a]) * scale[a]).clamp(0.0, 2097151.0) as u64;
        code |= spread(q) << a;
    }
    code
}

impl Tree {
    /// A tree over leaves with finite bounds (`leaves[i]` for each `i` in
    /// `ids`), at least one.
    fn build(leaves: &[Bounds], ids: Vec<u32>) -> Tree {
        let n = ids.len();
        let mut centre_low = [f32::INFINITY; 3];
        let mut centre_high = [f32::NEG_INFINITY; 3];
        for &i in &ids {
            let b = &leaves[i as usize];
            for a in 0..3 {
                let c = 0.5 * (b[a] + b[3 + a]);
                centre_low[a] = centre_low[a].min(c);
                centre_high[a] = centre_high[a].max(c);
            }
        }
        let scale: [f32; 3] = std::array::from_fn(|a| {
            let e = centre_high[a] - centre_low[a];
            if e > 0.0 { 2097151.0 / e } else { 0.0 }
        });
        let mut keyed: Vec<(u64, u32)> = ids
            .iter()
            .map(|&i| (morton(&leaves[i as usize], centre_low, scale), i))
            .collect();
        keyed.sort_unstable();
        let order: Vec<u32> = keyed.iter().map(|k| k.1).collect();
        let codes: Vec<u64> = keyed.iter().map(|k| k.0).collect();
        let mut bin = Binary {
            left: Vec::with_capacity(2 * n),
            right: Vec::with_capacity(2 * n),
            lo: Vec::with_capacity(2 * n),
            bounds: Vec::with_capacity(2 * n),
        };
        let root = split(&mut bin, &codes, &order, leaves, 0, n);
        let mut boxes = Vec::with_capacity(n / 2 + 1);
        if bin.left[root as usize] == u32::MAX {
            // One leaf: a box with one child.
            boxes.push([Child::Leaf(0, bin.bounds[root as usize]), Child::None, Child::None, Child::None]);
        } else {
            emit_box(&bin, root, &mut boxes);
        }
        Tree { order, boxes }
    }
}

fn split(bin: &mut Binary, codes: &[u64], order: &[u32], leaves: &[Bounds], lo: usize, hi: usize) -> u32 {
    let id = bin.left.len() as u32;
    bin.left.push(u32::MAX);
    bin.right.push(u32::MAX);
    bin.lo.push(lo as u32);
    bin.bounds.push(EMPTY);
    if hi - lo == 1 {
        bin.bounds[id as usize] = leaves[order[lo] as usize];
        return id;
    }
    let (first, last) = (codes[lo], codes[hi - 1]);
    let mid = if first == last {
        (lo + hi) / 2
    } else {
        // The first position whose code has the highest differing bit set.
        let bit = 63 - (first ^ last).leading_zeros();
        let (mut a, mut b) = (lo, hi - 1);
        while b - a > 1 {
            let m = (a + b) / 2;
            if (codes[m] >> bit) & 1 == 0 {
                a = m;
            } else {
                b = m;
            }
        }
        b
    };
    let l = split(bin, codes, order, leaves, lo, mid);
    let r = split(bin, codes, order, leaves, mid, hi);
    bin.left[id as usize] = l;
    bin.right[id as usize] = r;
    bin.bounds[id as usize] = union(&bin.bounds[l as usize], &bin.bounds[r as usize]);
    id
}

fn emit_box(bin: &Binary, node: u32, boxes: &mut Vec<[Child; 4]>) -> u32 {
    let mut list = [node, u32::MAX, u32::MAX, u32::MAX];
    let mut n = 1;
    // Open the child of the largest surface area until there are four.
    let (l, r) = (bin.left[node as usize], bin.right[node as usize]);
    list[0] = l;
    list[1] = r;
    n += 1;
    while n < 4 {
        let mut best = None;
        for (k, &c) in list[..n].iter().enumerate() {
            if bin.left[c as usize] != u32::MAX {
                let a = area(&bin.bounds[c as usize]);
                if best.is_none_or(|(_, ba)| a > ba) {
                    best = Some((k, a));
                }
            }
        }
        let Some((k, _)) = best else { break };
        let c = list[k];
        list[k] = bin.left[c as usize];
        list[n] = bin.right[c as usize];
        n += 1;
    }
    let index = boxes.len() as u32;
    boxes.push([Child::None; 4]);
    let mut children = [Child::None; 4];
    for (k, &c) in list[..n].iter().enumerate() {
        let b = bin.bounds[c as usize];
        children[k] = if bin.left[c as usize] == u32::MAX {
            Child::Leaf(bin.lo[c as usize], b)
        } else {
            Child::Box(emit_box(bin, c, boxes), b)
        };
    }
    boxes[index as usize] = children;
    index
}

// ---- writing -------------------------------------------------------------

fn put(bytes: &mut [u8], at: usize, v: u32) {
    bytes[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

fn putf(bytes: &mut [u8], at: usize, v: f32) {
    put(bytes, at, v.to_bits());
}

/// The pointer to a node `offset` bytes from the structure's start.
pub fn pointer(offset: usize, kind: u32) -> u32 {
    (offset as u32 >> 3) | kind
}

/// A half-precision float's bits for `x` rounded toward zero (finite `x`;
/// beyond the range: the largest finite half).
fn half_toward_zero(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = (bits >> 16 & 0x8000) as u16;
    let a = x.abs();
    if a >= 65520.0 {
        return sign | 0x7bff;
    }
    if a < 6.103_515_6e-5 {
        // Subnormal halves: multiples of 2^-24.
        return sign | (a * 16_777_216.0) as u16;
    }
    let e = (bits >> 23 & 0xff) as i32 - 127 + 15;
    sign | ((e as u16) << 10) | ((bits >> 13) & 0x3ff) as u16
}

/// The half-precision float at or below (`up` false) or at or above (`up`)
/// `x`: infinity past the range, so a box only ever grows.
pub fn half_bound(x: f32, up: bool) -> u16 {
    if x.is_nan() {
        return 0x7e00;
    }
    if x.is_infinite() || x.abs() >= 65520.0 {
        // Past the largest half: the way that grows the box is infinity; the
        // other way is the largest finite half.
        return if (x > 0.0) == up { if x > 0.0 { 0x7c00 } else { 0xfc00 } } else if x > 0.0 { 0x7bff } else { 0xfbff };
    }
    let z = half_toward_zero(x);
    if bvh::half_to_f32(z) == x {
        return z;
    }
    // `z` is on the zero side of `x`: right when rounding toward zero is the
    // wanted direction (positive and down, negative and up); else the next
    // magnitude.
    if (x > 0.0) != up { z } else { z + 1 }
}

#[allow(clippy::too_many_arguments)]
fn write_box(bytes: &mut [u8], at: usize, children: &[Child; 4], box_at: usize, wide: bool, leaf_at: usize, leaf_size: usize, leaf_kind: u32) {
    let (size, kind) = if wide { (BOX32, KIND_BOX32) } else { (BOX16, KIND_BOX16) };
    for (k, c) in children.iter().enumerate() {
        let (p, b) = match *c {
            Child::None => (NO_NODE, EMPTY),
            Child::Leaf(pos, b) => (pointer(leaf_at + pos as usize * leaf_size, leaf_kind), b),
            Child::Box(i, b) => (pointer(box_at + i as usize * size, kind), b),
        };
        put(bytes, at + 4 * k, p);
        let b = if finite(&b) { pad(&b) } else { b };
        put_bounds(bytes, at, k, wide, &b);
    }
}

fn put_bounds(bytes: &mut [u8], at: usize, k: usize, wide: bool, b: &Bounds) {
    for (a, x) in b.iter().enumerate() {
        if wide {
            putf(bytes, at + 16 + 24 * k + 4 * a, *x);
        } else {
            let o = at + 16 + 12 * k + 2 * a;
            bytes[o..o + 2].copy_from_slice(&half_bound(*x, a >= 3).to_le_bytes());
        }
    }
}

fn header(bytes: &mut [u8], flags: u32, elements: usize, bounds: &Bounds, root: u32, leaves: usize) {
    let len = bytes.len();
    put(bytes, 0, flags);
    put(bytes, 4, (len / 64 - 1) as u32);
    put(bytes, 12, elements as u32);
    for (a, x) in bounds.iter().enumerate() {
        putf(bytes, 16 + 4 * a, *x);
    }
    put(bytes, 0x28, HEADER as u32);
    put(bytes, 0x30, root);
    put(bytes, 0x34, leaves as u32);
    put(bytes, 0x38, len as u32);
    put(bytes, 0x3c, NO_TRANSFORM);
}

fn write_triangle(bytes: &mut [u8], at: usize, t: &Tri) {
    bytes[at..at + TRIANGLE].fill(0);
    for (k, v) in t.v.iter().enumerate() {
        for a in 0..3 {
            putf(bytes, at + 12 * k + 4 * a, v[a]);
        }
    }
    for a in 0..3 {
        putf(bytes, at + 36 + 4 * a, f32::NAN);
    }
    put(bytes, at + 48, t.geometry << 8 | u32::from(t.flags));
    put(bytes, at + 52, t.primitive);
    put(bytes, at + 56, NO_NODE);
    put(bytes, at + 60, TRIANGLE_ID);
}

/// The size [`build_blas`] gives `triangles` triangles at most (a box per
/// two triangles at worst is generous; the real count is about a third).
pub fn blas_size_bound(triangles: usize) -> usize {
    HEADER + triangles.div_ceil(2).max(1) * BOX32 + triangles * TRIANGLE
}

/// A bottom-level structure over `tris`, with box32 nodes.
pub fn build_blas(tris: &[Tri]) -> Vec<u8> {
    build_blas_with(tris, true)
}

/// A bottom-level structure over `tris`, with box32 nodes (`wide`) or box16
/// ones (half the size; bounds rounded outward). Triangles with a vertex
/// that is not finite are left out.
pub fn build_blas_with(tris: &[Tri], wide: bool) -> Vec<u8> {
    let bounds: Vec<Bounds> = tris.iter().map(Tri::bounds).collect();
    let ids: Vec<u32> = (0..tris.len() as u32).filter(|&i| finite(&bounds[i as usize])).collect();
    if ids.is_empty() {
        let mut bytes = vec![0u8; HEADER];
        header(&mut bytes, BLAS_FLAGS, 0, &EMPTY, NO_NODE, HEADER);
        return bytes;
    }
    let tree = Tree::build(&bounds, ids);
    let size = if wide { BOX32 } else { BOX16 };
    let box_bytes = tree.boxes.len() * size;
    let leaf_at = HEADER + box_bytes;
    let mut bytes = vec![0u8; leaf_at + tree.order.len() * TRIANGLE];
    let mut all = EMPTY;
    for (k, b) in tree.boxes.iter().enumerate() {
        write_box(&mut bytes, HEADER + k * size, b, HEADER, wide, leaf_at, TRIANGLE, 0);
    }
    for (pos, &i) in tree.order.iter().enumerate() {
        write_triangle(&mut bytes, leaf_at + pos * TRIANGLE, &tris[i as usize]);
        all = union(&all, &bounds[i as usize]);
    }
    header(&mut bytes, BLAS_FLAGS, tree.order.len(), &all, pointer(HEADER, if wide { KIND_BOX32 } else { KIND_BOX16 }), leaf_at);
    bytes
}

/// Refits the structure in `bytes` (a [`build_blas`] result, or one a
/// previous build left in memory) to the same triangles moved: `fetch`
/// gives a triangle by its geometry and primitive index. The tree is kept;
/// the leaves and every box bound are rewritten bottom-up. False when the
/// structure is not one of ours (the caller rebuilds).
pub fn refit(bytes: &mut [u8], fetch: &mut dyn FnMut(u32, u32) -> Option<Tri>) -> bool {
    let rd = |b: &[u8], at: usize| u32::from_le_bytes(b[at..at + 4].try_into().unwrap());
    if bytes.len() < HEADER || rd(bytes, 0) != BLAS_FLAGS {
        return false;
    }
    let (box_at, leaf_at, count) = (rd(bytes, 0x28) as usize, rd(bytes, 0x34) as usize, rd(bytes, 12) as usize);
    if box_at != HEADER || leaf_at < box_at || leaf_at > bytes.len() || leaf_at + count * TRIANGLE > bytes.len() {
        return false;
    }
    // The root's kind says how big the boxes are.
    let (wide, size) = match rd(bytes, 0x30) & 7 {
        KIND_BOX32 => (true, BOX32),
        KIND_BOX16 => (false, BOX16),
        _ => return false,
    };
    if (leaf_at - box_at) % size != 0 {
        return false;
    }
    let boxes = (leaf_at - box_at) / size;
    let mut leaf_bounds = Vec::with_capacity(count);
    for pos in 0..count {
        let at = leaf_at + pos * TRIANGLE;
        let (g, p) = (rd(bytes, at + 48) >> 8, rd(bytes, at + 52));
        let flags = bytes[at + 48];
        let Some(mut t) = fetch(g, p) else { return false };
        t.flags = flags;
        leaf_bounds.push(t.bounds());
        write_triangle(bytes, at, &t);
    }
    // Boxes in preorder: every box's children come after it, so bottom-up
    // is the reverse order.
    let mut box_bounds = vec![EMPTY; boxes];
    let mut all = EMPTY;
    for k in (0..boxes).rev() {
        let at = box_at + k * size;
        for c in 0..4 {
            let p = rd(bytes, at + 4 * c);
            if p == NO_NODE {
                continue;
            }
            let off = ((p >> 3) as usize) << 6;
            let b = if p & 7 >= KIND_BOX16 {
                if off < box_at || off >= leaf_at {
                    return false;
                }
                box_bounds[(off - box_at) / size]
            } else {
                if off < leaf_at || (off - leaf_at) / TRIANGLE >= count {
                    return false;
                }
                leaf_bounds[(off - leaf_at) / TRIANGLE]
            };
            let w = if finite(&b) { pad(&b) } else { [f32::NAN; 6] };
            put_bounds(bytes, at, c, wide, &w);
            if finite(&b) {
                box_bounds[k] = union(&box_bounds[k], &b);
            }
        }
    }
    if boxes > 0 {
        all = union(&all, &box_bounds[0]);
    }
    for (a, x) in all.iter().enumerate() {
        putf(bytes, 16 + 4 * a, *x);
    }
    true
}

// ---- instances -----------------------------------------------------------

/// The inverse of a 3x3 (row-major), or None when singular.
fn invert3(m: &[f32; 9]) -> Option<[f32; 9]> {
    let c = |i: usize| f64::from(m[i]);
    let (a, b, cc, d, e, f, g, h, i) = (c(0), c(1), c(2), c(3), c(4), c(5), c(6), c(7), c(8));
    let det = a * (e * i - f * h) - b * (d * i - f * g) + cc * (d * h - e * g);
    if det == 0.0 || !det.is_finite() {
        return None;
    }
    let k = 1.0 / det;
    let r = [
        (e * i - f * h) * k,
        (cc * h - b * i) * k,
        (b * f - cc * e) * k,
        (f * g - d * i) * k,
        (a * i - cc * g) * k,
        (cc * d - a * f) * k,
        (d * h - e * g) * k,
        (b * g - a * h) * k,
        (a * e - b * d) * k,
    ];
    Some(r.map(|x| x as f32))
}

fn rotation(t: &[f32; 12]) -> [f32; 9] {
    [t[0], t[1], t[2], t[4], t[5], t[6], t[8], t[9], t[10]]
}

/// The bounds of an instance in the world.
pub fn world_bounds(i: &Instance) -> Bounds {
    let mut out = EMPTY;
    for corner in 0..8 {
        let p: [f32; 3] = std::array::from_fn(|a| if corner >> a & 1 == 1 { i.bounds[3 + a] } else { i.bounds[a] });
        let w: [f32; 3] = std::array::from_fn(|r| {
            i.transform[4 * r] * p[0] + i.transform[4 * r + 1] * p[1] + i.transform[4 * r + 2] * p[2] + i.transform[4 * r + 3]
        });
        out = union(&out, &[w[0], w[1], w[2], w[0], w[1], w[2]]);
    }
    out
}

fn write_instance(bytes: &mut [u8], at: usize, i: &Instance, index: u32) -> bool {
    let Some(inv) = invert3(&rotation(&i.transform)) else { return false };
    bytes[at..at + INSTANCE].fill(0);
    for (k, x) in i.transform.iter().enumerate() {
        putf(bytes, at + 4 * k, *x);
    }
    put(bytes, at + 0x30, (i.id & 0xff_ffff) | u32::from(i.mask) << 24);
    put(bytes, at + 0x34, (i.contribution & 0xff_ffff) | u32::from(i.flags) << 24);
    // The root only; the bits above it are left clear (FiveStar sets bit 31
    // for a translation-only transform: unverified, and the matrix is right
    // either way).
    put(bytes, at + 0x38, i.root & 0x3fff_ffff);
    put(bytes, at + 0x3c, (i.blas >> 8) as u32);
    let t = [i.transform[3], i.transform[7], i.transform[11]];
    for r in 0..3 {
        let moved = -(inv[3 * r] * t[0] + inv[3 * r + 1] * t[1] + inv[3 * r + 2] * t[2]);
        putf(bytes, at + 0x40 + 4 * r, moved);
        for c in 0..3 {
            putf(bytes, at + 0x4c + 4 * (3 * r + c), inv[3 * r + c]);
        }
    }
    put(bytes, at + 0x70, index);
    true
}

/// A top-level structure over `instances`. One whose BLAS is null or has no
/// root, or whose matrix cannot be inverted, is left out.
pub fn build_tlas(instances: &[Instance]) -> Vec<u8> {
    let active: Vec<u32> = (0..instances.len() as u32)
        .filter(|&k| {
            let i = &instances[k as usize];
            i.blas != 0 && i.root != NO_NODE && finite(&i.bounds) && invert3(&rotation(&i.transform)).is_some()
        })
        .collect();
    let bounds: Vec<Bounds> = instances.iter().map(|i| if finite(&i.bounds) { world_bounds(i) } else { EMPTY }).collect();
    let active: Vec<u32> = active.into_iter().filter(|&k| finite(&bounds[k as usize])).collect();
    if active.is_empty() {
        let mut bytes = vec![0u8; HEADER];
        header(&mut bytes, TLAS_FLAGS, 0, &EMPTY, NO_NODE, HEADER);
        return bytes;
    }
    let tree = Tree::build(&bounds, active);
    let leaf_at = HEADER + tree.boxes.len() * BOX32;
    let mut bytes = vec![0u8; leaf_at + tree.order.len() * INSTANCE];
    let mut all = EMPTY;
    for (k, b) in tree.boxes.iter().enumerate() {
        write_box(&mut bytes, HEADER + k * BOX32, b, HEADER, true, leaf_at, INSTANCE, KIND_INSTANCE);
    }
    for (pos, &k) in tree.order.iter().enumerate() {
        write_instance(&mut bytes, leaf_at + pos * INSTANCE, &instances[k as usize], k);
        all = union(&all, &bounds[k as usize]);
    }
    header(&mut bytes, TLAS_FLAGS, tree.order.len(), &all, pointer(HEADER, KIND_BOX32), leaf_at);
    bytes
}

// ---- vertex formats ------------------------------------------------------

/// Whether [`decode_position`] reads this DXGI vertex format.
pub fn supported_format(format: u32) -> bool {
    matches!(format, 6 | 10 | 11 | 13 | 16 | 24 | 28 | 31 | 34 | 35 | 37 | 49 | 51)
}

/// The bytes a vertex of `format` takes.
pub fn format_size(format: u32) -> usize {
    match format {
        6 => 12,
        10 | 11 | 13 | 16 => 8,
        24 | 28 | 31 | 34 | 35 | 37 => 4,
        _ => 2,
    }
}

/// A position from a vertex's bytes (missing components are 0).
pub fn decode_position(format: u32, b: &[u8]) -> Option<[f32; 3]> {
    if b.len() < format_size(format) {
        return None;
    }
    let u16_at = |k: usize| u16::from_le_bytes([b[2 * k], b[2 * k + 1]]);
    let f32_at = |k: usize| f32::from_le_bytes(b[4 * k..4 * k + 4].try_into().unwrap());
    let snorm16 = |k: usize| (f32::from(u16_at(k) as i16) / 32767.0).max(-1.0);
    let unorm16 = |k: usize| f32::from(u16_at(k)) / 65535.0;
    let snorm8 = |k: usize| (f32::from(b[k] as i8) / 127.0).max(-1.0);
    let unorm8 = |k: usize| f32::from(b[k]) / 255.0;
    Some(match format {
        6 => [f32_at(0), f32_at(1), f32_at(2)],
        16 => [f32_at(0), f32_at(1), 0.0],
        10 => [0, 1, 2].map(|k| bvh::half_to_f32(u16_at(k))),
        34 => [bvh::half_to_f32(u16_at(0)), bvh::half_to_f32(u16_at(1)), 0.0],
        11 => [0, 1, 2].map(unorm16),
        35 => [unorm16(0), unorm16(1), 0.0],
        13 => [0, 1, 2].map(snorm16),
        37 => [snorm16(0), snorm16(1), 0.0],
        28 => [unorm8(0), unorm8(1), unorm8(2)],
        31 => [snorm8(0), snorm8(1), snorm8(2)],
        49 => [unorm8(0), unorm8(1), 0.0],
        51 => [snorm8(0), snorm8(1), 0.0],
        24 => {
            let w = u32::from_le_bytes(b[..4].try_into().unwrap());
            [(w & 1023) as f32 / 1023.0, (w >> 10 & 1023) as f32 / 1023.0, (w >> 20 & 1023) as f32 / 1023.0]
        }
        _ => return None,
    })
}

// ---- a software walker (the reference the tests and diagnostics use) -----

/// A hit: distance, geometry and primitive index (and instance index in a
/// top-level walk).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    pub t: f32,
    pub geometry: u32,
    pub primitive: u32,
    pub instance: Option<u32>,
}

fn words(bytes: &[u8], at: usize) -> [u32; 28] {
    std::array::from_fn(|k| {
        let o = at + 4 * k;
        bytes.get(o..o + 4).map_or(0, |s| u32::from_le_bytes(s.try_into().unwrap()))
    })
}

fn rd(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

/// The closest hit of `ray` in the bottom-level structure `bytes`, walked
/// the way a trace program does: a stack, the node test of
/// [`bvh::intersect`], the hit distance `numerator / denominator`.
pub fn walk_blas(bytes: &[u8], ray: &bvh::Ray) -> Option<Hit> {
    let root = rd(bytes, 0x30);
    if root == NO_NODE {
        return None;
    }
    let mut best: Option<Hit> = None;
    let mut extent = ray.extent;
    let mut stack = vec![root];
    while let Some(p) = stack.pop() {
        let at = ((p >> 3) as usize) << 6;
        let kind = p & 7;
        let mut r = *ray;
        r.extent = extent;
        let out = bvh::intersect(&words(bytes, at), kind, &r, true);
        if kind >= 4 {
            for &c in out.iter().rev() {
                if c != bvh::NONE {
                    stack.push(c);
                }
            }
        } else {
            let (num, den) = (f32::from_bits(out[0]), f32::from_bits(out[1]));
            let t = num / den;
            if t.is_finite() && t >= 0.0 && t < extent {
                extent = t;
                best = Some(Hit { t, geometry: rd(bytes, at + 48) >> 8, primitive: rd(bytes, at + 52), instance: None });
            }
        }
    }
    best
}

/// The closest hit in a top-level structure whose instances' bottom-level
/// structures `blas` gives by address; the ray goes into an instance by its
/// world-to-object fields (0x40, 0x4C), as the shadow shader reads them.
pub fn walk_tlas(bytes: &[u8], blas: &dyn Fn(u64) -> Option<Vec<u8>>, ray: &bvh::Ray) -> Option<Hit> {
    let root = rd(bytes, 0x30);
    if root == NO_NODE {
        return None;
    }
    let mut best: Option<Hit> = None;
    let mut extent = ray.extent;
    let mut stack = vec![root];
    while let Some(p) = stack.pop() {
        let at = ((p >> 3) as usize) << 6;
        let kind = p & 7;
        if kind == KIND_INSTANCE {
            let f = |o: usize| f32::from_bits(rd(bytes, at + o));
            let t = [f(0x40), f(0x44), f(0x48)];
            let m: [f32; 9] = std::array::from_fn(|k| f(0x4c + 4 * k));
            let o = ray.origin;
            let d = ray.direction;
            let origin: [f32; 3] = std::array::from_fn(|r| t[r] + m[3 * r] * o[0] + m[3 * r + 1] * o[1] + m[3 * r + 2] * o[2]);
            let dir: [f32; 3] = std::array::from_fn(|r| m[3 * r] * d[0] + m[3 * r + 1] * d[1] + m[3 * r + 2] * d[2]);
            let address = u64::from(rd(bytes, at + 0x3c)) << 8;
            let Some(b) = blas(address) else { continue };
            let mut local = bvh::Ray::new(origin, dir, extent);
            local.extent = extent;
            if let Some(h) = walk_blas_from(&b, &local, rd(bytes, at + 0x38) & 0x3fff_ffff) {
                if h.t < extent {
                    extent = h.t;
                    best = Some(Hit { instance: Some(rd(bytes, at + 0x70)), ..h });
                }
            }
            continue;
        }
        let mut r = *ray;
        r.extent = extent;
        let out = bvh::intersect(&words(bytes, at), kind, &r, true);
        for &c in out.iter().rev() {
            if c != bvh::NONE {
                stack.push(c);
            }
        }
    }
    best
}

fn walk_blas_from(bytes: &[u8], ray: &bvh::Ray, root: u32) -> Option<Hit> {
    let mut copy = bytes.to_vec();
    copy[0x30..0x34].copy_from_slice(&root.to_le_bytes());
    walk_blas(&copy, ray)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn unit(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32
        }
        fn point(&mut self, scale: f32) -> [f32; 3] {
            [self.unit() * scale, self.unit() * scale, self.unit() * scale]
        }
    }

    fn soup(rng: &mut Rng, n: usize, geometries: u32) -> Vec<Tri> {
        (0..n)
            .map(|k| {
                let c = rng.point(20.0);
                let v = std::array::from_fn(|_| {
                    let o = rng.point(1.5);
                    [c[0] + o[0], c[1] + o[1], c[2] + o[2]]
                });
                Tri { v, geometry: k as u32 % geometries, primitive: k as u32, flags: 1 }
            })
            .collect()
    }

    fn brute(tris: &[Tri], ray: &bvh::Ray) -> Option<(f32, u32)> {
        let mut best: Option<(f32, u32)> = None;
        for t in tris {
            let r = bvh::triangle(&t.v, ray);
            let h = r[0] / r[1];
            if h.is_finite() && h >= 0.0 && h < ray.extent && best.is_none_or(|b| h < b.0) {
                best = Some((h, t.primitive));
            }
        }
        best
    }

    fn rays(rng: &mut Rng, n: usize) -> Vec<bvh::Ray> {
        (0..n)
            .map(|_| {
                let o = rng.point(24.0);
                let d = [rng.unit() - 0.5, rng.unit() - 0.5, rng.unit() - 0.5];
                bvh::Ray::new([o[0] - 2.0, o[1] - 2.0, o[2] - 2.0], d, 1000.0)
            })
            .collect()
    }

    fn same(a: Option<Hit>, b: Option<(f32, u32)>) {
        match (a, b) {
            (None, None) => {}
            (Some(h), Some((t, p))) => {
                assert!((h.t - t).abs() <= 1e-4 * t.max(1.0), "t {} vs {}", h.t, t);
                assert!(h.primitive == p || (h.t - t).abs() < 1e-6, "primitive {} vs {}", h.primitive, p);
            }
            (a, b) => panic!("tree {a:?} brute {b:?}"),
        }
    }

    #[test]
    fn a_built_blas_finds_what_brute_force_finds() {
        for n in [1usize, 2, 3, 4, 5, 17, 500, 3000] {
            let mut rng = Rng(0x9e3779b97f4a7c15 ^ n as u64);
            let tris = soup(&mut rng, n, 3);
            let bytes = build_blas(&tris);
            assert_eq!(bytes.len() % 64, 0);
            assert_eq!(rd(&bytes, 4) as usize, bytes.len() / 64 - 1);
            assert!(bytes.len() <= blas_size_bound(n));
            let mut hits = 0;
            for ray in rays(&mut rng, 400) {
                let want = brute(&tris, &ray);
                hits += usize::from(want.is_some());
                same(walk_blas(&bytes, &ray), want);
            }
            if n >= 500 {
                assert!(hits > 20, "{hits} hits");
            }
        }
    }

    #[test]
    fn a_box16_blas_finds_what_brute_force_finds_and_is_smaller() {
        for n in [1usize, 4, 300, 3000] {
            let mut rng = Rng(0xabcdef ^ n as u64);
            let tris = soup(&mut rng, n, 2);
            let wide = build_blas_with(&tris, true);
            let narrow = build_blas_with(&tris, false);
            if n > 4 {
                assert!(narrow.len() < wide.len());
            }
            for ray in rays(&mut rng, 300) {
                same(walk_blas(&narrow, &ray), brute(&tris, &ray));
            }
            // And a refit keeps the node size.
            let mut bytes = narrow.clone();
            let moved: Vec<Tri> = tris.iter().map(|t| Tri { v: t.v.map(|v| [v[0] + 1.0, v[1], v[2]]), ..*t }).collect();
            assert!(refit(&mut bytes, &mut |_, p| Some(moved[p as usize])));
            assert_eq!(bytes.len(), narrow.len());
            for ray in rays(&mut rng, 100) {
                same(walk_blas(&bytes, &ray), brute(&moved, &ray));
            }
        }
    }

    #[test]
    fn half_bounds_round_outward() {
        for x in [0.0f32, 1.0, 1.0001, -1.0001, 0.3, -0.3, 1234.567, -1234.567, 1e-6, -1e-6, 65000.0, 70000.0, -70000.0, 3e-5] {
            let (lo, hi) = (bvh::half_to_f32(half_bound(x, false)), bvh::half_to_f32(half_bound(x, true)));
            assert!(lo <= x && x <= hi, "{x}: {lo} {hi}");
            assert!(x.abs() > 60000.0 || (hi - lo).abs() <= x.abs() * 2.0e-3 + 1.2e-7, "{x}: {lo} {hi}");
        }
    }

    #[test]
    fn an_empty_blas_has_no_root() {
        let bytes = build_blas(&[]);
        assert_eq!(rd(&bytes, 0x30), NO_NODE);
        assert!(walk_blas(&bytes, &bvh::Ray::new([0.0; 3], [1.0, 0.0, 0.0], 10.0)).is_none());
        let nan = Tri { v: [[f32::NAN; 3]; 3], geometry: 0, primitive: 0, flags: 0 };
        assert_eq!(rd(&build_blas(&[nan]), 0x30), NO_NODE);
    }

    #[test]
    fn axis_aligned_rays_on_a_grid_do_not_slip_through() {
        // Two triangles per cell of a flat grid: rays straight down through
        // every cell centre and along every shared edge's neighbourhood hit.
        let mut tris = Vec::new();
        for x in 0..20 {
            for y in 0..20 {
                let (x0, y0) = (x as f32, y as f32);
                let k = tris.len() as u32;
                tris.push(Tri { v: [[x0, y0, 0.0], [x0 + 1.0, y0, 0.0], [x0, y0 + 1.0, 0.0]], geometry: 0, primitive: k, flags: 0 });
                tris.push(Tri { v: [[x0 + 1.0, y0, 0.0], [x0 + 1.0, y0 + 1.0, 0.0], [x0, y0 + 1.0, 0.0]], geometry: 0, primitive: k + 1, flags: 0 });
            }
        }
        let bytes = build_blas(&tris);
        for x in 0..40 {
            for y in 0..40 {
                let o = [x as f32 * 0.5 + 0.25, y as f32 * 0.5 + 0.25, 5.0];
                let ray = bvh::Ray::new(o, [0.0, 0.0, -1.0], 100.0);
                same(walk_blas(&bytes, &ray), brute(&tris, &ray));
            }
        }
    }

    #[test]
    fn a_refit_follows_moved_triangles() {
        let mut rng = Rng(77);
        let tris = soup(&mut rng, 800, 2);
        let mut bytes = build_blas(&tris);
        let moved: Vec<Tri> = tris
            .iter()
            .map(|t| Tri { v: t.v.map(|v| [v[0] + 3.0, v[1] - 1.0, v[2] * 1.1]), ..*t })
            .collect();
        assert!(refit(&mut bytes, &mut |_, p| Some(moved[p as usize])));
        for ray in rays(&mut rng, 400) {
            same(walk_blas(&bytes, &ray), brute(&moved, &ray));
        }
    }

    fn matrix(scale: f32, turn: f32, at: [f32; 3]) -> [f32; 12] {
        let (s, c) = turn.sin_cos();
        [scale * c, -scale * s, 0.0, at[0], scale * s, scale * c, 0.0, at[1], 0.0, 0.0, scale, at[2]]
    }

    fn apply(m: &[f32; 12], p: [f32; 3]) -> [f32; 3] {
        std::array::from_fn(|r| m[4 * r] * p[0] + m[4 * r + 1] * p[1] + m[4 * r + 2] * p[2] + m[4 * r + 3])
    }

    #[test]
    fn a_built_tlas_finds_what_brute_force_finds_in_the_world() {
        let mut rng = Rng(5);
        let meshes: Vec<Vec<Tri>> = (0..3).map(|_| soup(&mut rng, 300, 1)).collect();
        let blases: Vec<Vec<u8>> = meshes.iter().map(|m| build_blas(m)).collect();
        let placements = [
            (0usize, matrix(1.0, 0.0, [0.0, 0.0, 0.0])),
            (0, matrix(0.5, 1.0, [30.0, 5.0, 0.0])),
            (1, matrix(2.0, 2.0, [-10.0, 20.0, 3.0])),
            (2, matrix(1.0, 0.3, [10.0, -30.0, 8.0])),
            (1, matrix(1.0, 0.0, [0.0, 40.0, 0.0])),
        ];
        let mut instances = Vec::new();
        for (k, (mesh, m)) in placements.iter().enumerate() {
            let b = &blases[*mesh];
            let mut bounds = [0f32; 6];
            for (a, x) in bounds.iter_mut().enumerate() {
                *x = f32::from_bits(rd(b, 16 + 4 * a));
            }
            instances.push(Instance { transform: *m, id: k as u32, mask: 0xff, contribution: 0, flags: 0, blas: 0x1000 * (*mesh as u64 + 1), root: rd(b, 0x30), bounds });
        }
        // A null BLAS and a singular matrix are left out.
        instances.push(Instance { blas: 0, ..instances[0] });
        instances.push(Instance { transform: matrix(0.0, 0.0, [0.0; 3]), ..instances[1] });
        let tlas = build_tlas(&instances);
        assert_eq!(rd(&tlas, 12), 5);
        let lookup = |a: u64| blases.get((a / 0x1000) as usize - 1).cloned();
        // The world's triangles, for the brute force.
        let mut world = Vec::new();
        for (k, (mesh, m)) in placements.iter().enumerate() {
            for t in &meshes[*mesh] {
                world.push(Tri { v: t.v.map(|v| apply(m, v)), geometry: k as u32, primitive: t.primitive + 1000 * k as u32, flags: 0 });
            }
        }
        let mut hits = 0;
        for ray in rays(&mut rng, 4000) {
            let ray = bvh::Ray::new(ray.origin.map(|x| x * 2.0 - 20.0), ray.direction, 1000.0);
            let want = brute(&world, &ray);
            let got = walk_tlas(&tlas, &lookup, &ray);
            hits += usize::from(want.is_some());
            match (got, want) {
                (None, None) => {}
                (Some(g), Some((t, _))) => assert!((g.t - t).abs() <= 2e-3 * t.max(1.0), "{} vs {}", g.t, t),
                (g, w) => panic!("tlas {g:?} brute {w:?}"),
            }
        }
        assert!(hits > 20, "{hits} hits");
    }

    #[test]
    fn vertex_formats_decode() {
        assert_eq!(decode_position(6, &[0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64]), Some([1.0, 2.0, 3.0]));
        assert_eq!(decode_position(16, &[0, 0, 128, 63, 0, 0, 0, 64]), Some([1.0, 2.0, 0.0]));
        assert_eq!(decode_position(10, &[0, 0x3c, 0, 0x40, 0, 0x42, 0, 0]), Some([1.0, 2.0, 3.0]));
        assert_eq!(decode_position(13, &[0xff, 0x7f, 0x01, 0x80, 0, 0, 0, 0]), Some([1.0, -1.0, 0.0]));
        assert_eq!(decode_position(28, &[255, 0, 51, 0]), Some([1.0, 0.0, 0.2]));
        assert_eq!(decode_position(24, &[0xff, 0x03, 0, 0]), Some([1.0, 0.0, 0.0]));
        assert!(decode_position(6, &[0; 8]).is_none());
        assert!(decode_position(99, &[0; 16]).is_none());
        assert!(!supported_format(99) && supported_format(37));
    }
}
