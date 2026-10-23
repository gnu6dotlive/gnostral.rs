//! Q-029 K-quant gates: reference parity, ISA bit-equality, thread and grouping invariance.
//!
//! "Reference" below means a line-by-line transcription of the vendored Candle
//! `k_quants.rs` loops (`BlockQ2K::to_float`, `BlockQ2K::vec_dot_unopt`,
//! `BlockQ3K::to_float`, `BlockQ3K::vec_dot_unopt`, `BlockQ8K::from_float`) at
//! candle rev 35d7ae7, which is what the engine executes today. The kernels in
//! this crate index blocks differently, so agreement is a real cross-check of
//! the block layout, not a tautology.

use gnostral_cpu_kernels::kquant::{
    block_q2k, block_q3k, blocks_from_bytes, dequantize_row_q2k, dequantize_row_q3k,
    dequantize_row_q8k, quantize_row_q8k,
};
use gnostral_cpu_kernels::{
    f16_to_f32, gemm_q8k, grouped_expert_ffn, quantize_rows_q8k, silu, BlockQ2K, BlockQ3K,
    BlockQ8K, DotKernel, ExpertGroup, FfnShape, GemmShape, Isa, KQuantBlock, KernelError, Numerics,
    Pool, QK_K,
};

/// Frozen before the first run: relative bound for Q3_K against the 8-lane reference
/// and for any kernel against an f64 dequantised reference (2^-16 of the absolute sum).
const KQUANT_REL_TOL: f64 = 1.0 / 65536.0;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn byte(&mut self) -> u8 {
        (self.next() >> 32) as u8
    }
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn bytes<const N: usize>(&mut self) -> [u8; N] {
        let mut out = [0u8; N];
        out.iter_mut().for_each(|b| *b = self.byte());
        out
    }
}

fn q2k(rng: &mut Rng) -> BlockQ2K {
    block_q2k(
        rng.bytes(),
        rng.bytes(),
        0.01 + rng.unit().abs() * 0.05,
        rng.unit().abs() * 0.02,
    )
}

fn q3k(rng: &mut Rng) -> BlockQ3K {
    block_q3k(
        rng.bytes(),
        rng.bytes(),
        rng.bytes(),
        0.002 + rng.unit().abs() * 0.01,
    )
}

fn activations(rng: &mut Rng, len: usize) -> Vec<f32> {
    (0..len).map(|_| rng.unit() * 3.0).collect()
}

fn q8k_row(x: &[f32]) -> Vec<BlockQ8K> {
    let mut out = vec![BlockQ8K::ZERO; x.len() / QK_K];
    quantize_row_q8k(x, &mut out).unwrap();
    out
}

fn extreme_q8k() -> BlockQ8K {
    let mut b = BlockQ8K::ZERO;
    b.d = 0.5;
    for (i, q) in b.qs.iter_mut().enumerate() {
        *q = if i % 3 == 0 { -128 } else { 127 };
    }
    for j in 0..16 {
        b.bsums[j] = b.qs[j * 16..(j + 1) * 16].iter().map(|&q| q as i16).sum();
    }
    b
}

mod reference {
    use super::*;

    pub fn q8k_from_float(x: &[f32]) -> BlockQ8K {
        let mut max = 0f32;
        let mut amax = 0f32;
        for &v in x {
            if amax < v.abs() {
                amax = v.abs();
                max = v;
            }
        }
        let mut y = BlockQ8K::ZERO;
        if amax == 0f32 {
            return y;
        }
        let iscale = -127f32 / max;
        for (j, q) in y.qs.iter_mut().enumerate() {
            let v = (iscale * x[j]).round();
            *q = v.clamp(-128., 127.) as i8
        }
        for j in 0..QK_K / 16 {
            let mut sum = 0i32;
            for ii in 0..16 {
                sum += y.qs[j * 16 + ii] as i32
            }
            y.bsums[j] = sum as i16
        }
        y.d = 1.0 / iscale;
        y
    }

    pub fn q2k_to_float(block: &BlockQ2K) -> Vec<f32> {
        let mut y = vec![0f32; QK_K];
        let d = f16_to_f32(block.d);
        let min = f16_to_f32(block.dmin);
        let mut is = 0;
        for (y_block, qs) in y.chunks_exact_mut(128).zip(block.qs.chunks_exact(32)) {
            let mut shift = 0;
            let mut y_block_index = 0;
            for _j in 0..4 {
                let sc = block.scales[is];
                is += 1;
                let dl = d * (sc & 0xF) as f32;
                let ml = min * (sc >> 4) as f32;
                for q in &qs[..16] {
                    y_block[y_block_index] = dl * ((q >> shift) & 3) as f32 - ml;
                    y_block_index += 1;
                }
                let sc = block.scales[is];
                is += 1;
                let dl = d * (sc & 0xF) as f32;
                let ml = min * (sc >> 4) as f32;
                for q in &qs[16..] {
                    y_block[y_block_index] = dl * ((q >> shift) & 3) as f32 - ml;
                    y_block_index += 1;
                }
                shift += 2;
            }
        }
        y
    }

    pub fn q2k_vec_dot(xs: &[BlockQ2K], ys: &[BlockQ8K]) -> f32 {
        let mut sumf = 0.0;
        for (x, y) in xs.iter().zip(ys.iter()) {
            let mut q2: &[_] = &x.qs;
            let mut q8: &[_] = &y.qs;
            let sc = &x.scales;
            let mut summs = 0;
            for (bsum, scale) in y.bsums.iter().zip(sc) {
                summs += *bsum as i32 * ((scale >> 4) as i32);
            }
            let dall = y.d * f16_to_f32(x.d);
            let dmin = y.d * f16_to_f32(x.dmin);
            let mut isum = 0;
            let mut is = 0;
            for _ in 0..(QK_K / 128) {
                let mut shift = 0;
                for _ in 0..4 {
                    let d = (sc[is] & 0xF) as i32;
                    is += 1;
                    let mut isuml = 0;
                    for l in 0..16 {
                        isuml += q8[l] as i32 * (((q2[l] >> shift) & 3) as i32);
                    }
                    isum += d * isuml;
                    let d = (sc[is] & 0xF) as i32;
                    is += 1;
                    isuml = 0;
                    for l in 16..32 {
                        isuml += q8[l] as i32 * (((q2[l] >> shift) & 3) as i32);
                    }
                    isum += d * isuml;
                    shift += 2;
                    q8 = &q8[32..];
                }
                q2 = &q2[32..];
            }
            sumf += dall * isum as f32 - dmin * summs as f32;
        }
        sumf
    }

    fn q3k_scales(scales: &[u8; 12]) -> [i8; 16] {
        const KMASK1: u32 = 0x03030303;
        const KMASK2: u32 = 0x0f0f0f0f;
        let mut aux = [0u32; 4];
        for i in 0..3 {
            aux[i] = u32::from_le_bytes(scales[4 * i..4 * i + 4].try_into().unwrap());
        }
        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
        aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
        aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
        aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
        let mut out = [0i8; 16];
        for (i, a) in aux.iter().enumerate() {
            for (j, b) in a.to_le_bytes().iter().enumerate() {
                out[4 * i + j] = *b as i8;
            }
        }
        out
    }

    pub fn q3k_to_float(block: &BlockQ3K) -> Vec<f32> {
        let mut y = vec![0f32; QK_K];
        let scales = q3k_scales(&block.scales);
        let d_all = f16_to_f32(block.d);
        let mut m = 1u8;
        let mut is = 0;
        for (y, qs) in y.chunks_exact_mut(128).zip(block.qs.chunks_exact(32)) {
            let mut shift = 0;
            for shift_scoped_y in y.chunks_exact_mut(32) {
                for (scale_index, scale_scoped_y) in shift_scoped_y.chunks_exact_mut(16).enumerate()
                {
                    let dl = d_all * (scales[is] as f32 - 32.0);
                    for (i, inner_y) in scale_scoped_y.iter_mut().enumerate() {
                        let q = ((qs[i + 16 * scale_index] >> shift) & 3) as i8
                            - if (block.hmask[i + 16 * scale_index] & m) == 0 {
                                4
                            } else {
                                0
                            };
                        *inner_y = dl * q as f32;
                    }
                    is += 1;
                }
                shift += 2;
                m <<= 1;
            }
        }
        y
    }

    pub fn q3k_vec_dot(xs: &[BlockQ3K], ys: &[BlockQ8K]) -> f32 {
        let mut sums = [0f32; 8];
        for (x, y) in xs.iter().zip(ys.iter()) {
            let w = q3k_to_float_ints(x);
            let scales = q3k_scales(&x.scales);
            let mut aux32 = [0i32; 8];
            for e in 0..QK_K {
                let scale = scales[e / 16] as i32 - 32;
                aux32[e % 8] += scale * (y.qs[e] as i16 * w[e] as i16) as i32;
            }
            let d = f16_to_f32(x.d) * y.d;
            for l in 0..8 {
                sums[l] += d * aux32[l] as f32;
            }
        }
        sums.iter().sum()
    }

    fn q3k_to_float_ints(block: &BlockQ3K) -> [i8; QK_K] {
        let mut a = [0i8; QK_K];
        let mut m = 1u8;
        for chunk in 0..2 {
            for s in 0..4 {
                for l in 0..32 {
                    let q = (block.qs[chunk * 32 + l] >> (2 * s)) & 3;
                    let h = if block.hmask[l] & m != 0 { 0 } else { 4 };
                    a[(chunk * 4 + s) * 32 + l] = q as i8 - h;
                }
                m <<= 1;
            }
        }
        a
    }
}

fn isas() -> Vec<Isa> {
    [Isa::Scalar, Isa::Avx2, Isa::Avx512]
        .into_iter()
        .filter(|isa| isa.available())
        .collect()
}

#[test]
fn block_layouts_match_ggml_sizes() {
    assert_eq!(std::mem::size_of::<BlockQ8K>(), 292);
    assert_eq!(std::mem::size_of::<BlockQ2K>(), 84);
    assert_eq!(std::mem::size_of::<BlockQ3K>(), 110);
    assert_eq!(std::mem::align_of::<BlockQ2K>(), 2);
    assert_eq!(std::mem::align_of::<BlockQ3K>(), 2);
}

#[test]
fn q8k_quantisation_matches_reference_bit_for_bit() {
    let mut rng = Rng(0x0123_4567_89ab_cdef);
    for case in 0..64 {
        let mut x = activations(&mut rng, QK_K);
        if case == 0 {
            x.fill(0.0);
        }
        if case == 1 {
            x[17] = -9.5;
        }
        let ours = q8k_row(&x);
        let theirs = reference::q8k_from_float(&x);
        assert_eq!(ours[0].qs, theirs.qs, "case {case}");
        assert_eq!(ours[0].bsums, theirs.bsums, "case {case}");
        assert_eq!(ours[0].d.to_bits(), theirs.d.to_bits(), "case {case}");
    }
}

#[test]
fn q8k_round_trip_within_half_step() {
    let mut rng = Rng(7);
    let x = activations(&mut rng, 4 * QK_K);
    let q = q8k_row(&x);
    let mut back = vec![0f32; x.len()];
    dequantize_row_q8k(&q, &mut back).unwrap();
    for (b, block) in q.iter().enumerate() {
        let step = block.d.abs();
        for i in 0..QK_K {
            let e = b * QK_K + i;
            assert!(
                (back[e] - x[e]).abs() <= 0.5 * step * (1.0 + 1e-6),
                "element {e}"
            );
        }
    }
}

#[test]
fn q8k_rejects_bad_input() {
    let mut out = vec![BlockQ8K::ZERO; 1];
    assert!(matches!(
        quantize_row_q8k(&[0.0; 100], &mut out),
        Err(KernelError::RowNotBlockAligned { .. })
    ));
    let mut x = vec![0.0f32; QK_K];
    x[5] = f32::NAN;
    assert_eq!(
        quantize_row_q8k(&x, &mut out),
        Err(KernelError::NonFiniteInput { index: 5 })
    );
}

#[test]
fn q2k_and_q3k_dequantisation_byte_equal_to_reference() {
    let mut rng = Rng(0xdead_beef);
    for _ in 0..200 {
        let a = q2k(&mut rng);
        let mut ours = vec![0f32; QK_K];
        dequantize_row_q2k(&[a], &mut ours).unwrap();
        let theirs = reference::q2k_to_float(&a);
        assert!(ours
            .iter()
            .zip(&theirs)
            .all(|(x, y)| x.to_bits() == y.to_bits()));

        let b = q3k(&mut rng);
        dequantize_row_q3k(&[b], &mut ours).unwrap();
        let theirs = reference::q3k_to_float(&b);
        assert!(ours
            .iter()
            .zip(&theirs)
            .all(|(x, y)| x.to_bits() == y.to_bits()));
    }
}

#[test]
fn q2k_dot_bit_equal_to_reference_on_every_isa() {
    let mut rng = Rng(11);
    for blocks in [1usize, 2, 8] {
        for _ in 0..50 {
            let w: Vec<BlockQ2K> = (0..blocks).map(|_| q2k(&mut rng)).collect();
            let a = q8k_row(&activations(&mut rng, blocks * QK_K));
            let expected = reference::q2k_vec_dot(&w, &a);
            for isa in isas() {
                assert_eq!(
                    BlockQ2K::dot(isa, &w, &a).to_bits(),
                    expected.to_bits(),
                    "{isa:?}"
                );
            }
        }
    }
}

#[test]
fn isa_paths_bit_equal_on_extreme_blocks() {
    let mut rng = Rng(99);
    let a = vec![extreme_q8k(); 2];
    let mut w2 = vec![q2k(&mut rng), q2k(&mut rng)];
    w2[0].scales = [0xFF; 16];
    w2[0].qs = [0xFF; 64];
    let mut w3 = vec![q3k(&mut rng), q3k(&mut rng)];
    w3[0].scales = [0xFF; 12];
    w3[0].qs = [0xFF; 64];
    w3[1].hmask = [0; 32];
    w3[1].scales = [0; 12];
    let s2 = BlockQ2K::dot(Isa::Scalar, &w2, &a);
    let s3 = BlockQ3K::dot(Isa::Scalar, &w3, &a);
    for isa in isas() {
        assert_eq!(
            BlockQ2K::dot(isa, &w2, &a).to_bits(),
            s2.to_bits(),
            "{isa:?}"
        );
        assert_eq!(
            BlockQ3K::dot(isa, &w3, &a).to_bits(),
            s3.to_bits(),
            "{isa:?}"
        );
    }
}

#[test]
fn q3k_dot_isa_bit_equal_and_within_frozen_tolerance_of_reference() {
    let mut rng = Rng(12);
    for blocks in [1usize, 3, 8] {
        for _ in 0..50 {
            let w: Vec<BlockQ3K> = (0..blocks).map(|_| q3k(&mut rng)).collect();
            let a = q8k_row(&activations(&mut rng, blocks * QK_K));
            let scalar = BlockQ3K::dot(Isa::Scalar, &w, &a);
            for isa in isas() {
                assert_eq!(
                    BlockQ3K::dot(isa, &w, &a).to_bits(),
                    scalar.to_bits(),
                    "{isa:?}"
                );
            }
            let reference = reference::q3k_vec_dot(&w, &a) as f64;
            let scale = abs_dot::<BlockQ3K>(&w, &a);
            assert!(
                (scalar as f64 - reference).abs() <= KQUANT_REL_TOL * scale + 1e-30,
                "{scalar} vs {reference}"
            );
        }
    }
}

fn abs_dot<W: KQuantBlock>(w: &[W], a: &[BlockQ8K]) -> f64 {
    let mut wf = vec![0f32; w.len() * QK_K];
    W::dequantize(w, &mut wf).unwrap();
    let mut af = vec![0f32; a.len() * QK_K];
    dequantize_row_q8k(a, &mut af).unwrap();
    wf.iter()
        .zip(&af)
        .map(|(&x, &y)| (x as f64 * y as f64).abs())
        .sum()
}

fn f64_dot<W: KQuantBlock>(w: &[W], a: &[BlockQ8K]) -> f64 {
    let mut wf = vec![0f32; w.len() * QK_K];
    W::dequantize(w, &mut wf).unwrap();
    let mut af = vec![0f32; a.len() * QK_K];
    dequantize_row_q8k(a, &mut af).unwrap();
    wf.iter().zip(&af).map(|(&x, &y)| x as f64 * y as f64).sum()
}

#[test]
fn dots_agree_with_f64_dequantised_reference() {
    let mut rng = Rng(13);
    for _ in 0..100 {
        let w2: Vec<BlockQ2K> = (0..4).map(|_| q2k(&mut rng)).collect();
        let w3: Vec<BlockQ3K> = (0..4).map(|_| q3k(&mut rng)).collect();
        let a = q8k_row(&activations(&mut rng, 4 * QK_K));
        for (got, want, scale) in [
            (
                BlockQ2K::dot(Isa::detect(), &w2, &a) as f64,
                f64_dot(&w2, &a),
                abs_dot(&w2, &a),
            ),
            (
                BlockQ3K::dot(Isa::detect(), &w3, &a) as f64,
                f64_dot(&w3, &a),
                abs_dot(&w3, &a),
            ),
        ] {
            assert!(
                (got - want).abs() <= KQUANT_REL_TOL * scale + 1e-30,
                "{got} vs {want}"
            );
        }
    }
}

#[test]
fn zero_copy_reinterpretation_round_trips() {
    let mut rng = Rng(14);
    let blocks: Vec<BlockQ3K> = (0..3).map(|_| q3k(&mut rng)).collect();
    // SAFETY: test-only view of plain-old-data blocks as bytes.
    let bytes = unsafe {
        std::slice::from_raw_parts(
            blocks.as_ptr() as *const u8,
            3 * std::mem::size_of::<BlockQ3K>(),
        )
    };
    assert_eq!(blocks_from_bytes::<BlockQ3K>(bytes).unwrap(), &blocks[..]);
    assert!(blocks_from_bytes::<BlockQ3K>(&bytes[..bytes.len() - 1]).is_none());
    assert!(blocks_from_bytes::<BlockQ3K>(&bytes[1..]).is_none());
}

#[test]
fn gemm_matches_per_element_dot_and_is_thread_invariant() {
    let mut rng = Rng(15);
    let (m, k, n) = (5, 2 * QK_K, 37);
    let w: Vec<BlockQ2K> = (0..n * k / QK_K).map(|_| q2k(&mut rng)).collect();
    let act = quantize_rows_q8k(&activations(&mut rng, m * k), m, k).unwrap();
    let kb = k / QK_K;
    let mut expected = vec![0f32; m * n];
    for r in 0..m {
        for c in 0..n {
            expected[r * n + c] = BlockQ2K::dot(
                Isa::Scalar,
                &w[c * kb..(c + 1) * kb],
                &act[r * kb..(r + 1) * kb],
            );
        }
    }
    for threads in [1usize, 2, 4, 8] {
        let pool = Pool::new(threads);
        for isa in isas() {
            let mut out = vec![0f32; m * n];
            gemm_q8k(isa, &pool, GemmShape { m, k, n }, &act, &w, &mut out).unwrap();
            assert!(
                out.iter()
                    .zip(&expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "threads {threads} {isa:?}"
            );
        }
    }
}

#[test]
fn gemm_rejects_bad_shapes() {
    let pool = Pool::new(1);
    let act = vec![BlockQ8K::ZERO; 2];
    let w: Vec<BlockQ2K> = vec![block_q2k([0; 16], [0; 64], 0.0, 0.0); 3];
    let mut out = vec![0f32; 6];
    assert!(gemm_q8k(
        Isa::Scalar,
        &pool,
        GemmShape { m: 2, k: 100, n: 3 },
        &act,
        &w,
        &mut out
    )
    .is_err());
    assert!(gemm_q8k(
        Isa::Scalar,
        &pool,
        GemmShape {
            m: 2,
            k: QK_K,
            n: 2
        },
        &act,
        &w,
        &mut out
    )
    .is_err());
    assert!(gemm_q8k(
        Isa::Scalar,
        &pool,
        GemmShape {
            m: 2,
            k: QK_K,
            n: 3
        },
        &act,
        &w,
        &mut out[..5]
    )
    .is_err());
}

struct Expert {
    gate: Vec<BlockQ2K>,
    up: Vec<BlockQ2K>,
    down: Vec<BlockQ3K>,
}

fn expert(rng: &mut Rng, shape: FfnShape) -> Expert {
    let hb = shape.hidden / QK_K;
    let ib = shape.inter / QK_K;
    Expert {
        gate: (0..shape.inter * hb).map(|_| q2k(rng)).collect(),
        up: (0..shape.inter * hb).map(|_| q2k(rng)).collect(),
        down: (0..shape.hidden * ib).map(|_| q3k(rng)).collect(),
    }
}

fn group<'a>(e: &'a Expert, x: &'a [f32], rows: usize) -> ExpertGroup<'a, BlockQ2K, BlockQ3K> {
    ExpertGroup {
        gate: &e.gate,
        up: &e.up,
        down: &e.down,
        x,
        rows,
    }
}

#[test]
fn grouped_ffn_is_bit_equal_to_per_route_and_thread_invariant() {
    let mut rng = Rng(16);
    let shape = FfnShape {
        hidden: 2 * QK_K,
        inter: QK_K,
    };
    let experts: Vec<Expert> = (0..3).map(|_| expert(&mut rng, shape)).collect();
    let rows = [3usize, 1, 4];
    let inputs: Vec<Vec<f32>> = rows
        .iter()
        .map(|&r| activations(&mut rng, r * shape.hidden))
        .collect();

    let single = Pool::new(1);
    let mut per_route: Vec<Vec<f32>> = Vec::new();
    for (e, (x, &r)) in experts.iter().zip(inputs.iter().zip(&rows)) {
        let mut out = Vec::new();
        for row in 0..r {
            let xr = &x[row * shape.hidden..(row + 1) * shape.hidden];
            let y =
                grouped_expert_ffn(Isa::Scalar, &single, shape, &[group(e, xr, 1)], silu).unwrap();
            out.extend_from_slice(&y[0]);
        }
        per_route.push(out);
    }

    for threads in [1usize, 3, 8] {
        let pool = Pool::new(threads);
        for isa in isas() {
            let groups: Vec<_> = experts
                .iter()
                .zip(inputs.iter().zip(&rows))
                .map(|(e, (x, &r))| group(e, x, r))
                .collect();
            let got = grouped_expert_ffn(isa, &pool, shape, &groups, silu).unwrap();
            for (g, want) in got.iter().zip(&per_route) {
                assert!(
                    g.iter().zip(want).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "threads {threads} {isa:?}"
                );
            }
        }
    }
}

#[test]
fn grouped_ffn_tracks_f64_dequantised_mlp() {
    let mut rng = Rng(17);
    let shape = FfnShape {
        hidden: QK_K,
        inter: QK_K,
    };
    let e = expert(&mut rng, shape);
    let x = activations(&mut rng, 2 * shape.hidden);
    let got = grouped_expert_ffn(
        Isa::detect(),
        Pool::global(),
        shape,
        &[group(&e, &x, 2)],
        silu,
    )
    .unwrap()
    .remove(0);

    let deq = |w: &dyn Fn(&mut [f32]), len: usize| {
        let mut v = vec![0f32; len];
        w(&mut v);
        v
    };
    let wg = deq(
        &|v| BlockQ2K::dequantize(&e.gate, v).unwrap(),
        shape.inter * shape.hidden,
    );
    let wu = deq(
        &|v| BlockQ2K::dequantize(&e.up, v).unwrap(),
        shape.inter * shape.hidden,
    );
    let wd = deq(
        &|v| BlockQ3K::dequantize(&e.down, v).unwrap(),
        shape.hidden * shape.inter,
    );
    for row in 0..2 {
        let xr = &x[row * shape.hidden..(row + 1) * shape.hidden];
        let h: Vec<f64> = (0..shape.inter)
            .map(|c| {
                let g: f64 = (0..shape.hidden)
                    .map(|i| wg[c * shape.hidden + i] as f64 * xr[i] as f64)
                    .sum();
                let u: f64 = (0..shape.hidden)
                    .map(|i| wu[c * shape.hidden + i] as f64 * xr[i] as f64)
                    .sum();
                g / (1.0 + (-g).exp()) * u
            })
            .collect();
        let norm: f64 = (0..shape.hidden)
            .map(|c| {
                (0..shape.inter)
                    .map(|i| (wd[c * shape.inter + i] as f64 * h[i]).abs())
                    .sum::<f64>()
            })
            .fold(0.0, f64::max);
        for c in 0..shape.hidden {
            let want: f64 = (0..shape.inter)
                .map(|i| wd[c * shape.inter + i] as f64 * h[i])
                .sum();
            // Two Q8_K activation quantisations dominate: a loose, documented 2 % of the row scale.
            assert!(
                (got[row * shape.hidden + c] as f64 - want).abs() <= 0.02 * norm + 1e-6,
                "row {row} col {c}"
            );
        }
    }
}

#[test]
fn unavailable_isa_is_a_typed_error() {
    let pool = Pool::new(1);
    for isa in [Isa::Avx2, Isa::Avx512] {
        if !isa.available() {
            let mut out = vec![0f32; 1];
            let act = vec![BlockQ8K::ZERO; 1];
            let w = vec![block_q2k([0; 16], [0; 64], 0.0, 0.0); 1];
            assert_eq!(
                gemm_q8k(
                    isa,
                    &pool,
                    GemmShape {
                        m: 1,
                        k: QK_K,
                        n: 1
                    },
                    &act,
                    &w,
                    &mut out
                ),
                Err(KernelError::IsaUnavailable { isa: isa.name() })
            );
        }
    }
}

/// Line-by-line transcription of the vendored Candle `avx.rs` `vec_dot_q2k_q8k` and
/// `vec_dot_q3k_q8k` (candle rev 35d7ae7), the kernels Candle runs when AVX2 and FMA are present.
#[cfg(target_arch = "x86_64")]
mod candle_avx {
    use super::*;
    use core::arch::x86_64::*;

    const K_SHUFFLE: [u8; 128] = [
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
        2, 3, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
        6, 7, 6, 7, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 10, 11, 10, 11, 10, 11, 10, 11,
        10, 11, 10, 11, 10, 11, 10, 11, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12,
        13, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
    ];

    #[target_feature(enable = "avx2,fma")]
    unsafe fn get_scale_shuffle_q3k(i: usize) -> __m256i {
        unsafe { _mm256_loadu_si256((K_SHUFFLE.as_ptr() as *const __m256i).add(i)) }
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn mm256_set_m128i(a: __m128i, b: __m128i) -> __m256i {
        _mm256_insertf128_si256(_mm256_castsi128_si256(b), a, 1)
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn hsum_float_8(x: __m256) -> f32 {
        let res = _mm256_extractf128_ps(x, 1);
        let res = _mm_add_ps(res, _mm256_castps256_ps128(x));
        let res = _mm_add_ps(res, _mm_movehl_ps(res, res));
        let res = _mm_add_ss(res, _mm_movehdup_ps(res));
        _mm_cvtss_f32(res)
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn vec_dot_q2k_q8k(xs: &[BlockQ2K], ys: &[BlockQ8K]) -> f32 {
        unsafe {
            let m3 = _mm256_set1_epi8(3);
            let m4 = _mm_set1_epi8(0xF);
            let mut acc = _mm256_setzero_ps();
            for (x, y) in xs.iter().zip(ys.iter()) {
                let d = y.d * f16_to_f32(x.d);
                let dmin = -y.d * f16_to_f32(x.dmin);
                let mut q2 = x.qs.as_ptr();
                let mut q8 = y.qs.as_ptr();
                let mins_and_scales = _mm_loadu_si128(x.scales.as_ptr() as *const __m128i);
                let scales8 = _mm_and_si128(mins_and_scales, m4);
                let mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
                let mins = _mm256_cvtepi8_epi16(mins8);
                let prod =
                    _mm256_madd_epi16(mins, _mm256_loadu_si256(y.bsums.as_ptr() as *const __m256i));
                acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&dmin), _mm256_cvtepi32_ps(prod), acc);
                let all_scales = _mm256_cvtepi8_epi16(scales8);
                let l_scales = _mm256_extracti128_si256(all_scales, 0);
                let h_scales = _mm256_extracti128_si256(all_scales, 1);
                let scales = [
                    mm256_set_m128i(l_scales, l_scales),
                    mm256_set_m128i(h_scales, h_scales),
                ];
                let mut sumi = _mm256_setzero_si256();
                for scale in scales {
                    let q2bits = _mm256_loadu_si256(q2 as *const __m256i);
                    q2 = q2.add(32);
                    let q8_0 = _mm256_loadu_si256(q8 as *const __m256i);
                    q8 = q8.add(32);
                    let q8_1 = _mm256_loadu_si256(q8 as *const __m256i);
                    q8 = q8.add(32);
                    let q8_2 = _mm256_loadu_si256(q8 as *const __m256i);
                    q8 = q8.add(32);
                    let q8_3 = _mm256_loadu_si256(q8 as *const __m256i);
                    q8 = q8.add(32);
                    let q2_0 = _mm256_and_si256(q2bits, m3);
                    let q2_1 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), m3);
                    let q2_2 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), m3);
                    let q2_3 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), m3);
                    let p0 = _mm256_maddubs_epi16(q2_0, q8_0);
                    let p1 = _mm256_maddubs_epi16(q2_1, q8_1);
                    let p2 = _mm256_maddubs_epi16(q2_2, q8_2);
                    let p3 = _mm256_maddubs_epi16(q2_3, q8_3);
                    let p0 =
                        _mm256_madd_epi16(_mm256_shuffle_epi8(scale, get_scale_shuffle_q3k(0)), p0);
                    let p1 =
                        _mm256_madd_epi16(_mm256_shuffle_epi8(scale, get_scale_shuffle_q3k(1)), p1);
                    let p2 =
                        _mm256_madd_epi16(_mm256_shuffle_epi8(scale, get_scale_shuffle_q3k(2)), p2);
                    let p3 =
                        _mm256_madd_epi16(_mm256_shuffle_epi8(scale, get_scale_shuffle_q3k(3)), p3);
                    let p0 = _mm256_add_epi32(p0, p1);
                    let p2 = _mm256_add_epi32(p2, p3);
                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p0, p2));
                }
                acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
            }
            hsum_float_8(acc)
        }
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn q3h(hbits: __m256i, mone: __m256i, bit: i32) -> __m256i {
        let count = _mm_cvtsi32_si128(bit);
        let masked = _mm256_andnot_si256(hbits, _mm256_sll_epi16(mone, count));
        _mm256_slli_epi16(_mm256_srl_epi16(masked, count), 2)
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn vec_dot_q3k_q8k(xs: &[BlockQ3K], ys: &[BlockQ8K]) -> f32 {
        const KMASK1: u32 = 0x03030303;
        const KMASK2: u32 = 0x0f0f0f0f;
        let mut aux = [0u32; 3];
        unsafe {
            let m3 = _mm256_set1_epi8(3);
            let mone = _mm256_set1_epi8(1);
            let m32 = _mm_set1_epi8(32);
            let mut acc = _mm256_setzero_ps();
            for (x, y) in xs.iter().zip(ys.iter()) {
                let d = y.d * f16_to_f32(x.d);
                let mut q3 = x.qs.as_ptr();
                let mut q8 = y.qs.as_ptr();
                for (i, a) in aux.iter_mut().enumerate() {
                    *a = u32::from_le_bytes(x.scales[4 * i..4 * i + 4].try_into().unwrap());
                }
                let scales128 = _mm_set_epi32(
                    (((aux[1] >> 4) & KMASK2) | (((aux[2] >> 6) & KMASK1) << 4)) as i32,
                    (((aux[0] >> 4) & KMASK2) | (((aux[2] >> 4) & KMASK1) << 4)) as i32,
                    ((aux[1] & KMASK2) | (((aux[2] >> 2) & KMASK1) << 4)) as i32,
                    ((aux[0] & KMASK2) | (((aux[2]) & KMASK1) << 4)) as i32,
                );
                let scales128 = _mm_sub_epi8(scales128, m32);
                let all_scales = _mm256_cvtepi8_epi16(scales128);
                let l_scales = _mm256_extracti128_si256(all_scales, 0);
                let h_scales = _mm256_extracti128_si256(all_scales, 1);
                let scales = [
                    mm256_set_m128i(l_scales, l_scales),
                    mm256_set_m128i(h_scales, h_scales),
                ];
                let hbits = _mm256_loadu_si256(x.hmask.as_ptr() as *const __m256i);
                let mut sumi = _mm256_setzero_si256();
                for (j, scale) in scales.iter().enumerate() {
                    let q3bits = _mm256_loadu_si256(q3 as *const __m256i);
                    q3 = q3.add(32);
                    let base = if j == 0 { 0 } else { 4 };
                    let q3l_0 = _mm256_and_si256(q3bits, m3);
                    let q3h_0 = q3h(hbits, mone, base);
                    let q3l_1 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 2), m3);
                    let q3h_1 = q3h(hbits, mone, base + 1);
                    let q3l_2 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 4), m3);
                    let q3h_2 = q3h(hbits, mone, base + 2);
                    let q3l_3 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 6), m3);
                    let q3h_3 = q3h(hbits, mone, base + 3);
                    let q8_0 = _mm256_loadu_si256(q8 as *const __m256i);
                    q8 = q8.add(32);
                    let q8_1 = _mm256_loadu_si256(q8 as *const __m256i);
                    q8 = q8.add(32);
                    let q8_2 = _mm256_loadu_si256(q8 as *const __m256i);
                    q8 = q8.add(32);
                    let q8_3 = _mm256_loadu_si256(q8 as *const __m256i);
                    q8 = q8.add(32);
                    let q8s_0 = _mm256_maddubs_epi16(q3h_0, q8_0);
                    let q8s_1 = _mm256_maddubs_epi16(q3h_1, q8_1);
                    let q8s_2 = _mm256_maddubs_epi16(q3h_2, q8_2);
                    let q8s_3 = _mm256_maddubs_epi16(q3h_3, q8_3);
                    let p16_0 = _mm256_sub_epi16(_mm256_maddubs_epi16(q3l_0, q8_0), q8s_0);
                    let p16_1 = _mm256_sub_epi16(_mm256_maddubs_epi16(q3l_1, q8_1), q8s_1);
                    let p16_2 = _mm256_sub_epi16(_mm256_maddubs_epi16(q3l_2, q8_2), q8s_2);
                    let p16_3 = _mm256_sub_epi16(_mm256_maddubs_epi16(q3l_3, q8_3), q8s_3);
                    let p16_0 = _mm256_madd_epi16(
                        _mm256_shuffle_epi8(*scale, get_scale_shuffle_q3k(0)),
                        p16_0,
                    );
                    let p16_1 = _mm256_madd_epi16(
                        _mm256_shuffle_epi8(*scale, get_scale_shuffle_q3k(1)),
                        p16_1,
                    );
                    let p16_2 = _mm256_madd_epi16(
                        _mm256_shuffle_epi8(*scale, get_scale_shuffle_q3k(2)),
                        p16_2,
                    );
                    let p16_3 = _mm256_madd_epi16(
                        _mm256_shuffle_epi8(*scale, get_scale_shuffle_q3k(3)),
                        p16_3,
                    );
                    let p16_0 = _mm256_add_epi32(p16_0, p16_1);
                    let p16_2 = _mm256_add_epi32(p16_2, p16_3);
                    sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_2));
                }
                acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
            }
            hsum_float_8(acc)
        }
    }
}

/// `Numerics::CandleCompat` must equal what the vendored Candle `vec_dot` returns on this host.
fn candle_expected_q2k(w: &[BlockQ2K], a: &[BlockQ8K]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if gnostral_cpu_kernels::candle_uses_avx2_fma() {
        // SAFETY: AVX2 and FMA detected.
        return unsafe { candle_avx::vec_dot_q2k_q8k(w, a) };
    }
    reference::q2k_vec_dot(w, a)
}

fn candle_expected_q3k(w: &[BlockQ3K], a: &[BlockQ8K]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if gnostral_cpu_kernels::candle_uses_avx2_fma() {
        // SAFETY: AVX2 and FMA detected.
        return unsafe { candle_avx::vec_dot_q3k_q8k(w, a) };
    }
    reference::q3k_vec_dot(w, a)
}

#[test]
fn candle_compat_fallback_without_fma_is_bit_equal_to_vec_dot_unopt() {
    let mut rng = Rng(0xab);
    for blocks in [1usize, 3, 8] {
        for _ in 0..50 {
            let w: Vec<BlockQ3K> = (0..blocks).map(|_| q3k(&mut rng)).collect();
            let a = q8k_row(&activations(&mut rng, blocks * QK_K));
            assert_eq!(
                gnostral_cpu_kernels::kquant::q3k_dot_candle_unopt(&w, &a).to_bits(),
                reference::q3k_vec_dot(&w, &a).to_bits()
            );
        }
    }
}

fn compat(isa: Isa) -> DotKernel {
    DotKernel {
        isa,
        numerics: Numerics::CandleCompat,
    }
}

#[test]
fn candle_compat_dot_bit_equal_to_candle_on_every_isa() {
    let mut rng = Rng(0x5eed_cafe);
    let mut extreme = vec![extreme_q8k(); 2];
    extreme[1].d = -0.25;
    for blocks in [1usize, 2, 5, 8] {
        for case in 0..60 {
            let a = if case == 0 && blocks == 2 {
                extreme.clone()
            } else {
                q8k_row(&activations(&mut rng, blocks * QK_K))
            };
            let w2: Vec<BlockQ2K> = (0..blocks).map(|_| q2k(&mut rng)).collect();
            let mut w3: Vec<BlockQ3K> = (0..blocks).map(|_| q3k(&mut rng)).collect();
            if case == 1 {
                w3[0].hmask = [0; 32];
                w3[0].scales = [0xFF; 12];
            }
            let e2 = candle_expected_q2k(&w2, &a);
            let e3 = candle_expected_q3k(&w3, &a);
            for isa in isas() {
                assert_eq!(
                    BlockQ2K::dot_with(compat(isa), &w2, &a).to_bits(),
                    e2.to_bits(),
                    "q2k {isa:?} blocks {blocks} case {case}"
                );
                assert_eq!(
                    BlockQ3K::dot_with(compat(isa), &w3, &a).to_bits(),
                    e3.to_bits(),
                    "q3k {isa:?} blocks {blocks} case {case}"
                );
            }
        }
    }
}

#[test]
fn candle_compat_grouped_ffn_bit_equal_to_candle_composition() {
    let mut rng = Rng(0xf00d);
    let (hidden, inter) = (512, 256);
    let (hb, ib) = (hidden / QK_K, inter / QK_K);
    let gate: Vec<BlockQ2K> = (0..inter * hb).map(|_| q2k(&mut rng)).collect();
    let up: Vec<BlockQ2K> = (0..inter * hb).map(|_| q2k(&mut rng)).collect();
    let down: Vec<BlockQ3K> = (0..hidden * ib).map(|_| q3k(&mut rng)).collect();
    let rows = 3;
    let x = activations(&mut rng, rows * hidden);
    let group = ExpertGroup {
        gate: &gate,
        up: &up,
        down: &down,
        x: &x,
        rows,
    };
    for isa in isas() {
        let got = grouped_expert_ffn(
            compat(isa),
            Pool::global(),
            FfnShape { hidden, inter },
            &[group_ref(&group)],
            silu,
        )
        .unwrap();
        for r in 0..rows {
            let xq = q8k_row(&x[r * hidden..(r + 1) * hidden]);
            let h: Vec<f32> = (0..inter)
                .map(|c| {
                    let g = candle_expected_q2k(&gate[c * hb..(c + 1) * hb], &xq);
                    let u = candle_expected_q2k(&up[c * hb..(c + 1) * hb], &xq);
                    silu(g) * u
                })
                .collect();
            let hq = q8k_row(&h);
            for c in 0..hidden {
                let want = candle_expected_q3k(&down[c * ib..(c + 1) * ib], &hq);
                assert_eq!(
                    got[0][r * hidden + c].to_bits(),
                    want.to_bits(),
                    "{isa:?} row {r} col {c}"
                );
            }
        }
    }
}

fn group_ref<'a>(g: &ExpertGroup<'a, BlockQ2K, BlockQ3K>) -> ExpertGroup<'a, BlockQ2K, BlockQ3K> {
    ExpertGroup {
        gate: g.gate,
        up: g.up,
        down: g.down,
        x: g.x,
        rows: g.rows,
    }
}
