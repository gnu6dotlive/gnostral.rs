//! Parity tests for the public Q8_0 kernels.
//!
//! Every tolerance comes from [`gnostral_cpu_kernels::dot_error_bound`], which
//! is derived from the binary16 scale error and the Q8_0 rounding error (see the
//! `tolerance` module). Tests compare against a naive `f32` oracle.

use gnostral_cpu_kernels::{
    dequant_error_bound, dot_error_bound, dot_q8_0, gate_up_row_q8_0, gemm_q8_0, gemm_q8_0_scalar,
    quantize_row_q8_0, KernelError, MAX_SAFE_K, QK8_0,
};

/// Deterministic xorshift64 generator: no external crates, same sequence on
/// every platform.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in `[low, high)`.
    fn range(&mut self, low: f32, high: f32) -> f32 {
        let unit = ((self.next_u64() >> 40) as f32) / 16_777_216.0;
        low + (high - low) * unit
    }
}

fn random_row(rng: &mut Rng, len: usize, magnitude: f32) -> Vec<f32> {
    (0..len).map(|_| rng.range(-magnitude, magnitude)).collect()
}

fn reference_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .fold(0.0f32, |sum, (left, right)| sum + left * right)
}

/// Compare a kernel result with the oracle using the derived bound plus the
/// final `f32` rounding of the two results.
fn assert_within_bound(actual: f32, reference: f32, bound: f64, context: &str) {
    let difference = (f64::from(actual) - f64::from(reference)).abs();
    let matches = bound.is_finite()
        && difference <= bound * (1.0 + 1.0e-6) + 2.0 * 4.0e-8 * f64::from(reference.abs());
    assert!(
        matches,
        "{context}: |{actual} - {reference}| = {difference} exceeds derived bound {bound}"
    );
}

#[test]
fn dequantisation_stays_within_the_derived_bound() {
    let mut rng = Rng::new(0x0d15_ea5e_1234_0001);
    for magnitude in [1.0e-3, 1.0, 137.0, 1.0e5] {
        let values = random_row(&mut rng, QK8_0 * 4, magnitude);
        let blocks = quantize_row_q8_0(&values).unwrap();
        let restored = gnostral_cpu_kernels::dequantize_row_q8_0(&blocks);
        for block in 0..blocks.len() {
            let start = block * QK8_0;
            let source = &values[start..start + QK8_0];
            let max_abs = source
                .iter()
                .fold(0.0f64, |current, value| current.max(f64::from(value.abs())));
            let bound = dequant_error_bound(max_abs);
            for (offset, (&original, &round_trip)) in
                source.iter().zip(&restored[start..]).enumerate()
            {
                let difference = (f64::from(round_trip) - f64::from(original)).abs();
                assert!(
                    difference <= bound * (1.0 + 1.0e-6),
                    "lane {offset} in block {block}: {difference} > {bound}"
                );
            }
        }
    }
}

#[test]
fn dot_matches_f32_reference_within_derived_bound() {
    let mut rng = Rng::new(0x0d07_0000_0000_0001);
    for &k in &[QK8_0, QK8_0 * 4, QK8_0 * 32] {
        for &magnitude in &[1.0e-3f32, 1.0, 300.0, 1.0e5] {
            let left = random_row(&mut rng, k, magnitude);
            let right = random_row(&mut rng, k, magnitude);
            let left_blocks = quantize_row_q8_0(&left).unwrap();
            let right_blocks = quantize_row_q8_0(&right).unwrap();
            let actual = dot_q8_0(&left_blocks, &right_blocks).unwrap();
            let reference = reference_dot(&left, &right);
            assert_within_bound(
                actual,
                reference,
                dot_error_bound(&left, &right),
                &format!("dot k={k} magnitude={magnitude}"),
            );
        }
    }
}

#[test]
fn dot_of_identical_quantised_rows_is_non_negative() {
    let mut rng = Rng::new(0x0d07_0000_0000_0002);
    let row = random_row(&mut rng, QK8_0 * 8, 1.0);
    let blocks = quantize_row_q8_0(&row).unwrap();
    assert!(dot_q8_0(&blocks, &blocks).unwrap() >= 0.0);
}

#[test]
fn dot_rejects_mismatched_lengths() {
    let a = quantize_row_q8_0(&[1.0f32; QK8_0]).unwrap();
    let b = quantize_row_q8_0(&[1.0f32; QK8_0 * 2]).unwrap();
    assert_eq!(
        dot_q8_0(&a, &b).unwrap_err(),
        KernelError::DotLengthMismatch {
            a_blocks: 1,
            b_blocks: 2
        }
    );
}

#[test]
fn dot_rejects_k_beyond_the_i32_budget() {
    let blocks = vec![gnostral_cpu_kernels::BlockQ8_0::zero(); MAX_SAFE_K / QK8_0 + 1];
    assert_eq!(
        dot_q8_0(&blocks, &blocks).unwrap_err(),
        KernelError::AccumulationOverflowRisk {
            k: (MAX_SAFE_K / QK8_0 + 1) * QK8_0,
            max_k: MAX_SAFE_K,
        }
    );
}

fn reference_gemm(a: &[f32], rows: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    // `b` is output-major `[n, k]`; the mathematical product is `[rows, k] * [k, n]`.
    let mut out = vec![0.0f32; rows * n];
    for row in 0..rows {
        for column in 0..n {
            let mut total = 0.0f32;
            for inner in 0..k {
                total += a[row * k + inner] * b[column * k + inner];
            }
            out[row * n + column] = total;
        }
    }
    out
}

fn assert_gemm_parity(rows: usize, k: usize, n: usize, magnitude: f32, seed: u64) {
    let mut rng = Rng::new(seed);
    let a = random_row(&mut rng, rows * k, magnitude);
    let b = random_row(&mut rng, k * n, magnitude);
    let a_blocks = quantize_row_q8_0(&a).unwrap();
    let b_blocks = quantize_row_q8_0(&b).unwrap();
    let mut out = vec![0.0f32; rows * n];
    gemm_q8_0(&a_blocks, rows, k, &b_blocks, n, &mut out).unwrap();

    let mut scalar_out = vec![0.0f32; rows * n];
    gemm_q8_0_scalar(&a_blocks, rows, k, &b_blocks, n, &mut scalar_out).unwrap();
    assert_eq!(out, scalar_out, "SIMD and scalar GEMM must agree exactly");

    let reference = reference_gemm(&a, rows, k, &b, n);
    for row in 0..rows {
        for column in 0..n {
            let bound =
                dot_error_bound(&a[row * k..(row + 1) * k], &b[column * k..(column + 1) * k]);
            assert_within_bound(
                out[row * n + column],
                reference[row * n + column],
                bound,
                &format!("gemm row={row} column={column} rows={rows} k={k} n={n}"),
            );
        }
    }
}

#[test]
fn gemm_matches_f32_reference_within_derived_bound() {
    assert_gemm_parity(4, 128, 64, 1.0, 0x0d0e_0000_0000_0001);
    assert_gemm_parity(1, 64, 32, 250.0, 0x0d0e_0000_0000_0002);
    assert_gemm_parity(3, 32, 96, 1.0e4, 0x0d0e_0000_0000_0003);
}

#[test]
fn gemm_handles_zero_and_saturated_input() {
    let k = 64;
    let n = 32;
    let rows = 2;
    let a_blocks = quantize_row_q8_0(&[0.0f32; 2 * 64]).unwrap();
    let b_blocks = quantize_row_q8_0(&[0.0f32; 64 * 32]).unwrap();
    let mut out = vec![f32::NAN; rows * n];
    gemm_q8_0(&a_blocks, rows, k, &b_blocks, n, &mut out).unwrap();
    assert!(out.iter().all(|&value| value == 0.0));

    let mut saturated = vec![0.0f32; 32];
    saturated[0] = 1.0e6;
    let a_blocks = quantize_row_q8_0(&saturated).unwrap();
    let mut weights = vec![0.0f32; 32 * 32];
    weights[..32].copy_from_slice(&saturated);
    let b_blocks = quantize_row_q8_0(&weights).unwrap();
    let mut out = vec![0.0f32; 32];
    gemm_q8_0(&a_blocks, 1, 32, &b_blocks, 32, &mut out).unwrap();
    assert!(out.iter().all(|&value| value.is_finite()));
    assert!(out[0] > 9.0e11);
    assert_eq!(out[1], 0.0);
}

#[test]
fn gemm_rejects_unaligned_k_and_n() {
    let a = vec![gnostral_cpu_kernels::BlockQ8_0::zero(); 4];
    let b = vec![gnostral_cpu_kernels::BlockQ8_0::zero(); 4];
    let mut out = vec![0.0f32; 4];
    assert_eq!(
        gemm_q8_0(&a, 4, 31, &b, 32, &mut out).unwrap_err(),
        KernelError::GemmShapeNotBlockAligned {
            k: 31,
            n: 32,
            block: 32
        }
    );
    assert_eq!(
        gemm_q8_0(&a, 4, 32, &b, 33, &mut out).unwrap_err(),
        KernelError::GemmShapeNotBlockAligned {
            k: 32,
            n: 33,
            block: 32
        }
    );
}

#[test]
fn gemm_rejects_wrong_buffer_lengths() {
    let a = vec![gnostral_cpu_kernels::BlockQ8_0::zero(); 4];
    let b = vec![gnostral_cpu_kernels::BlockQ8_0::zero(); 32];
    let mut out = vec![0.0f32; 4];
    assert_eq!(
        gemm_q8_0(&a[..3], 2, 32, &b, 32, &mut out).unwrap_err(),
        KernelError::BufferSize {
            buffer: "a",
            expected: 2,
            actual: 3
        }
    );
    assert_eq!(
        gemm_q8_0(&a, 4, 32, &b, 32, &mut out[..3]).unwrap_err(),
        KernelError::BufferSize {
            buffer: "out",
            expected: 128,
            actual: 3
        }
    );
}

#[test]
fn gemm_accepts_empty_output_columns_but_not_unaligned_ones() {
    let a = vec![gnostral_cpu_kernels::BlockQ8_0::zero(); 2];
    let b: Vec<gnostral_cpu_kernels::BlockQ8_0> = Vec::new();
    let mut out: Vec<f32> = Vec::new();
    gemm_q8_0(&a, 2, 32, &b, 0, &mut out).unwrap();
    assert!(out.is_empty());
}

#[test]
fn fused_gate_up_equals_two_independent_dots() {
    let mut rng = Rng::new(0x0d0f_0000_0000_0001);
    let k = 96;
    let n = 64;
    let x = random_row(&mut rng, k, 2.0);
    let gate = random_row(&mut rng, n * k, 1.5);
    let up = random_row(&mut rng, n * k, 0.75);
    let x_blocks = quantize_row_q8_0(&x).unwrap();
    let gate_blocks = quantize_row_q8_0(&gate).unwrap();
    let up_blocks = quantize_row_q8_0(&up).unwrap();

    let mut out_gate = vec![0.0f32; n];
    let mut out_up = vec![0.0f32; n];
    gate_up_row_q8_0(
        &x_blocks,
        &gate_blocks,
        &up_blocks,
        n,
        &mut out_gate,
        &mut out_up,
    )
    .unwrap();

    let k_blocks = k / QK8_0;
    for column in 0..n {
        let gate_row = &gate_blocks[column * k_blocks..(column + 1) * k_blocks];
        let up_row = &up_blocks[column * k_blocks..(column + 1) * k_blocks];
        assert_eq!(out_gate[column], dot_q8_0(&x_blocks, gate_row).unwrap());
        assert_eq!(out_up[column], dot_q8_0(&x_blocks, up_row).unwrap());
    }
}

#[test]
fn fused_gate_up_rejects_unaligned_n_and_wrong_buffers() {
    let x = quantize_row_q8_0(&[1.0f32; QK8_0]).unwrap();
    let weight = quantize_row_q8_0(&[1.0f32; 32 * QK8_0]).unwrap();
    let mut out_gate = vec![0.0f32; 32];
    let mut out_up = vec![0.0f32; 32];
    assert_eq!(
        gate_up_row_q8_0(&x, &weight, &weight, 31, &mut out_gate, &mut out_up).unwrap_err(),
        KernelError::GemmShapeNotBlockAligned {
            k: QK8_0,
            n: 31,
            block: QK8_0
        }
    );
    assert_eq!(
        gate_up_row_q8_0(&x, &weight, &weight, 32, &mut out_gate[..31], &mut out_up).unwrap_err(),
        KernelError::BufferSize {
            buffer: "out_gate",
            expected: 32,
            actual: 31
        }
    );
}
