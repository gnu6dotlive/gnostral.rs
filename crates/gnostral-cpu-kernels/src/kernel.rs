//! Dot product, blocked GEMM and the fused gate/up scheduling helper.
//!
//! All kernels consume [`BlockQ8_0`] slices, so block alignment is guaranteed by
//! the type of the input. Shape constraints that the type cannot express are
//! checked here and returned as [`KernelError`].

use crate::block::{BlockQ8_0, QK8_0};
use crate::KernelError;

/// Largest `K` for which the `i32` accumulation in these kernels cannot
/// overflow.
///
/// The worst-case product of two lanes is `-128 * -128 = 16384`; 32 lanes times
/// `MAX_SAFE_K` values must stay below `i32::MAX`. This budget is accepted, not
/// assumed: the quantiser emits `[-127, 127]`, but dequantisation and the
/// kernels tolerate a foreign `-128`.
pub const MAX_SAFE_K: usize = (i32::MAX as usize) / (128 * 128);

/// True when this process has an AVX2 path available.
///
/// It is true when the crate itself was compiled with `-C target-feature=+avx2`
/// or when the CPU reports AVX2 at runtime. Every caller must still treat the
/// scalar path as the reference.
pub fn avx2_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        build_time_avx2() || std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        build_time_avx2()
    }
}

#[cfg(target_feature = "avx2")]
const fn build_time_avx2() -> bool {
    true
}

#[cfg(not(target_feature = "avx2"))]
const fn build_time_avx2() -> bool {
    false
}

fn check_i32_budget(k: usize) -> Result<(), KernelError> {
    if k > MAX_SAFE_K {
        Err(KernelError::AccumulationOverflowRisk {
            k,
            max_k: MAX_SAFE_K,
        })
    } else {
        Ok(())
    }
}

/// Scalar reference implementation of the integer block dot product.
mod scalar {
    /// Sum of the 32 lane products. The mod-2^32 wrap cannot occur for the
    /// lengths accepted by this crate, so the result is a plain `i32`.
    pub(crate) fn dot_i32(a: &[i8; 32], b: &[i8; 32]) -> i32 {
        let mut sum = 0i32;
        for (&x, &y) in a.iter().zip(b.iter()) {
            sum += i32::from(x) * i32::from(y);
        }
        sum
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use core::arch::x86_64::{
        __m256i, _mm256_abs_epi8, _mm256_castsi256_si128, _mm256_extracti128_si256,
        _mm256_hadd_epi32, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_maddubs_epi16,
        _mm256_set1_epi16, _mm256_sign_epi8, _mm_add_epi32, _mm_cvtsi128_si32,
    };

    /// AVX2 integer block dot product.
    ///
    /// # Safety
    ///
    /// The caller must ensure AVX2 is available (runtime detection or
    /// `-C target-feature=+avx2`), and both pointers must be valid for reading
    /// 32 bytes. Neither block may contain `-128`: `_mm256_abs_epi8` saturates
    /// that value and the identity used below would no longer hold. The public
    /// dispatcher checks both conditions and falls back to the scalar path.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn dot_i32(a: *const i8, b: *const i8) -> i32 {
        // SAFETY: guaranteed by the function contract above.
        unsafe {
            let left = _mm256_loadu_si256(a as *const __m256i);
            let right = _mm256_loadu_si256(b as *const __m256i);
            // |a| * sign(a) * b == a * b, with |a| <= 127 so the signed 16-bit
            // pairwise sum (at most 127 * 127 * 2 = 32258) cannot saturate.
            let signed_right = _mm256_sign_epi8(right, left);
            let absolute_left = _mm256_abs_epi8(left);
            let pairs = _mm256_maddubs_epi16(absolute_left, signed_right);
            let quads = _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
            let first = _mm256_hadd_epi32(quads, quads);
            let second = _mm256_hadd_epi32(first, first);
            let low = _mm256_castsi256_si128(second);
            let high = _mm256_extracti128_si256(second, 1);
            _mm_cvtsi128_si32(_mm_add_epi32(low, high))
        }
    }
}

/// Integer dot product of one pair of Q8_0 blocks.
///
/// Dispatches to AVX2 when available and the blocks do not contain `-128`;
/// otherwise the scalar path runs. This is the unit the dot product and GEMM
/// share.
#[inline]
pub fn dot_block_q8_0_i32(a: &BlockQ8_0, b: &BlockQ8_0) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if avx2_available() && !a.qs.contains(&-128) && !b.qs.contains(&-128) {
            // SAFETY: AVX2 was detected, and both blocks are `[i8; 32]` so each
            // pointer is valid for 32 bytes. `-128` was excluded above.
            return unsafe { avx2::dot_i32(a.qs.as_ptr(), b.qs.as_ptr()) };
        }
    }
    scalar::dot_i32(&a.qs, &b.qs)
}

/// Integer dot product of one pair of Q8_0 blocks, forcing the scalar path.
pub fn dot_block_q8_0_i32_scalar(a: &BlockQ8_0, b: &BlockQ8_0) -> i32 {
    scalar::dot_i32(&a.qs, &b.qs)
}

fn dot_scaled(
    a: &[BlockQ8_0],
    b: &[BlockQ8_0],
    block_dot: fn(&BlockQ8_0, &BlockQ8_0) -> i32,
) -> f64 {
    let mut total = 0.0f64;
    for (left, right) in a.iter().zip(b.iter()) {
        total += f64::from(block_dot(left, right)) * left.scale_f64() * right.scale_f64();
    }
    total
}

fn dot_impl(
    a: &[BlockQ8_0],
    b: &[BlockQ8_0],
    block_dot: fn(&BlockQ8_0, &BlockQ8_0) -> i32,
) -> Result<f32, KernelError> {
    if a.len() != b.len() {
        return Err(KernelError::DotLengthMismatch {
            a_blocks: a.len(),
            b_blocks: b.len(),
        });
    }
    check_i32_budget(a.len() * QK8_0)?;
    Ok(dot_scaled(a, b, block_dot) as f32)
}

/// Reference Q8_0 dot product with `i32` block accumulation.
///
/// Both operands must cover the same number of blocks. Each block contributes
/// `d_a * d_b * sum(q_a[i] * q_b[i])`; the per-block sum is an `i32` and the
/// scaled contributions are accumulated in `f64` before the final `f32` cast so
/// the reference itself does not add avoidable rounding.
pub fn dot_q8_0(a: &[BlockQ8_0], b: &[BlockQ8_0]) -> Result<f32, KernelError> {
    dot_impl(a, b, dot_block_q8_0_i32)
}

/// Scalar-only Q8_0 dot product, for parity checks against the SIMD path.
pub fn dot_q8_0_scalar(a: &[BlockQ8_0], b: &[BlockQ8_0]) -> Result<f32, KernelError> {
    dot_impl(a, b, dot_block_q8_0_i32_scalar)
}

fn gemm_impl(
    a: &[BlockQ8_0],
    rows: usize,
    k: usize,
    b: &[BlockQ8_0],
    n: usize,
    out: &mut [f32],
    block_dot: fn(&BlockQ8_0, &BlockQ8_0) -> i32,
) -> Result<(), KernelError> {
    if !k.is_multiple_of(QK8_0) || !n.is_multiple_of(QK8_0) {
        return Err(KernelError::GemmShapeNotBlockAligned { k, n, block: QK8_0 });
    }
    check_i32_budget(k)?;

    let k_blocks = k / QK8_0;
    let expected_a = rows
        .checked_mul(k_blocks)
        .ok_or(KernelError::ShapeOverflow)?;
    // `b` stores the weight matrix output-major, `[n, k]`, which is how GGML
    // stores quantised weights: one row per output channel, `k` contiguous.
    let expected_b = n.checked_mul(k_blocks).ok_or(KernelError::ShapeOverflow)?;
    let expected_out = rows.checked_mul(n).ok_or(KernelError::ShapeOverflow)?;

    if a.len() != expected_a {
        return Err(KernelError::BufferSize {
            buffer: "a",
            expected: expected_a,
            actual: a.len(),
        });
    }
    if b.len() != expected_b {
        return Err(KernelError::BufferSize {
            buffer: "b",
            expected: expected_b,
            actual: b.len(),
        });
    }
    if out.len() != expected_out {
        return Err(KernelError::BufferSize {
            buffer: "out",
            expected: expected_out,
            actual: out.len(),
        });
    }

    out.fill(0.0);
    for row in 0..rows {
        for column in 0..n {
            let mut total = 0.0f64;
            for block in 0..k_blocks {
                let left = &a[row * k_blocks + block];
                let right = &b[column * k_blocks + block];
                total += f64::from(block_dot(left, right)) * left.scale_f64() * right.scale_f64();
            }
            out[row * n + column] = total as f32;
        }
    }
    Ok(())
}

/// Blocked Q8_0 GEMM: `out = a[rows, k] * b[k, n]`.
///
/// `a` holds `rows * k / 32` blocks in row-major order. `b` holds the weight
/// matrix **output-major**, `[n, k]` — one row per output column, `k` values
/// contiguous, `n * k / 32` blocks — which is how GGML stores quantised
/// weights; it computes the mathematical `[rows, k] x [k, n]` product. `out`
/// holds `rows * n` `f32` values. Both `k` and `n` must be multiples of 32 and
/// `k` must be at most [`MAX_SAFE_K`]; both constraints are enforced with
/// [`KernelError`]. The `rows` dimension is not constrained.
///
/// This is a straightforward reference loop, not a cache-blocked or tiled
/// implementation; see the crate-level claim status.
pub fn gemm_q8_0(
    a: &[BlockQ8_0],
    rows: usize,
    k: usize,
    b: &[BlockQ8_0],
    n: usize,
    out: &mut [f32],
) -> Result<(), KernelError> {
    gemm_impl(a, rows, k, b, n, out, dot_block_q8_0_i32)
}

/// Scalar-only blocked Q8_0 GEMM, for parity checks against the SIMD path.
pub fn gemm_q8_0_scalar(
    a: &[BlockQ8_0],
    rows: usize,
    k: usize,
    b: &[BlockQ8_0],
    n: usize,
    out: &mut [f32],
) -> Result<(), KernelError> {
    gemm_impl(a, rows, k, b, n, out, dot_block_q8_0_i32_scalar)
}

/// Fused gate/up projection for one activation row.
///
/// `x` is one quantised activation row of `k` values (`k / 32` blocks); `gate`
/// and `up` are `[n, k]` weight matrices in row-major block order; `out_gate` and
/// `out_up` receive `n` values each. Both projections share each `x` block load,
/// so the activation row is traversed once for a small tile of output columns.
///
/// This is a **scheduling helper**, not a performance claim: no receipt exists
/// that it is faster than two separate [`dot_q8_0`] calls. `n` must be a
/// multiple of 32 and `k` must respect [`MAX_SAFE_K`].
pub fn gate_up_row_q8_0(
    x: &[BlockQ8_0],
    gate: &[BlockQ8_0],
    up: &[BlockQ8_0],
    n: usize,
    out_gate: &mut [f32],
    out_up: &mut [f32],
) -> Result<(), KernelError> {
    if !n.is_multiple_of(QK8_0) {
        return Err(KernelError::GemmShapeNotBlockAligned {
            k: x.len() * QK8_0,
            n,
            block: QK8_0,
        });
    }
    let k_blocks = x.len();
    check_i32_budget(k_blocks * QK8_0)?;
    let expected_weights = n.checked_mul(k_blocks).ok_or(KernelError::ShapeOverflow)?;

    if gate.len() != expected_weights {
        return Err(KernelError::BufferSize {
            buffer: "gate",
            expected: expected_weights,
            actual: gate.len(),
        });
    }
    if up.len() != expected_weights {
        return Err(KernelError::BufferSize {
            buffer: "up",
            expected: expected_weights,
            actual: up.len(),
        });
    }
    if out_gate.len() != n {
        return Err(KernelError::BufferSize {
            buffer: "out_gate",
            expected: n,
            actual: out_gate.len(),
        });
    }
    if out_up.len() != n {
        return Err(KernelError::BufferSize {
            buffer: "out_up",
            expected: n,
            actual: out_up.len(),
        });
    }

    out_gate.fill(0.0);
    out_up.fill(0.0);
    if k_blocks == 0 {
        return Ok(());
    }

    // Tile the output columns so the shared `x` block stays in registers across
    // both projections for several columns before moving to the next block.
    const TILE: usize = 16;
    let mut base = 0;
    while base < n {
        let width = (n - base).min(TILE);
        let mut gate_accumulator = [0.0f64; TILE];
        let mut up_accumulator = [0.0f64; TILE];

        for block in 0..k_blocks {
            let activation = &x[block];
            let activation_scale = activation.scale_f64();
            for slot in 0..width {
                let column = base + slot;
                let gate_block = &gate[column * k_blocks + block];
                let up_block = &up[column * k_blocks + block];
                gate_accumulator[slot] += f64::from(dot_block_q8_0_i32(activation, gate_block))
                    * activation_scale
                    * gate_block.scale_f64();
                up_accumulator[slot] += f64::from(dot_block_q8_0_i32(activation, up_block))
                    * activation_scale
                    * up_block.scale_f64();
            }
        }

        for slot in 0..width {
            out_gate[base + slot] = gate_accumulator[slot] as f32;
            out_up[base + slot] = up_accumulator[slot] as f32;
        }
        base += width;
    }
    Ok(())
}

#[cfg(all(test, target_arch = "x86_64"))]
mod simd_tests {
    use super::*;

    struct Rng(u64);

    impl Rng {
        fn next_i8(&mut self) -> i8 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (((self.0 >> 33) % 255) as i16 - 127) as i8
        }
    }

    fn assert_avx2_matches_scalar(qs_a: [i8; QK8_0], qs_b: [i8; QK8_0]) {
        let a = BlockQ8_0::from_parts(1.0, qs_a);
        let b = BlockQ8_0::from_parts(1.0, qs_b);
        let expected = dot_block_q8_0_i32_scalar(&a, &b);
        // SAFETY: `avx2_available` performed runtime detection and neither block
        // contains -128 (both are in [-127, 127] by construction).
        let actual = unsafe { avx2::dot_i32(a.qs.as_ptr(), b.qs.as_ptr()) };
        assert_eq!(actual, expected);
    }

    #[test]
    fn avx2_integer_dot_matches_scalar() {
        if !avx2_available() {
            eprintln!("AVX2 not available on this host; parity test skipped");
            return;
        }
        let mut rng = Rng(0x5eed_1234_9abc_def1);
        for _ in 0..256 {
            let mut a = [0i8; QK8_0];
            let mut b = [0i8; QK8_0];
            for slot in 0..QK8_0 {
                a[slot] = rng.next_i8();
                b[slot] = rng.next_i8();
            }
            assert_avx2_matches_scalar(a, b);
        }
    }

    #[test]
    fn avx2_integer_dot_handles_extreme_rails() {
        if !avx2_available() {
            eprintln!("AVX2 not available on this host; parity test skipped");
            return;
        }
        assert_avx2_matches_scalar([127; QK8_0], [127; QK8_0]);
        assert_avx2_matches_scalar([-127; QK8_0], [-127; QK8_0]);
        assert_avx2_matches_scalar([127; QK8_0], [-127; QK8_0]);
        let mut alternating = [0i8; QK8_0];
        for (index, slot) in alternating.iter_mut().enumerate() {
            *slot = if index.is_multiple_of(2) { 127 } else { -127 };
        }
        assert_avx2_matches_scalar(alternating, [1; QK8_0]);
    }

    #[test]
    fn dispatch_falls_back_for_foreign_negative_128() {
        let mut qs = [7i8; QK8_0];
        qs[3] = -128;
        let a = BlockQ8_0::from_parts(1.0, qs);
        let b = BlockQ8_0::from_parts(1.0, [3; QK8_0]);
        // The dispatcher may pick AVX2, but it must exclude this block and still
        // return the scalar answer.
        assert_eq!(
            dot_block_q8_0_i32(&a, &b),
            dot_block_q8_0_i32_scalar(&a, &b)
        );
    }
}
