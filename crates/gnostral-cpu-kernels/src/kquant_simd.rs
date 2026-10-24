//! AVX2 and AVX-512BW integer parts of the K-quant dot products.
//!
//! Each function returns exactly the same eight `i32` lanes as its scalar
//! counterpart in [`crate::kquant`] (lane `j` holds the values `e` with
//! `(e % 32) / 4 == j`, Candle's AVX2 order): `u8 x i8` pairs go through
//! `maddubs` (no saturation is reachable: `u <= 7`, `|q8| <= 128`, so a pair sum
//! is at most 1792) and the per-sub-block scales through `madd` into `i32` lanes.

#![cfg(target_arch = "x86_64")]

use crate::f16::f16_to_f32;
use crate::kquant::{BlockQ2K, BlockQ3K, BlockQ8K};
use core::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2")]
fn lanes_avx2(acc: __m256i) -> [i32; 8] {
    let mut out = [0i32; 8];
    // SAFETY: `out` is 32 writable bytes.
    unsafe { _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, acc) };
    out
}

/// Lanes `j` and `j + 8` hold the same `(e % 32) / 4` class; fold them into eight lanes.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx2")]
fn fold_avx512(acc: __m512i) -> __m256i {
    _mm256_add_epi32(
        _mm512_castsi512_si256(acc),
        _mm512_extracti64x4_epi64::<1>(acc),
    )
}

/// `maddubs(u, q8) - maddubs(4, q8)`: the signed `Q3_K` value `u - 4` without a signed operand.
#[inline]
#[target_feature(enable = "avx2")]
fn q3k_signed_pairs_avx2(u: __m256i, q8: __m256i) -> __m256i {
    _mm256_sub_epi16(
        _mm256_maddubs_epi16(u, q8),
        _mm256_maddubs_epi16(_mm256_set1_epi8(4), q8),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn pair_scales_avx2(a: i16, b: i16) -> __m256i {
    _mm256_setr_epi16(a, a, a, a, a, a, a, a, b, b, b, b, b, b, b, b)
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn quad_scales_avx512(s: [i16; 4]) -> __m512i {
    let mut lanes = [0i16; 32];
    for (i, lane) in lanes.iter_mut().enumerate() {
        *lane = s[i / 8];
    }
    // SAFETY: `lanes` is 64 readable bytes.
    unsafe { _mm512_loadu_si512(lanes.as_ptr() as *const __m512i) }
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn join(lo: __m256i, hi: __m256i) -> __m512i {
    _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi)
}

/// # Safety
/// AVX2 must be available.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn q2k_lanes_avx2(x: &BlockQ2K, y: &BlockQ8K) -> [i32; 8] {
    // SAFETY: same target features as this function.
    lanes_avx2(unsafe { q2k_acc_avx2(x, y) })
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn q2k_acc_avx2(x: &BlockQ2K, y: &BlockQ8K) -> __m256i {
    let mask = _mm256_set1_epi8(3);
    let mut acc = _mm256_setzero_si256();
    for chunk in 0..2 {
        // SAFETY: qs has 64 bytes; chunk * 32 + 32 <= 64.
        let bits = unsafe { _mm256_loadu_si256(x.qs.as_ptr().add(chunk * 32) as *const __m256i) };
        for s in 0..4 {
            let g = chunk * 4 + s;
            let v = _mm256_and_si256(
                _mm256_srl_epi16(bits, _mm_cvtsi32_si128(2 * s as i32)),
                mask,
            );
            // SAFETY: y.qs has 256 bytes; g * 32 + 32 <= 256.
            let q8 = unsafe { _mm256_loadu_si256(y.qs.as_ptr().add(g * 32) as *const __m256i) };
            let p = _mm256_maddubs_epi16(v, q8);
            let sv = pair_scales_avx2(
                i16::from(x.scales[2 * g] & 0xF),
                i16::from(x.scales[2 * g + 1] & 0xF),
            );
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, sv));
        }
    }
    acc
}

/// # Safety
/// AVX-512F and AVX-512BW must be available.
#[target_feature(enable = "avx512f,avx512bw,avx2")]
pub(crate) unsafe fn q2k_lanes_avx512(x: &BlockQ2K, y: &BlockQ8K) -> [i32; 8] {
    // SAFETY: same target features as this function.
    lanes_avx2(unsafe { q2k_acc_avx512(x, y) })
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx2")]
unsafe fn q2k_acc_avx512(x: &BlockQ2K, y: &BlockQ8K) -> __m256i {
    let mask = _mm256_set1_epi8(3);
    let mut acc = _mm512_setzero_si512();
    for chunk in 0..2 {
        // SAFETY: as in the AVX2 path.
        let bits = unsafe { _mm256_loadu_si256(x.qs.as_ptr().add(chunk * 32) as *const __m256i) };
        for half in 0..2 {
            let s0 = 2 * half;
            let g = chunk * 4 + s0;
            let v0 = _mm256_and_si256(
                _mm256_srl_epi16(bits, _mm_cvtsi32_si128(2 * s0 as i32)),
                mask,
            );
            let v1 = _mm256_and_si256(
                _mm256_srl_epi16(bits, _mm_cvtsi32_si128(2 * (s0 + 1) as i32)),
                mask,
            );
            // SAFETY: g * 32 + 64 <= 256.
            let q8 = unsafe { _mm512_loadu_si512(y.qs.as_ptr().add(g * 32) as *const __m512i) };
            let p = _mm512_maddubs_epi16(join(v0, v1), q8);
            let sv = quad_scales_avx512([
                i16::from(x.scales[2 * g] & 0xF),
                i16::from(x.scales[2 * g + 1] & 0xF),
                i16::from(x.scales[2 * g + 2] & 0xF),
                i16::from(x.scales[2 * g + 3] & 0xF),
            ]);
            acc = _mm512_add_epi32(acc, _mm512_madd_epi16(p, sv));
        }
    }
    fold_avx512(acc)
}

#[inline]
#[target_feature(enable = "avx2")]
fn q3k_unsigned_avx2(bits: __m256i, hmask: __m256i, g: usize) -> __m256i {
    let shift = 2 * (g % 4) as i32;
    let low = _mm256_and_si256(
        _mm256_srl_epi16(bits, _mm_cvtsi32_si128(shift)),
        _mm256_set1_epi8(3),
    );
    let high = _mm256_and_si256(
        _mm256_srl_epi16(hmask, _mm_cvtsi32_si128(g as i32)),
        _mm256_set1_epi8(1),
    );
    _mm256_or_si256(low, _mm256_slli_epi16::<2>(high))
}

/// # Safety
/// AVX2 must be available.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn q3k_lanes_avx2(x: &BlockQ3K, y: &BlockQ8K, scales: &[i8; 16]) -> [i32; 8] {
    // SAFETY: same target features as this function.
    lanes_avx2(unsafe { q3k_acc_avx2(x, y, scales) })
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn q3k_acc_avx2(x: &BlockQ3K, y: &BlockQ8K, scales: &[i8; 16]) -> __m256i {
    // SAFETY: hmask has 32 bytes.
    let hmask = unsafe { _mm256_loadu_si256(x.hmask.as_ptr() as *const __m256i) };
    let mut acc = _mm256_setzero_si256();
    for chunk in 0..2 {
        // SAFETY: qs has 64 bytes.
        let bits = unsafe { _mm256_loadu_si256(x.qs.as_ptr().add(chunk * 32) as *const __m256i) };
        for s in 0..4 {
            let g = chunk * 4 + s;
            let u = q3k_unsigned_avx2(bits, hmask, g);
            // SAFETY: g * 32 + 32 <= 256.
            let q8 = unsafe { _mm256_loadu_si256(y.qs.as_ptr().add(g * 32) as *const __m256i) };
            let p = q3k_signed_pairs_avx2(u, q8);
            let sv = pair_scales_avx2(i16::from(scales[2 * g]), i16::from(scales[2 * g + 1]));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, sv));
        }
    }
    acc
}

/// # Safety
/// AVX-512F and AVX-512BW must be available.
#[target_feature(enable = "avx512f,avx512bw,avx2")]
pub(crate) unsafe fn q3k_lanes_avx512(x: &BlockQ3K, y: &BlockQ8K, scales: &[i8; 16]) -> [i32; 8] {
    // SAFETY: same target features as this function.
    lanes_avx2(unsafe { q3k_acc_avx512(x, y, scales) })
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx2")]
unsafe fn q3k_acc_avx512(x: &BlockQ3K, y: &BlockQ8K, scales: &[i8; 16]) -> __m256i {
    // SAFETY: hmask has 32 bytes.
    let hmask = unsafe { _mm256_loadu_si256(x.hmask.as_ptr() as *const __m256i) };
    let mut acc = _mm512_setzero_si512();
    for chunk in 0..2 {
        // SAFETY: qs has 64 bytes.
        let bits = unsafe { _mm256_loadu_si256(x.qs.as_ptr().add(chunk * 32) as *const __m256i) };
        for half in 0..2 {
            let g = chunk * 4 + 2 * half;
            let u = join(
                q3k_unsigned_avx2(bits, hmask, g),
                q3k_unsigned_avx2(bits, hmask, g + 1),
            );
            // SAFETY: g * 32 + 64 <= 256.
            let q8 = unsafe { _mm512_loadu_si512(y.qs.as_ptr().add(g * 32) as *const __m512i) };
            let p = _mm512_sub_epi16(
                _mm512_maddubs_epi16(u, q8),
                _mm512_maddubs_epi16(_mm512_set1_epi8(4), q8),
            );
            let sv = quad_scales_avx512([
                i16::from(scales[2 * g]),
                i16::from(scales[2 * g + 1]),
                i16::from(scales[2 * g + 2]),
                i16::from(scales[2 * g + 3]),
            ]);
            acc = _mm512_add_epi32(acc, _mm512_madd_epi16(p, sv));
        }
    }
    fold_avx512(acc)
}

/// Candle `hsum_float_8`: `((a4 + a0) + (a6 + a2)) + ((a5 + a1) + (a7 + a3))`.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn hsum_float_8(x: __m256) -> f32 {
    let r = _mm_add_ps(_mm256_extractf128_ps::<1>(x), _mm256_castps256_ps128(x));
    let r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    let r = _mm_add_ss(r, _mm_movehdup_ps(r));
    _mm_cvtss_f32(r)
}

/// One block of candle's `Q2_K` float combination: `fma(dmin, mins x bsums)` then `fma(d, sumi)`.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn q2k_candle_step(acc: __m256, x: &BlockQ2K, y: &BlockQ8K, sumi: __m256i) -> __m256 {
    let d = y.d * f16_to_f32(x.d);
    let dmin = -y.d * f16_to_f32(x.dmin);
    // SAFETY: scales has 16 bytes, bsums 16 i16 (32 bytes).
    let (scales, bsums) = unsafe {
        (
            _mm_loadu_si128(x.scales.as_ptr() as *const __m128i),
            _mm256_loadu_si256(y.bsums.as_ptr() as *const __m256i),
        )
    };
    let mins = _mm256_cvtepi8_epi16(_mm_and_si128(
        _mm_srli_epi16::<4>(scales),
        _mm_set1_epi8(0xF),
    ));
    let prod = _mm256_madd_epi16(mins, bsums);
    let acc = _mm256_fmadd_ps(_mm256_set1_ps(dmin), _mm256_cvtepi32_ps(prod), acc);
    _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc)
}

/// `Q2_K` dot with candle's AVX2+FMA float combination, bit-equal to candle's `vec_dot_q2k_q8k`.
///
/// # Safety
/// AVX2 and FMA must be available.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn q2k_dot_candle_avx2(w: &[BlockQ2K], a: &[BlockQ8K]) -> f32 {
    let mut acc = _mm256_setzero_ps();
    for (x, y) in w.iter().zip(a) {
        // SAFETY: same target features as this function.
        acc = unsafe { q2k_candle_step(acc, x, y, q2k_acc_avx2(x, y)) };
    }
    hsum_float_8(acc)
}

/// [`q2k_dot_candle_avx2`] with the AVX-512BW integer part.
///
/// # Safety
/// AVX-512F, AVX-512BW, AVX2 and FMA must be available.
#[target_feature(enable = "avx512f,avx512bw,avx2,fma")]
pub(crate) unsafe fn q2k_dot_candle_avx512(w: &[BlockQ2K], a: &[BlockQ8K]) -> f32 {
    let mut acc = _mm256_setzero_ps();
    for (x, y) in w.iter().zip(a) {
        // SAFETY: same target features as this function.
        acc = unsafe { q2k_candle_step(acc, x, y, q2k_acc_avx512(x, y)) };
    }
    hsum_float_8(acc)
}

/// `Q3_K` dot with candle's AVX2+FMA float combination (`fma(d, sumi)` per block).
///
/// # Safety
/// AVX2 and FMA must be available.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn q3k_dot_candle_avx2(w: &[BlockQ3K], a: &[BlockQ8K]) -> f32 {
    let mut acc = _mm256_setzero_ps();
    for (x, y) in w.iter().zip(a) {
        let d = _mm256_set1_ps(y.d * f16_to_f32(x.d));
        // SAFETY: same target features as this function.
        let sumi = unsafe { q3k_acc_avx2(x, y, &x.unpacked_scales()) };
        acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi), acc);
    }
    hsum_float_8(acc)
}

/// [`q3k_dot_candle_avx2`] with the AVX-512BW integer part.
///
/// # Safety
/// AVX-512F, AVX-512BW, AVX2 and FMA must be available.
#[target_feature(enable = "avx512f,avx512bw,avx2,fma")]
pub(crate) unsafe fn q3k_dot_candle_avx512(w: &[BlockQ3K], a: &[BlockQ8K]) -> f32 {
    let mut acc = _mm256_setzero_ps();
    for (x, y) in w.iter().zip(a) {
        let d = _mm256_set1_ps(y.d * f16_to_f32(x.d));
        // SAFETY: same target features as this function.
        let sumi = unsafe { q3k_acc_avx512(x, y, &x.unpacked_scales()) };
        acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi), acc);
    }
    hsum_float_8(acc)
}
