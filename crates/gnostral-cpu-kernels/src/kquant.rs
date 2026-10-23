//! GGML K-quant blocks (`Q2_K`, `Q3_K`) against `Q8_K` activations.
//!
//! The block structs are `#[repr(C)]` and byte-compatible with GGML's
//! `block_q2_K`, `block_q3_K` and `block_q8_K` (and with the vendored Candle
//! `BlockQ2K` / `BlockQ3K` / `BlockQ8K`), so a GGUF expert stack can be
//! reinterpreted in place without a copy.
//!
//! # Numerical contract
//!
//! Every dot product is split into an **exact integer part** per 256-value
//! block and a fixed floating-point combination:
//!
//! - `Q2_K`: `sum += (y.d * x.d) * isum - (y.d * x.dmin) * summs`, the same
//!   expression and order as the vendored scalar reference, so the scalar path
//!   here is bit-equal to Candle's `vec_dot_unopt` for `Q2_K`.
//! - `Q3_K`: `sum += (x.d * y.d) * isum` with one `i32` per block. Candle's
//!   scalar reference keeps eight partial `f32` lanes instead; the two agree
//!   within [`crate::kquant_dot_error_bound`], not bit for bit.
//!
//! The SIMD paths only compute the integer part. Because integer arithmetic is
//! exact and the float combination is shared, **scalar, AVX2 and AVX-512 return
//! bit-identical results**; tests assert equality, not a tolerance.
//!
//! `Q8_K` quantisation follows the vendored Candle rule (`round` half away from
//! zero, `iscale = -127 / max` where `max` is the signed value of largest
//! magnitude). GGML's reference rounds half to even; the difference only shows
//! on exact `.5` ties and is documented, not hidden.

use crate::f16::{f16_to_f32, f32_to_f16_bits};
use crate::KernelError;

/// Values per K-quant super-block.
pub const QK_K: usize = 256;
/// Values per K-quant sub-block (one scale each).
const SUB: usize = 16;
const SUBS: usize = QK_K / SUB;

/// One `Q8_K` activation block: `f32` scale, 256 `i8` values, 16 sub-block sums.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct BlockQ8K {
    pub d: f32,
    pub qs: [i8; QK_K],
    pub bsums: [i16; SUBS],
}
const _: () = assert!(std::mem::size_of::<BlockQ8K>() == 292);

/// One `Q2_K` weight block (84 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct BlockQ2K {
    /// Low nibble: scale, high nibble: min, one byte per 16-value sub-block.
    pub scales: [u8; SUBS],
    pub qs: [u8; QK_K / 4],
    /// `f16` bits.
    pub d: u16,
    /// `f16` bits.
    pub dmin: u16,
}
const _: () = assert!(std::mem::size_of::<BlockQ2K>() == 84);

/// One `Q3_K` weight block (110 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct BlockQ3K {
    pub hmask: [u8; QK_K / 8],
    pub qs: [u8; QK_K / 4],
    /// Sixteen 6-bit scales packed into 12 bytes.
    pub scales: [u8; 12],
    /// `f16` bits.
    pub d: u16,
}
const _: () = assert!(std::mem::size_of::<BlockQ3K>() == 110);

impl BlockQ8K {
    pub const ZERO: Self = Self {
        d: 0.0,
        qs: [0; QK_K],
        bsums: [0; SUBS],
    };
}

/// Quantise `xs` (a multiple of 256 values) into `Q8_K` blocks.
pub fn quantize_row_q8k(xs: &[f32], ys: &mut [BlockQ8K]) -> Result<(), KernelError> {
    if !xs.len().is_multiple_of(QK_K) {
        return Err(KernelError::RowNotBlockAligned {
            len: xs.len(),
            block: QK_K,
        });
    }
    if ys.len() != xs.len() / QK_K {
        return Err(KernelError::BufferSize {
            buffer: "q8k",
            expected: xs.len() / QK_K,
            actual: ys.len(),
        });
    }
    if let Some(index) = xs.iter().position(|x| !x.is_finite()) {
        return Err(KernelError::NonFiniteInput { index });
    }
    for (x, y) in xs.chunks_exact(QK_K).zip(ys.iter_mut()) {
        quantize_block_q8k(x, y);
    }
    Ok(())
}

fn quantize_block_q8k(x: &[f32], y: &mut BlockQ8K) {
    let mut max = 0f32;
    let mut amax = 0f32;
    for &v in x {
        if amax < v.abs() {
            amax = v.abs();
            max = v;
        }
    }
    if amax == 0.0 {
        *y = BlockQ8K::ZERO;
        return;
    }
    let iscale = -127f32 / max;
    for (q, &v) in y.qs.iter_mut().zip(x) {
        *q = (iscale * v).round().clamp(-128.0, 127.0) as i8;
    }
    for (j, sum) in y.bsums.iter_mut().enumerate() {
        *sum = y.qs[j * SUB..(j + 1) * SUB]
            .iter()
            .map(|&q| i32::from(q))
            .sum::<i32>() as i16;
    }
    y.d = 1.0 / iscale;
}

/// Dequantise `Q8_K` blocks.
pub fn dequantize_row_q8k(xs: &[BlockQ8K], ys: &mut [f32]) -> Result<(), KernelError> {
    check_dequant_len(xs.len(), ys.len())?;
    for (x, y) in xs.iter().zip(ys.chunks_exact_mut(QK_K)) {
        for (out, &q) in y.iter_mut().zip(&x.qs) {
            *out = x.d * f32::from(q);
        }
    }
    Ok(())
}

fn check_dequant_len(blocks: usize, values: usize) -> Result<(), KernelError> {
    let expected = blocks.checked_mul(QK_K).ok_or(KernelError::ShapeOverflow)?;
    if values != expected {
        return Err(KernelError::BufferSize {
            buffer: "dequantized",
            expected,
            actual: values,
        });
    }
    Ok(())
}

impl BlockQ2K {
    /// Unsigned 2-bit value of element `e` (0..256), in `Q8_K` element order.
    #[inline]
    fn u(&self, e: usize) -> u8 {
        let group = e / 32;
        let chunk = group / 4;
        let shift = 2 * (group % 4);
        (self.qs[chunk * 32 + e % 32] >> shift) & 3
    }
}

impl BlockQ3K {
    /// The sixteen 6-bit scales, unbiased (`raw - 32`).
    pub fn unpacked_scales(&self) -> [i8; SUBS] {
        const KMASK1: u32 = 0x0303_0303;
        const KMASK2: u32 = 0x0f0f_0f0f;
        let word = |i: usize| {
            u32::from_le_bytes([
                self.scales[4 * i],
                self.scales[4 * i + 1],
                self.scales[4 * i + 2],
                self.scales[4 * i + 3],
            ])
        };
        let (a0, a1, tmp) = (word(0), word(1), word(2));
        let aux = [
            (a0 & KMASK2) | ((tmp & KMASK1) << 4),
            (a1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4),
            ((a0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4),
            ((a1 >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4),
        ];
        let mut out = [0i8; SUBS];
        for (i, word) in aux.iter().enumerate() {
            for (j, byte) in word.to_le_bytes().iter().enumerate() {
                out[4 * i + j] = *byte as i8 - 32;
            }
        }
        out
    }

    /// Unsigned 3-bit value `u` of element `e`; the signed value is `u - 4`.
    #[inline]
    fn u(&self, e: usize) -> u8 {
        let group = e / 32;
        let chunk = group / 4;
        let shift = 2 * (group % 4);
        let low = (self.qs[chunk * 32 + e % 32] >> shift) & 3;
        let high = (self.hmask[e % 32] >> group) & 1;
        low | (high << 2)
    }
}

/// Dequantise `Q2_K` blocks with the GGML formula `d * scale * q - dmin * min`.
pub fn dequantize_row_q2k(xs: &[BlockQ2K], ys: &mut [f32]) -> Result<(), KernelError> {
    check_dequant_len(xs.len(), ys.len())?;
    for (x, y) in xs.iter().zip(ys.chunks_exact_mut(QK_K)) {
        let d = f16_to_f32(x.d);
        let min = f16_to_f32(x.dmin);
        for (e, out) in y.iter_mut().enumerate() {
            let sc = x.scales[e / SUB];
            let dl = d * f32::from(sc & 0xF);
            let ml = min * f32::from(sc >> 4);
            *out = dl * f32::from(x.u(e)) - ml;
        }
    }
    Ok(())
}

/// Dequantise `Q3_K` blocks with the GGML formula `d * (scale - 32) * (q - 4 * !hbit)`.
pub fn dequantize_row_q3k(xs: &[BlockQ3K], ys: &mut [f32]) -> Result<(), KernelError> {
    check_dequant_len(xs.len(), ys.len())?;
    for (x, y) in xs.iter().zip(ys.chunks_exact_mut(QK_K)) {
        let d = f16_to_f32(x.d);
        let scales = x.unpacked_scales();
        for (e, out) in y.iter_mut().enumerate() {
            let dl = d * f32::from(scales[e / SUB]);
            *out = dl * f32::from(x.u(e) as i8 - 4);
        }
    }
    Ok(())
}

/// Instruction set used for the integer part of a K-quant dot product.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Isa {
    Scalar,
    Avx2,
    Avx512,
}

impl Isa {
    /// Best ISA this process can run.
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx512bw") {
                return Isa::Avx512;
            }
            if std::arch::is_x86_feature_detected!("avx2") {
                return Isa::Avx2;
            }
        }
        Isa::Scalar
    }

    /// True when this process can run `self`.
    pub fn available(self) -> bool {
        match self {
            Isa::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => std::arch::is_x86_feature_detected!("avx2"),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => std::arch::is_x86_feature_detected!("avx512bw"),
            #[cfg(not(target_arch = "x86_64"))]
            _ => false,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Isa::Scalar => "scalar",
            Isa::Avx2 => "avx2",
            Isa::Avx512 => "avx512bw",
        }
    }
}

/// A K-quant weight block that can be dotted against `Q8_K` activations.
pub trait KQuantBlock: Copy + Send + Sync + 'static {
    const NAME: &'static str;
    /// Dot product of one weight row (`w.len()` blocks) with one activation row.
    ///
    /// # Panics
    ///
    /// Panics if `isa` is not available; use [`Isa::available`] or [`Isa::detect`].
    fn dot(isa: Isa, w: &[Self], a: &[BlockQ8K]) -> f32;
    fn dequantize(xs: &[Self], ys: &mut [f32]) -> Result<(), KernelError>;
}

impl KQuantBlock for BlockQ2K {
    const NAME: &'static str = "q2_k";

    fn dot(isa: Isa, w: &[Self], a: &[BlockQ8K]) -> f32 {
        assert_eq!(w.len(), a.len(), "q2_k dot length mismatch");
        assert!(isa.available(), "{} unavailable", isa.name());
        let mut sumf = 0f32;
        for (x, y) in w.iter().zip(a) {
            let isum = match isa {
                Isa::Scalar => q2k_isum_scalar(x, y),
                #[cfg(target_arch = "x86_64")]
                // SAFETY: availability asserted above.
                Isa::Avx2 => unsafe { crate::kquant_simd::q2k_isum_avx2(x, y) },
                #[cfg(target_arch = "x86_64")]
                // SAFETY: availability asserted above.
                Isa::Avx512 => unsafe { crate::kquant_simd::q2k_isum_avx512(x, y) },
                #[cfg(not(target_arch = "x86_64"))]
                _ => unreachable!(),
            };
            let summs = q2k_summs(x, y);
            let dall = y.d * f16_to_f32(x.d);
            let dmin = y.d * f16_to_f32(x.dmin);
            sumf += dall * isum as f32 - dmin * summs as f32;
        }
        sumf
    }

    fn dequantize(xs: &[Self], ys: &mut [f32]) -> Result<(), KernelError> {
        dequantize_row_q2k(xs, ys)
    }
}

impl KQuantBlock for BlockQ3K {
    const NAME: &'static str = "q3_k";

    fn dot(isa: Isa, w: &[Self], a: &[BlockQ8K]) -> f32 {
        assert_eq!(w.len(), a.len(), "q3_k dot length mismatch");
        assert!(isa.available(), "{} unavailable", isa.name());
        let mut sumf = 0f32;
        for (x, y) in w.iter().zip(a) {
            let scales = x.unpacked_scales();
            let unsigned = match isa {
                Isa::Scalar => q3k_isum_scalar(x, y, &scales),
                #[cfg(target_arch = "x86_64")]
                // SAFETY: availability asserted above.
                Isa::Avx2 => unsafe { crate::kquant_simd::q3k_isum_avx2(x, y, &scales) },
                #[cfg(target_arch = "x86_64")]
                // SAFETY: availability asserted above.
                Isa::Avx512 => unsafe { crate::kquant_simd::q3k_isum_avx512(x, y, &scales) },
                #[cfg(not(target_arch = "x86_64"))]
                _ => unreachable!(),
            };
            let offset: i32 = scales
                .iter()
                .zip(&y.bsums)
                .map(|(&s, &b)| i32::from(s) * i32::from(b))
                .sum();
            let isum = unsigned - 4 * offset;
            sumf += (f16_to_f32(x.d) * y.d) * isum as f32;
        }
        sumf
    }

    fn dequantize(xs: &[Self], ys: &mut [f32]) -> Result<(), KernelError> {
        dequantize_row_q3k(xs, ys)
    }
}

/// `sum_j (scale_j & 0xF) * sum_{e in j} u_e * q8_e`.
pub(crate) fn q2k_isum_scalar(x: &BlockQ2K, y: &BlockQ8K) -> i32 {
    let mut isum = 0i32;
    for j in 0..SUBS {
        let mut sub = 0i32;
        for e in j * SUB..(j + 1) * SUB {
            sub += i32::from(x.u(e)) * i32::from(y.qs[e]);
        }
        isum += i32::from(x.scales[j] & 0xF) * sub;
    }
    isum
}

fn q2k_summs(x: &BlockQ2K, y: &BlockQ8K) -> i32 {
    x.scales
        .iter()
        .zip(&y.bsums)
        .map(|(&s, &b)| i32::from(s >> 4) * i32::from(b))
        .sum()
}

/// `sum_j scale_j * sum_{e in j} u_e * q8_e` with unbiased scales and unsigned `u` in `0..8`.
pub(crate) fn q3k_isum_scalar(x: &BlockQ3K, y: &BlockQ8K, scales: &[i8; SUBS]) -> i32 {
    let mut isum = 0i32;
    for (j, &scale) in scales.iter().enumerate() {
        let mut sub = 0i32;
        for e in j * SUB..(j + 1) * SUB {
            sub += i32::from(x.u(e)) * i32::from(y.qs[e]);
        }
        isum += i32::from(scale) * sub;
    }
    isum
}

/// Reinterpret a little-endian GGUF byte slice as K-quant blocks without copying.
///
/// Returns `None` when the length is not a whole number of blocks or the
/// pointer is not aligned for `T`.
pub fn blocks_from_bytes<T: KQuantBlock>(bytes: &[u8]) -> Option<&[T]> {
    let size = std::mem::size_of::<T>();
    if !bytes.len().is_multiple_of(size)
        || !(bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<T>())
        || cfg!(target_endian = "big")
    {
        return None;
    }
    // SAFETY: length and alignment checked; every bit pattern is a valid block
    // (plain integer arrays and integer-encoded f16 fields).
    Some(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, bytes.len() / size) })
}

/// Build a `Q2_K` block from parts (used by tests and fixtures).
pub fn block_q2k(scales: [u8; SUBS], qs: [u8; QK_K / 4], d: f32, dmin: f32) -> BlockQ2K {
    BlockQ2K {
        scales,
        qs,
        d: f32_to_f16_bits(d),
        dmin: f32_to_f16_bits(dmin),
    }
}

/// Build a `Q3_K` block from parts (used by tests and fixtures).
pub fn block_q3k(hmask: [u8; QK_K / 8], qs: [u8; QK_K / 4], scales: [u8; 12], d: f32) -> BlockQ3K {
    BlockQ3K {
        hmask,
        qs,
        scales,
        d: f32_to_f16_bits(d),
    }
}
