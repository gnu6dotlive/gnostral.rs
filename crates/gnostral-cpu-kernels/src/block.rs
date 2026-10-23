//! The Q8_0 GGML block and its row-level quantise/dequantise operations.

use crate::f16::{f16_to_f32, f32_to_f16_bits};
use crate::KernelError;

/// Values per Q8_0 block.
pub const QK8_0: usize = 32;
/// Bytes occupied by one serialised Q8_0 block: one `f16` scale plus 32 `i8`.
pub const BLOCK_Q8_0_BYTES: usize = 34;

/// One GGML Q8_0 block: `d * qs[i]` reconstructs the original value.
///
/// `d_bits` is the raw `f16` bit pattern of the scale. Keeping the bits rather
/// than an `f32` makes the struct exactly 34 bytes and lets the block be written
/// to or read from a weight file without re-rounding the scale.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockQ8_0 {
    /// Binary16 bit pattern of the block scale.
    pub d_bits: u16,
    /// Quantised values in `[-127, 127]` as emitted by this crate.
    pub qs: [i8; QK8_0],
}

impl BlockQ8_0 {
    /// An all-zero block (`d = 0`, all `q = 0`).
    pub const fn zero() -> Self {
        Self {
            d_bits: 0,
            qs: [0; QK8_0],
        }
    }

    /// Build a block from a scale and quantised values.
    ///
    /// The scale is rounded to binary16 with round-to-nearest-even. A zero or
    /// negative scale maps to `0`; callers that need a negative scale must
    /// negate the quantised values instead, as GGML does.
    pub fn from_parts(scale: f32, qs: [i8; QK8_0]) -> Self {
        let d_bits = if scale > 0.0 && scale.is_finite() {
            f32_to_f16_bits(scale)
        } else {
            0
        };
        Self { d_bits, qs }
    }

    /// The block scale as `f32` (exactly the stored binary16 value).
    pub fn scale(&self) -> f32 {
        f16_to_f32(self.d_bits)
    }

    /// The block scale as `f64`, for error accounting in tests.
    pub fn scale_f64(&self) -> f64 {
        f64::from(self.scale())
    }

    /// True when the block reconstructs to all zeros.
    pub fn is_zero(&self) -> bool {
        self.d_bits == 0 || self.qs.iter().all(|&q| q == 0)
    }

    /// Serialise to the 34-byte GGML layout, little-endian.
    pub fn to_bytes(&self) -> [u8; BLOCK_Q8_0_BYTES] {
        let mut out = [0u8; BLOCK_Q8_0_BYTES];
        out[..2].copy_from_slice(&self.d_bits.to_le_bytes());
        for (dst, &q) in out[2..].iter_mut().zip(self.qs.iter()) {
            *dst = q as u8;
        }
        out
    }

    /// Parse the 34-byte GGML layout, little-endian.
    pub fn from_bytes(bytes: &[u8; BLOCK_Q8_0_BYTES]) -> Self {
        let mut block = Self::zero();
        block.d_bits = u16::from_le_bytes([bytes[0], bytes[1]]);
        for (q, &byte) in block.qs.iter_mut().zip(bytes[2..].iter()) {
            *q = byte as i8;
        }
        block
    }
}

impl Default for BlockQ8_0 {
    fn default() -> Self {
        Self::zero()
    }
}

/// Quantise a flat `f32` row into Q8_0 blocks.
///
/// `values.len()` must be a multiple of 32; the constraint is enforced with
/// [`KernelError::RowNotBlockAligned`] rather than silently dropping a tail.
///
/// Per block: `d_exact = max_abs / 127`, the stored scale is `d = f16(d_exact)`,
/// and `q[i] = clamp(round(x[i] / d), -127, 127)` with `round` being round half
/// away from zero. The stored (already rounded) scale is used for quantisation,
/// so `q * d` is the closest reconstruction for the scale that is actually
/// written to the weight. A block whose values are all zero stores `d = 0` and
/// `q = 0`. A block whose scale underflows binary16 stores `d = 0` and `q = 0`
/// (values below roughly `3.8e-6`); this is documented flush-to-zero, not a
/// silent sign change.
pub fn quantize_row_q8_0(values: &[f32]) -> Result<Vec<BlockQ8_0>, KernelError> {
    if !values.len().is_multiple_of(QK8_0) {
        return Err(KernelError::RowNotBlockAligned {
            len: values.len(),
            block: QK8_0,
        });
    }

    let mut blocks = Vec::with_capacity(values.len() / QK8_0);
    for (block_index, chunk) in values.chunks_exact(QK8_0).enumerate() {
        let mut max_abs = 0.0f32;
        for (offset, &value) in chunk.iter().enumerate() {
            if !value.is_finite() {
                return Err(KernelError::NonFiniteInput {
                    index: block_index * QK8_0 + offset,
                });
            }
            max_abs = max_abs.max(value.abs());
        }

        let mut block = BlockQ8_0::zero();
        if max_abs > 0.0 {
            let d_bits = f32_to_f16_bits(max_abs / 127.0);
            let d = f16_to_f32(d_bits);
            if d > 0.0 {
                block.d_bits = d_bits;
                let inverse = 1.0 / d;
                for (slot, &value) in block.qs.iter_mut().zip(chunk.iter()) {
                    *slot = (value * inverse).round().clamp(-127.0, 127.0) as i8;
                }
            }
        }
        blocks.push(block);
    }
    Ok(blocks)
}

/// Reconstruct a flat `f32` row from Q8_0 blocks.
///
/// Every block contributes exactly 32 values, in order.
pub fn dequantize_row_q8_0(blocks: &[BlockQ8_0]) -> Vec<f32> {
    let mut values = Vec::with_capacity(blocks.len() * QK8_0);
    for block in blocks {
        let d = block.scale();
        values.extend(block.qs.iter().map(|&q| f32::from(q) * d));
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_is_34_bytes() {
        assert_eq!(core::mem::size_of::<BlockQ8_0>(), BLOCK_Q8_0_BYTES);
    }

    #[test]
    fn bytes_round_trip_including_a_foreign_negative_128() {
        let mut qs = [0i8; QK8_0];
        for (index, slot) in qs.iter_mut().enumerate() {
            *slot = (index as i8) - 16;
        }
        qs[31] = -128;
        let block = BlockQ8_0::from_parts(0.125, qs);
        let bytes = block.to_bytes();
        assert_eq!(bytes[0..2], 0x3000u16.to_le_bytes());
        assert_eq!(BlockQ8_0::from_bytes(&bytes), block);
    }

    #[test]
    fn zero_row_throws_away_the_scale() {
        let blocks = quantize_row_q8_0(&[0.0f32; QK8_0]).unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0], BlockQ8_0::zero());
        assert_eq!(blocks[0].scale(), 0.0);
        assert!(dequantize_row_q8_0(&blocks).iter().all(|&v| v == 0.0));
    }

    #[test]
    fn max_magnitude_saturates_at_the_i8_rail() {
        let mut values = [0.0f32; QK8_0];
        values[0] = 700.0;
        values[1] = -700.0;
        let blocks = quantize_row_q8_0(&values).unwrap();
        assert_eq!(blocks[0].qs[0], 127);
        assert_eq!(blocks[0].qs[1], -127);
        assert!(blocks[0].qs.iter().all(|&q| (-127..=127).contains(&q)));
    }

    #[test]
    fn rejects_a_tail_shorter_than_a_block() {
        assert_eq!(
            quantize_row_q8_0(&[1.0f32; 31]).unwrap_err(),
            KernelError::RowNotBlockAligned { len: 31, block: 32 }
        );
        assert_eq!(
            quantize_row_q8_0(&[1.0f32; 33]).unwrap_err(),
            KernelError::RowNotBlockAligned { len: 33, block: 32 }
        );
    }

    #[test]
    fn rejects_non_finite_input() {
        let mut values = [0.5f32; QK8_0];
        values[7] = f32::NAN;
        assert_eq!(
            quantize_row_q8_0(&values).unwrap_err(),
            KernelError::NonFiniteInput { index: 7 }
        );
        values[7] = f32::INFINITY;
        assert_eq!(
            quantize_row_q8_0(&values).unwrap_err(),
            KernelError::NonFiniteInput { index: 7 }
        );
    }
}
