//! IEEE 754 binary16 (`f16`) to binary32 (`f32`) conversion by bit manipulation.
//!
//! The Q8_0 GGML block stores its scale as a half-precision float. The crate
//! deliberately has no dependencies, so the conversion is implemented here
//! instead of pulling in the `half` crate.
//!
//! The routines implement the IEEE 754 default rounding mode (round to nearest,
//! ties to even), preserve signed zero, map overflowing magnitudes to infinity,
//! and keep the payload of NaN values that originate from an `f16` bit pattern.
//! NaNs created directly from an `f32` are canonicalised to a non-zero payload
//! because a 10-bit `f16` payload cannot represent every 23-bit `f32` payload.

/// Sign bit of a binary16 value.
pub const F16_SIGN: u16 = 0x8000;
/// Exponent field of a binary16 value.
pub const F16_EXPONENT_MASK: u16 = 0x7c00;
/// Mantissa field of a binary16 value.
pub const F16_MANTISSA_MASK: u16 = 0x03ff;
/// Positive infinity in binary16.
pub const F16_INFINITY: u16 = 0x7c00;

/// Convert a binary16 bit pattern to the nearest `f32` value.
///
/// This conversion is exact for every finite `f16`: every binary16 value is
/// representable as a binary32 value.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & F16_SIGN) << 16;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let mantissa = u32::from(bits & F16_MANTISSA_MASK);

    let words = if exponent == 0 {
        if mantissa == 0 {
            sign
        } else {
            // Subnormal binary16: value = mantissa * 2^-24. Normalise the
            // mantissa into a binary32 significand with an implicit leading 1.
            let top_bit = 31 - mantissa.leading_zeros();
            let f32_exponent = top_bit + 103;
            let fraction = (mantissa << (23 - top_bit)) & 0x007f_ffff;
            sign | (f32_exponent << 23) | fraction
        }
    } else if exponent == 0x1f {
        sign | 0x7f80_0000 | (mantissa << 13)
    } else {
        sign | ((exponent + 112) << 23) | (mantissa << 13)
    };

    f32::from_bits(words)
}

/// Convert an `f32` to the nearest binary16 bit pattern.
///
/// Uses round-to-nearest-even; values above the largest finite binary16
/// (65504) round to infinity; values below half of the smallest subnormal round
/// to zero. The result is deterministic for every input because it depends only
/// on the input bit pattern.
pub fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & u32::from(F16_SIGN)) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;

    if exponent == 0xff {
        // Infinity or NaN. Keep the leading payload bits so f16 -> f32 -> f16
        // is the identity for every f16 bit pattern; canonicalise f32-only
        // payloads to the smallest non-zero payload instead of returning zero
        // (a zero mantissa would turn a NaN into infinity).
        let payload = if mantissa == 0 {
            0
        } else {
            ((mantissa >> 13) as u16).max(1)
        };
        return sign | F16_EXPONENT_MASK | payload;
    }

    let unbiased = exponent - 127;

    if unbiased > 15 {
        return sign | F16_INFINITY;
    }

    if unbiased >= -14 {
        // Normal binary16. A binary32 significand has 24 bits; binary16 keeps
        // 11, so 13 bits are rounded away.
        let mut half_exponent = (unbiased + 15) as u32;
        let mut half_mantissa = mantissa >> 13;
        let round_bits = mantissa & 0x1fff;
        // round_bits > 0x1000 -> round up; == 0x1000 -> tie, round to even.
        if round_bits > 0x1000 || (round_bits == 0x1000 && half_mantissa & 1 == 1) {
            half_mantissa += 1;
            if half_mantissa == 0x400 {
                half_mantissa = 0;
                half_exponent += 1;
                if half_exponent >= 0x1f {
                    return sign | F16_INFINITY;
                }
            }
        }
        return sign | ((half_exponent as u16) << 10) | (half_mantissa as u16);
    }

    // Values below half of the smallest binary16 subnormal round to zero.
    if unbiased < -25 {
        return sign;
    }

    // Subnormal binary16: represent value = 1.mantissa * 2^unbiased as
    // q * 2^-24 and round q to nearest even.
    let significand = mantissa | 0x0080_0000;
    let shift = (-unbiased - 1) as u32;
    let mut quantized = significand >> shift;
    let remainder = significand & ((1u32 << shift) - 1);
    let halfway = 1u32 << (shift - 1);
    if remainder > halfway || (remainder == halfway && quantized & 1 == 1) {
        quantized += 1;
    }
    if quantized >= 0x0400 {
        // Rounded up into the smallest normal binary16.
        return sign | 0x0400;
    }
    sign | (quantized as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_f16_bit_pattern_round_trips_through_f32() {
        for bits in 0..=u16::MAX {
            let round_trip = f32_to_f16_bits(f16_to_f32(bits));
            assert_eq!(round_trip, bits, "bits {bits:#06x}");
        }
    }

    #[test]
    fn known_f16_values_are_exact() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert!(f16_to_f32(0x8000).is_sign_negative());
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 1.0 / 16_777_216.0);
        assert_eq!(f16_to_f32(0x0400), 1.0 / 16_384.0);
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(f16_to_f32(0xfc00), f32::NEG_INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn known_f32_values_round_correctly() {
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        assert_eq!(f32_to_f16_bits(-0.0), 0x8000);
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(-2.0), 0xc000);
        assert_eq!(f32_to_f16_bits(65504.0), 0x7bff);
        assert_eq!(f32_to_f16_bits(65520.0), F16_INFINITY);
        assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_f16_bits(1.0 / 33_554_432.0), 0x0000);
        assert_eq!(f32_to_f16_bits(3.0 / 33_554_432.0), 0x0002);
    }

    #[test]
    fn ties_round_to_even() {
        // Exactly halfway between 1.0 (mantissa 0) and 1 + 2^-10 (mantissa 1).
        assert_eq!(f32_to_f16_bits(1.0 + 1.0 / 2048.0), 0x3c00);
        // Exactly halfway between 1 + 2^-10 and 1 + 2 * 2^-10.
        assert_eq!(f32_to_f16_bits(1.0 + 3.0 / 2048.0), 0x3c02);
    }

    #[test]
    fn nan_payload_is_preserved_for_f16_originated_values() {
        for payload in 1..=0x03ffu16 {
            let bits = 0x7c00 | payload;
            assert_eq!(f32_to_f16_bits(f16_to_f32(bits)), bits);
            let negative = 0xfc00 | payload;
            assert_eq!(f32_to_f16_bits(f16_to_f32(negative)), negative);
        }
    }
}
