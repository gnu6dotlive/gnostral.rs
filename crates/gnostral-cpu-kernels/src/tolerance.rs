//! Quantization-error bounds used by the parity tests.
//!
//! The tolerances are derived, not tuned. For one Q8_0 block with block maximum
//! `m = max |x_i|`:
//!
//! 1. The exact scale is `d_exact = m / 127`. The stored scale is
//!    `d = f16(d_exact)`, with `|d - d_exact| <= d_exact * 2^-11` for a normal
//!    binary16 scale and `<= 2^-25` for a subnormal one (binary16 spacing there
//!    is `2^-24`, and round-to-nearest gives half of that).
//! 2. `q_i = round(x_i / d)`, so `|q_i * d - x_i| <= d / 2`.
//! 3. Therefore the absolute dequantisation error of any lane satisfies
//!    `|x_hat_i - x_i| <= (d_exact + scale_error) / 2`, which is at most
//!    `m * 3.95e-3` for a normal scale.
//!
//! For a dot product `sum a_i b_i` with lane errors bounded by `e_a` and `e_b`,
//! `|a_hat_i b_hat_i - a_i b_i| <= |a_i| e_b + |b_i| e_a + e_a e_b`. Summing
//! gives [`dot_error_bound`]. That bound also reserves the `f32` accumulation
//! error of a naive reference loop (`K * 2^-24 * sum |a_i b_i|`).
//!
//! The bound is deliberately conservative. A parity test that passes it has not
//! been tuned to the observed error; a test that fails it is a real defect or an
//! input outside the documented contract.

/// Relative half-ulp of a normal binary16 value (`2^-11`).
pub const F16_REL_EPS: f64 = 1.0 / 2048.0;
/// Spacing of binary16 subnormals (`2^-24`).
pub const F16_SUBNORMAL_STEP: f64 = 1.0 / 16_777_216.0;
/// Largest magnitude representable by a Q8_0 lane as emitted by this crate.
pub const Q8_MAX_MAGNITUDE: f64 = 127.0;
/// Relative half-ulp of binary32 (`2^-24`).
pub const F32_REL_EPS: f64 = 1.0 / 16_777_216.0;
/// Boundary between normal and subnormal binary16 scales.
pub const F16_MIN_NORMAL: f64 = 1.0 / 16_384.0;

/// Upper bound on `|f16(d) - d|` for a non-negative exact scale `d`.
pub fn f16_scale_error(d_exact: f64) -> f64 {
    if d_exact < F16_MIN_NORMAL {
        // Subnormal: the rounding step is the full spacing, half of it is the
        // worst-case error.
        F16_SUBNORMAL_STEP / 2.0
    } else {
        d_exact * F16_REL_EPS
    }
}

/// Upper bound on the absolute error of one dequantised lane.
pub fn dequant_error_bound(max_abs: f64) -> f64 {
    if max_abs <= 0.0 {
        return 0.0;
    }
    let d_exact = max_abs / Q8_MAX_MAGNITUDE;
    (d_exact + f16_scale_error(d_exact)) / 2.0
}

/// Upper bound on the absolute error of `dot_q8_0` for these two rows.
///
/// Returns `f64::INFINITY` when the rows have different lengths, so a caller
/// cannot accidentally treat a shape error as a tolerance.
pub fn dot_error_bound(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    let error_a = dequant_error_bound(max_abs(a));
    let error_b = dequant_error_bound(max_abs(b));
    let sum_a: f64 = a.iter().map(|value| f64::from(value.abs())).sum();
    let sum_b: f64 = b.iter().map(|value| f64::from(value.abs())).sum();
    let sum_products: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(left, right)| f64::from(left.abs() * right.abs()))
        .sum();
    let k = a.len() as f64;

    error_a * sum_b + error_b * sum_a + k * error_a * error_b + k * F32_REL_EPS * sum_products
}

fn max_abs(values: &[f32]) -> f64 {
    values
        .iter()
        .fold(0.0f64, |current, value| current.max(f64::from(value.abs())))
}
