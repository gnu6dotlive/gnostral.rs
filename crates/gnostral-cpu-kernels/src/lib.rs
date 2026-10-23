//! Reference Q8_0 CPU kernels for the gnostral.rs engine (quest `Q-029`,
//! deliverable `CPU-KERNEL-Q8-01`).
//!
//! # What this crate is
//!
//! A small, dependency-free, CPU-only implementation of the GGML `Q8_0` block
//! format and two reference kernels over it:
//!
//! - [`quantize_row_q8_0`] / [`dequantize_row_q8_0`]: 32 `f32` values become
//!   32 `i8` values plus one `f16` scale (`d = max_abs / 127`), and back.
//! - [`dot_q8_0`]: scalar reference dot product with `i32` block accumulation,
//!   plus an AVX2 path selected at runtime when the CPU supports it.
//! - [`gemm_q8_0`]: blocked `[rows, K] x [K, N]` GEMM. `K` and `N` must be
//!   multiples of 32 and `K` must stay within the documented `i32` accumulation
//!   budget; both constraints are enforced with [`KernelError`]. The weight
//!   operand is stored output-major (`[N, K]`, as GGML stores weights).
//! - [`gate_up_row_q8_0`]: a fused gate/up scheduling helper that reads one
//!   activation row once and produces two projection blocks.
//!
//! And the K-quant expert path the engine actually needs (`Q2_K`/`Q3_K`
//! weights against `Q8_K` activations, GGML block layout):
//!
//! - [`kquant`]: blocks, `Q8_K` quantisation, dequantisation and dot products
//!   with scalar, AVX2 and AVX-512BW integer cores that are bit-identical.
//! - [`gemm_q8k`] / [`grouped_expert_ffn`]: batched GEMM and the full gated
//!   expert MLP for all expert groups of a layer on a persistent [`Pool`];
//!   results are invariant to thread count and grouping.
//!
//! The scalar path is always compiled and always tested. The AVX2 path is
//! compiled on `x86_64`, selected only after runtime feature detection, and is
//! covered by parity tests against the scalar path. It is a reference kernel,
//! not a tuned one.
//!
//! # Claim status
//!
//! **Indicative, local, unverified against ggml.** No speedup is claimed
//! anywhere in this crate. `src/bin/bench.rs` prints throughput numbers so a
//! later run can be frozen into a receipt; until such a receipt exists, every
//! number it prints is indicative only and must not be quoted as a comparison
//! with `ggml`, `llama.cpp` or any engine.
//!
//! # Layout and constraints
//!
//! One [`BlockQ8_0`] is 32 `i8` quantised values and one `f16` scale: 34 bytes,
//! the same layout as GGML's `block_q8_0` (`ggml_half d; int8_t qs[QK8_0]`).
//! [`BlockQ8_0::to_bytes`] and [`BlockQ8_0::from_bytes`] serialise that layout
//! explicitly in little-endian order; the in-memory struct is `#[repr(C)]`.
//!
//! The quantiser emits `q` in `[-127, 127]` (GGML reserves `-128` in practice).
//! Dequantisation and both kernels accept a foreign `-128` and interpret it as
//! `-128 * d`; the AVX2 fast path falls back to scalar for such a block because
//! `pabsb` saturates on `-128`.
//!
//! # Error model
//!
//! Every shape or input defect is a typed [`KernelError`]; the crate does not
//! panic on caller data and does not silently truncate a row.

#![deny(unsafe_op_in_unsafe_fn)]

mod block;
mod f16;
mod grouped;
mod kernel;
pub mod kquant;
mod kquant_simd;
mod pool;
mod tolerance;

pub use block::{dequantize_row_q8_0, quantize_row_q8_0, BlockQ8_0, QK8_0};
pub use f16::{f16_to_f32, f32_to_f16_bits};
pub use grouped::{
    gemm_q8k, grouped_expert_ffn, quantize_rows_q8k, silu, ExpertGroup, FfnShape, GemmShape,
};
pub use kernel::{
    avx2_available, dot_block_q8_0_i32, dot_block_q8_0_i32_scalar, dot_q8_0, dot_q8_0_scalar,
    gate_up_row_q8_0, gemm_q8_0, gemm_q8_0_scalar, MAX_SAFE_K,
};
pub use kquant::{BlockQ2K, BlockQ3K, BlockQ8K, Isa, KQuantBlock, QK_K};
pub use pool::{Pool, THREADS_ENV};
pub use tolerance::{dequant_error_bound, dot_error_bound, F16_REL_EPS, F16_SUBNORMAL_STEP};

/// A rejected Q8_0 operation.
///
/// Errors carry the offending dimensions so a caller can fix the shape instead
/// of guessing. No variant is returned for a valid kernel invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelError {
    /// A row length is not a whole number of 32-value Q8_0 blocks.
    RowNotBlockAligned { len: usize, block: usize },
    /// A quantiser input contains `NaN` or an infinity.
    NonFiniteInput { index: usize },
    /// The two dot-product operands cover a different number of blocks.
    DotLengthMismatch { a_blocks: usize, b_blocks: usize },
    /// `K` exceeds the largest length for which `i32` block accumulation is
    /// guaranteed not to overflow (see [`MAX_SAFE_K`]).
    AccumulationOverflowRisk { k: usize, max_k: usize },
    /// A GEMM dimension is not a multiple of the 32-value Q8_0 block.
    GemmShapeNotBlockAligned { k: usize, n: usize, block: usize },
    /// The computed buffer size does not fit in `usize`.
    ShapeOverflow,
    /// A buffer's length does not match the declared shape.
    BufferSize {
        buffer: &'static str,
        expected: usize,
        actual: usize,
    },
    /// The requested instruction set is not available on this CPU.
    IsaUnavailable { isa: &'static str },
}
