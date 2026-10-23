//! Batched and grouped K-quant expert kernels.
//!
//! - [`gemm_q8k`]: `out[m, n] = act[m, k] x W[n, k]^T` with `Q8_K` activations
//!   and K-quant weights stored output-major (GGML layout).
//! - [`grouped_expert_ffn`]: the full gated expert MLP
//!   `down(act(gate(x)) * up(x))` for **every expert group of one layer**, with
//!   one pool dispatch per projection instead of one per route.
//!
//! # Determinism
//!
//! Every output element is produced by one call to [`KQuantBlock::dot_with`] on one
//! activation row and one weight row; activation rows are quantised
//! independently. An element therefore does not depend on the thread count, on
//! which other rows share its group, or on how groups are batched together.
//! The tests assert bit equality for all three.

use crate::kquant::{quantize_row_q8k, BlockQ8K, DotKernel, Isa, KQuantBlock, QK_K};
use crate::pool::{Pool, SharedOut};
use crate::KernelError;

/// Output columns per task. Each task keeps one weight row hot across all `m` activation rows.
const COLUMN_TILE: usize = 8;

fn tiles(n: usize) -> usize {
    n.div_ceil(COLUMN_TILE)
}

fn check_len(buffer: &'static str, expected: usize, actual: usize) -> Result<(), KernelError> {
    if expected != actual {
        return Err(KernelError::BufferSize {
            buffer,
            expected,
            actual,
        });
    }
    Ok(())
}

fn check_k(k: usize) -> Result<usize, KernelError> {
    if !k.is_multiple_of(QK_K) || k == 0 {
        return Err(KernelError::RowNotBlockAligned {
            len: k,
            block: QK_K,
        });
    }
    Ok(k / QK_K)
}

fn check_isa(isa: Isa) -> Result<(), KernelError> {
    if !isa.available() {
        return Err(KernelError::IsaUnavailable { isa: isa.name() });
    }
    Ok(())
}

/// `[m, k] x [k, n]` GEMM dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmShape {
    pub m: usize,
    pub k: usize,
    pub n: usize,
}

/// `out[m, n] = act[m, k] x w[n, k]^T`.
///
/// `act` holds `m * k / 256` blocks, row-major; `w` holds `n * k / 256` blocks,
/// one row per output column; `out` holds `m * n` values.
pub fn gemm_q8k<W: KQuantBlock>(
    kernel: impl Into<DotKernel>,
    pool: &Pool,
    shape: GemmShape,
    act: &[BlockQ8K],
    w: &[W],
    out: &mut [f32],
) -> Result<(), KernelError> {
    let GemmShape { m, k, n } = shape;
    let kernel = kernel.into();
    check_isa(kernel.isa)?;
    let kb = check_k(k)?;
    check_len(
        "act",
        m.checked_mul(kb).ok_or(KernelError::ShapeOverflow)?,
        act.len(),
    )?;
    check_len(
        "w",
        n.checked_mul(kb).ok_or(KernelError::ShapeOverflow)?,
        w.len(),
    )?;
    check_len(
        "out",
        m.checked_mul(n).ok_or(KernelError::ShapeOverflow)?,
        out.len(),
    )?;
    let dst = SharedOut::new(out);
    pool.run(tiles(n), |tile| {
        let start = tile * COLUMN_TILE;
        for col in start..(start + COLUMN_TILE).min(n) {
            let row_w = &w[col * kb..(col + 1) * kb];
            for row in 0..m {
                let value = W::dot_with(kernel, row_w, &act[row * kb..(row + 1) * kb]);
                // SAFETY: (row, col) is in bounds and owned by this tile only.
                unsafe { dst.write(row * n + col, value) };
            }
        }
    });
    Ok(())
}

/// Quantise `m` rows of `k` values into `Q8_K`.
pub fn quantize_rows_q8k(x: &[f32], m: usize, k: usize) -> Result<Vec<BlockQ8K>, KernelError> {
    let kb = check_k(k)?;
    check_len(
        "x",
        m.checked_mul(k).ok_or(KernelError::ShapeOverflow)?,
        x.len(),
    )?;
    let mut q = vec![BlockQ8K::ZERO; m * kb];
    quantize_row_q8k(x, &mut q)?;
    Ok(q)
}

/// One expert's weights and the token rows routed to it.
pub struct ExpertGroup<'a, G, D> {
    /// `[inter, hidden]` gate projection, output-major.
    pub gate: &'a [G],
    /// `[inter, hidden]` up projection, output-major.
    pub up: &'a [G],
    /// `[hidden, inter]` down projection, output-major.
    pub down: &'a [D],
    /// `[rows, hidden]` input rows, row-major.
    pub x: &'a [f32],
    pub rows: usize,
}

/// Shape and activation shared by every group in a layer.
#[derive(Clone, Copy)]
pub struct FfnShape {
    pub hidden: usize,
    pub inter: usize,
}

struct TileMap {
    starts: Vec<usize>,
    total: usize,
}

impl TileMap {
    fn new(per_group: impl Iterator<Item = usize>) -> Self {
        let mut starts = Vec::new();
        let mut total = 0;
        for tiles in per_group {
            starts.push(total);
            total += tiles;
        }
        Self { starts, total }
    }
    fn locate(&self, task: usize) -> (usize, usize) {
        let group = self.starts.partition_point(|&s| s <= task) - 1;
        (group, task - self.starts[group])
    }
}

/// Gated expert MLP for every group of one layer.
///
/// Returns one `[rows, hidden]` output per group, in group order. Per group:
/// `x` is quantised to `Q8_K` once and shared by gate and up; `act(gate) * up`
/// is computed in `f32` and quantised once for down. Each of the three phases
/// is a single pool dispatch over every group's column tiles.
pub fn grouped_expert_ffn<G: KQuantBlock, D: KQuantBlock>(
    kernel: impl Into<DotKernel>,
    pool: &Pool,
    shape: FfnShape,
    groups: &[ExpertGroup<'_, G, D>],
    act: impl Fn(f32) -> f32 + Sync,
) -> Result<Vec<Vec<f32>>, KernelError> {
    let kernel = kernel.into();
    check_isa(kernel.isa)?;
    let FfnShape { hidden, inter } = shape;
    let hb = check_k(hidden)?;
    let ib = check_k(inter)?;
    for g in groups {
        check_len("gate", inter * hb, g.gate.len())?;
        check_len("up", inter * hb, g.up.len())?;
        check_len("down", hidden * ib, g.down.len())?;
        check_len(
            "x",
            g.rows
                .checked_mul(hidden)
                .ok_or(KernelError::ShapeOverflow)?,
            g.x.len(),
        )?;
        if let Some(index) = g.x.iter().position(|v| !v.is_finite()) {
            return Err(KernelError::NonFiniteInput { index });
        }
    }

    let xq: Vec<Vec<BlockQ8K>> = groups
        .iter()
        .map(|g| quantize_rows_q8k(g.x, g.rows, hidden))
        .collect::<Result<_, _>>()?;

    let mut gate_out: Vec<Vec<f32>> = groups.iter().map(|g| vec![0f32; g.rows * inter]).collect();
    let mut up_out: Vec<Vec<f32>> = groups.iter().map(|g| vec![0f32; g.rows * inter]).collect();
    {
        let map = TileMap::new(groups.iter().map(|_| tiles(inter)));
        let gate_dst: Vec<SharedOut> = gate_out.iter_mut().map(|v| SharedOut::new(v)).collect();
        let up_dst: Vec<SharedOut> = up_out.iter_mut().map(|v| SharedOut::new(v)).collect();
        pool.run(map.total, |task| {
            let (gi, tile) = map.locate(task);
            let g = &groups[gi];
            let start = tile * COLUMN_TILE;
            for col in start..(start + COLUMN_TILE).min(inter) {
                let wg = &g.gate[col * hb..(col + 1) * hb];
                let wu = &g.up[col * hb..(col + 1) * hb];
                for row in 0..g.rows {
                    let a = &xq[gi][row * hb..(row + 1) * hb];
                    // SAFETY: (row, col) of group gi is owned by this tile only.
                    unsafe {
                        gate_dst[gi].write(row * inter + col, G::dot_with(kernel, wg, a));
                        up_dst[gi].write(row * inter + col, G::dot_with(kernel, wu, a));
                    }
                }
            }
        });
    }

    let mut hq: Vec<Vec<BlockQ8K>> = Vec::with_capacity(groups.len());
    for (gate, up) in gate_out.iter_mut().zip(&up_out) {
        for (g, &u) in gate.iter_mut().zip(up) {
            *g = act(*g) * u;
        }
        let rows = gate.len() / inter;
        hq.push(quantize_rows_q8k(gate, rows, inter)?);
    }

    let mut outputs: Vec<Vec<f32>> = groups.iter().map(|g| vec![0f32; g.rows * hidden]).collect();
    {
        let map = TileMap::new(groups.iter().map(|_| tiles(hidden)));
        let dst: Vec<SharedOut> = outputs.iter_mut().map(|v| SharedOut::new(v)).collect();
        pool.run(map.total, |task| {
            let (gi, tile) = map.locate(task);
            let g = &groups[gi];
            let start = tile * COLUMN_TILE;
            for col in start..(start + COLUMN_TILE).min(hidden) {
                let wd = &g.down[col * ib..(col + 1) * ib];
                for row in 0..g.rows {
                    let a = &hq[gi][row * ib..(row + 1) * ib];
                    // SAFETY: (row, col) of group gi is owned by this tile only.
                    unsafe { dst[gi].write(row * hidden + col, D::dot_with(kernel, wd, a)) };
                }
            }
        });
    }
    Ok(outputs)
}

/// SiLU as computed by the vendored Candle CPU op: `x / (1 + exp(-x))`.
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
