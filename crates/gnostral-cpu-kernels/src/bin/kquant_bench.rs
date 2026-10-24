//! K-quant kernel microbenchmarks (Q-029; Phase 2 instrumentation, docs/FINDINGS.md).
//!
//! ```text
//! cargo run --release --bin kquant_bench -- [--mode ffn|dispatch|dot]
//!     [--hidden H] [--inter I] [--groups G] [--rows R] [--pool-experts E]
//!     [--warmup W] [--reps N] [--threads T] [--isa scalar|avx2|avx512]
//!     [--numerics native|candle] [--seed S]
//! ```
//!
//! Modes, each printing one JSON object with every sample:
//!
//! - `ffn` (default): `grouped_expert_ffn` for `G` experts of `R` rows each. Defaults follow
//!   Qwen3-30B-A3B single-token decode (hidden 2048, inter 768, 8 experts of 1 row). With
//!   `--pool-experts 0` the same `G` experts run every iteration, as llama.cpp's
//!   `test-backend-ops` does (weights stay hot in cache); with `--pool-experts E` each
//!   iteration draws `G` distinct experts from `E`, so weights stream from memory as in real
//!   decode.
//! - `dispatch`: `Pool::run` over trivial tasks, the fixed cost of waking the pool once.
//! - `dot`: single-thread `Q2_K` and `Q3_K` row dot products, nanoseconds per 256-value block.
//!
//! INDICATIVE ONLY: a synthetic-block microbenchmark is not engine throughput and must never
//! be reported as tok/s for a model.

use std::hint::black_box;
use std::time::Instant;

use gnostral_cpu_kernels::kquant::{block_q2k, block_q3k, quantize_row_q8k};
use gnostral_cpu_kernels::{
    grouped_expert_ffn, silu, BlockQ2K, BlockQ3K, BlockQ8K, DotKernel, ExpertGroup, FfnShape, Isa,
    KQuantBlock, Numerics, Pool, QK_K,
};

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Ffn,
    Dispatch,
    Dot,
}

struct Config {
    mode: Mode,
    hidden: usize,
    inter: usize,
    groups: usize,
    rows: usize,
    pool_experts: usize,
    warmup: usize,
    reps: usize,
    threads: usize,
    isa: Isa,
    numerics: Numerics,
    seed: u64,
}

fn parse() -> Result<Config, String> {
    let mut c = Config {
        mode: Mode::Ffn,
        hidden: 2048,
        inter: 768,
        groups: 8,
        rows: 1,
        pool_experts: 0,
        warmup: 1,
        reps: 5,
        threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
        isa: Isa::detect(),
        numerics: Numerics::Native,
        seed: 0x9e37_79b9_7f4a_7c15,
    };
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value"));
        let num = |v: String| v.parse::<usize>().map_err(|e| format!("{v}: {e}"));
        match flag.as_str() {
            "--mode" => {
                c.mode = match value()?.as_str() {
                    "ffn" => Mode::Ffn,
                    "dispatch" => Mode::Dispatch,
                    "dot" => Mode::Dot,
                    other => return Err(format!("unknown mode {other}")),
                }
            }
            "--hidden" => c.hidden = num(value()?)?,
            "--inter" => c.inter = num(value()?)?,
            "--groups" => c.groups = num(value()?)?,
            "--rows" => c.rows = num(value()?)?,
            "--pool-experts" => c.pool_experts = num(value()?)?,
            "--warmup" => c.warmup = num(value()?)?,
            "--reps" => c.reps = num(value()?)?,
            "--threads" => c.threads = num(value()?)?,
            "--seed" => c.seed = num(value()?)? as u64,
            "--isa" => {
                c.isa = match value()?.as_str() {
                    "scalar" => Isa::Scalar,
                    "avx2" => Isa::Avx2,
                    "avx512" => Isa::Avx512,
                    other => return Err(format!("unknown isa {other}")),
                }
            }
            "--numerics" => {
                c.numerics = match value()?.as_str() {
                    "native" => Numerics::Native,
                    "candle" => Numerics::CandleCompat,
                    other => return Err(format!("unknown numerics {other}")),
                }
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if c.reps < 3 || c.warmup < 1 {
        return Err("the protocol needs --warmup >= 1 and --reps >= 3".into());
    }
    if !c.hidden.is_multiple_of(QK_K)
        || !c.inter.is_multiple_of(QK_K)
        || c.groups == 0
        || c.rows == 0
    {
        return Err("hidden and inter must be multiples of 256; groups and rows positive".into());
    }
    if c.pool_experts != 0 && c.pool_experts < c.groups {
        return Err("--pool-experts must be 0 or at least --groups".into());
    }
    if !c.isa.available() {
        return Err(format!("{} is not available on this CPU", c.isa.name()));
    }
    Ok(c)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn bytes<const N: usize>(&mut self) -> [u8; N] {
        let mut b = [0u8; N];
        b.iter_mut().for_each(|x| *x = (self.next() >> 24) as u8);
        b
    }
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

type Expert = (Vec<BlockQ2K>, Vec<BlockQ2K>, Vec<BlockQ3K>);

fn expert(rng: &mut Rng, hidden: usize, inter: usize) -> Expert {
    let (hb, ib) = (hidden / QK_K, inter / QK_K);
    let q2 = |rng: &mut Rng| block_q2k(rng.bytes(), rng.bytes(), 0.02, 0.01);
    (
        (0..inter * hb).map(|_| q2(rng)).collect(),
        (0..inter * hb).map(|_| q2(rng)).collect(),
        (0..hidden * ib)
            .map(|_| block_q3k(rng.bytes(), rng.bytes(), rng.bytes(), 0.005))
            .collect(),
    )
}

fn median(samples: &[u64]) -> f64 {
    let mut s = samples.to_vec();
    s.sort_unstable();
    if s.len() % 2 == 1 {
        s[s.len() / 2] as f64
    } else {
        (s[s.len() / 2 - 1] + s[s.len() / 2]) as f64 / 2.0
    }
}

fn kernel_name(c: &Config) -> &'static str {
    match c.numerics {
        Numerics::Native => "native",
        Numerics::CandleCompat => "candle",
    }
}

fn bench_ffn(c: &Config, rng: &mut Rng) -> String {
    let n_experts = if c.pool_experts == 0 {
        c.groups
    } else {
        c.pool_experts
    };
    let experts: Vec<Expert> = (0..n_experts)
        .map(|_| expert(rng, c.hidden, c.inter))
        .collect();
    let inputs: Vec<Vec<f32>> = (0..c.groups)
        .map(|_| (0..c.rows * c.hidden).map(|_| rng.unit()).collect())
        .collect();
    let pool = Pool::new(c.threads);
    let shape = FfnShape {
        hidden: c.hidden,
        inter: c.inter,
    };
    let kernel = DotKernel {
        isa: c.isa,
        numerics: c.numerics,
    };
    let mut checksum = 0f64;
    let mut samples_ns = Vec::new();
    for i in 0..c.warmup + c.reps {
        // Draw distinct experts for this iteration (a fixed set when the pool equals the groups).
        let mut chosen: Vec<usize> = Vec::with_capacity(c.groups);
        while chosen.len() < c.groups {
            let e = if c.pool_experts == 0 {
                chosen.len()
            } else {
                rng.below(n_experts)
            };
            if !chosen.contains(&e) {
                chosen.push(e);
            }
        }
        let groups: Vec<ExpertGroup<'_, BlockQ2K, BlockQ3K>> = chosen
            .iter()
            .zip(&inputs)
            .map(|(&e, x)| ExpertGroup {
                gate: &experts[e].0,
                up: &experts[e].1,
                down: &experts[e].2,
                x,
                rows: c.rows,
            })
            .collect();
        let start = Instant::now();
        let out = grouped_expert_ffn(kernel, &pool, shape, black_box(&groups), silu)
            .expect("valid shapes");
        let ns = start.elapsed().as_nanos() as u64;
        checksum = out.iter().flatten().map(|&v| v as f64).sum();
        if i >= c.warmup {
            samples_ns.push(ns);
        }
    }
    format!(
        "{{\"bench\":\"grouped_expert_ffn\",\"indicative_only\":true,\"hidden\":{},\"inter\":{},\
\"groups\":{},\"rows_per_group\":{},\"pool_experts\":{},\"isa\":\"{}\",\"numerics\":\"{}\",\
\"threads\":{},\"warmup\":{},\"samples_ns\":{:?},\"median_ns\":{},\"checksum\":{}}}",
        c.hidden,
        c.inter,
        c.groups,
        c.rows,
        c.pool_experts,
        c.isa.name(),
        kernel_name(c),
        c.threads,
        c.warmup,
        samples_ns,
        median(&samples_ns),
        checksum
    )
}

fn bench_dispatch(c: &Config) -> String {
    const DISPATCHES_PER_SAMPLE: u64 = 1000;
    let pool = Pool::new(c.threads);
    let tasks = c.threads * 4;
    let mut samples_ns = Vec::new();
    for i in 0..c.warmup + c.reps {
        let start = Instant::now();
        for _ in 0..DISPATCHES_PER_SAMPLE {
            pool.run(tasks, |t| {
                black_box(t);
            });
        }
        let ns = start.elapsed().as_nanos() as u64 / DISPATCHES_PER_SAMPLE;
        if i >= c.warmup {
            samples_ns.push(ns);
        }
    }
    format!(
        "{{\"bench\":\"pool_dispatch\",\"indicative_only\":true,\"threads\":{},\"tasks\":{},\
\"samples_ns_per_dispatch\":{:?},\"median_ns_per_dispatch\":{}}}",
        c.threads,
        tasks,
        samples_ns,
        median(&samples_ns)
    )
}

fn dot_ns_per_block<W: KQuantBlock>(c: &Config, rows: &[Vec<W>], act: &[BlockQ8K]) -> Vec<u64> {
    let kernel = DotKernel {
        isa: c.isa,
        numerics: c.numerics,
    };
    let blocks = (rows.len() * act.len()) as u64;
    let mut samples = Vec::new();
    for i in 0..c.warmup + c.reps {
        let start = Instant::now();
        let mut acc = 0f32;
        for row in rows {
            acc += W::dot_with(kernel, black_box(row), black_box(act));
        }
        black_box(acc);
        let ns = start.elapsed().as_nanos() as u64;
        if i >= c.warmup {
            samples.push(ns * 1000 / blocks);
        }
    }
    samples
}

fn bench_dot(c: &Config, rng: &mut Rng) -> String {
    let e = expert(rng, c.hidden, c.inter);
    let (hb, ib) = (c.hidden / QK_K, c.inter / QK_K);
    let quantized = |rng: &mut Rng, len: usize| {
        let x: Vec<f32> = (0..len).map(|_| rng.unit()).collect();
        let mut q = vec![BlockQ8K::ZERO; len / QK_K];
        quantize_row_q8k(&x, &mut q).expect("aligned");
        q
    };
    let act_h = quantized(rng, c.hidden);
    let act_i = quantized(rng, c.inter);
    let q2_rows: Vec<Vec<BlockQ2K>> = e.0.chunks(hb).map(<[_]>::to_vec).collect();
    let q3_rows: Vec<Vec<BlockQ3K>> = e.2.chunks(ib).map(<[_]>::to_vec).collect();
    let q2 = dot_ns_per_block(c, &q2_rows, &act_h);
    let q3 = dot_ns_per_block(c, &q3_rows, &act_i);
    format!(
        "{{\"bench\":\"row_dot\",\"indicative_only\":true,\"isa\":\"{}\",\"numerics\":\"{}\",\
\"q2k_rows\":{},\"q2k_blocks_per_row\":{},\"q2k_ps_per_block\":{:?},\"q2k_median_ps_per_block\":{},\
\"q3k_rows\":{},\"q3k_blocks_per_row\":{},\"q3k_ps_per_block\":{:?},\"q3k_median_ps_per_block\":{}}}",
        c.isa.name(),
        kernel_name(c),
        q2_rows.len(),
        hb,
        q2,
        median(&q2),
        q3_rows.len(),
        ib,
        q3,
        median(&q3)
    )
}

fn main() {
    let c = match parse() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("kquant_bench: {e}");
            std::process::exit(2);
        }
    };
    let mut rng = Rng(c.seed);
    let line = match c.mode {
        Mode::Ffn => bench_ffn(&c, &mut rng),
        Mode::Dispatch => bench_dispatch(&c),
        Mode::Dot => bench_dot(&c, &mut rng),
    };
    println!("{line}");
}
