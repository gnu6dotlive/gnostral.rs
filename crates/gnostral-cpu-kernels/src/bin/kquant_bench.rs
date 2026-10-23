//! Grouped K-quant expert FFN microbenchmark (Q-029 definition of done, item 2).
//!
//! ```text
//! cargo run --release --bin kquant_bench -- [--hidden H] [--inter I] [--groups G]
//!     [--rows R] [--warmup W] [--reps N] [--threads T] [--isa scalar|avx2|avx512]
//! ```
//!
//! Defaults follow the Qwen3-30B-A3B expert shape (hidden 2048, inter 768) with
//! 8 groups of 1 row (single-token decode, top-8 routing). Weights are synthetic
//! Q2_K gate/up and Q3_K down blocks from a fixed seed. Output is one JSON
//! object with the configuration, ISA, thread count, **every** sample and the
//! median, so a run can be frozen into a receipt.
//!
//! INDICATIVE ONLY: a synthetic-block microbenchmark is not engine throughput
//! and must never be reported as tok/s for a model.

use std::hint::black_box;
use std::time::Instant;

use gnostral_cpu_kernels::kquant::{block_q2k, block_q3k};
use gnostral_cpu_kernels::{
    grouped_expert_ffn, silu, BlockQ2K, BlockQ3K, ExpertGroup, FfnShape, Isa, Pool, QK_K,
};

struct Config {
    hidden: usize,
    inter: usize,
    groups: usize,
    rows: usize,
    warmup: usize,
    reps: usize,
    threads: usize,
    isa: Isa,
}

fn parse() -> Result<Config, String> {
    let mut c = Config {
        hidden: 2048,
        inter: 768,
        groups: 8,
        rows: 1,
        warmup: 1,
        reps: 5,
        threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
        isa: Isa::detect(),
    };
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value"));
        let num = |v: String| v.parse::<usize>().map_err(|e| format!("{v}: {e}"));
        match flag.as_str() {
            "--hidden" => c.hidden = num(value()?)?,
            "--inter" => c.inter = num(value()?)?,
            "--groups" => c.groups = num(value()?)?,
            "--rows" => c.rows = num(value()?)?,
            "--warmup" => c.warmup = num(value()?)?,
            "--reps" => c.reps = num(value()?)?,
            "--threads" => c.threads = num(value()?)?,
            "--isa" => {
                c.isa = match value()?.as_str() {
                    "scalar" => Isa::Scalar,
                    "avx2" => Isa::Avx2,
                    "avx512" => Isa::Avx512,
                    other => return Err(format!("unknown isa {other}")),
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
}

fn main() {
    let c = match parse() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("kquant_bench: {e}");
            std::process::exit(2);
        }
    };
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let hb = c.hidden / QK_K;
    let ib = c.inter / QK_K;
    let experts: Vec<(Vec<BlockQ2K>, Vec<BlockQ2K>, Vec<BlockQ3K>)> = (0..c.groups)
        .map(|_| {
            let q2 = |rng: &mut Rng| block_q2k(rng.bytes(), rng.bytes(), 0.02, 0.01);
            (
                (0..c.inter * hb).map(|_| q2(&mut rng)).collect(),
                (0..c.inter * hb).map(|_| q2(&mut rng)).collect(),
                (0..c.hidden * ib)
                    .map(|_| block_q3k(rng.bytes(), rng.bytes(), rng.bytes(), 0.005))
                    .collect(),
            )
        })
        .collect();
    let inputs: Vec<Vec<f32>> = (0..c.groups)
        .map(|_| (0..c.rows * c.hidden).map(|_| rng.unit()).collect())
        .collect();
    let groups: Vec<ExpertGroup<'_, BlockQ2K, BlockQ3K>> = experts
        .iter()
        .zip(&inputs)
        .map(|((g, u, d), x)| ExpertGroup {
            gate: g,
            up: u,
            down: d,
            x,
            rows: c.rows,
        })
        .collect();
    let pool = Pool::new(c.threads);
    let shape = FfnShape {
        hidden: c.hidden,
        inter: c.inter,
    };
    let mut checksum = 0f64;
    let mut samples_ns = Vec::new();
    for i in 0..c.warmup + c.reps {
        let start = Instant::now();
        let out = grouped_expert_ffn(c.isa, &pool, shape, black_box(&groups), silu)
            .expect("valid shapes");
        let ns = start.elapsed().as_nanos() as u64;
        checksum = out.iter().flatten().map(|&v| v as f64).sum();
        if i >= c.warmup {
            samples_ns.push(ns);
        }
    }
    let mut sorted = samples_ns.clone();
    sorted.sort_unstable();
    let median = if sorted.len() % 2 == 1 {
        sorted[sorted.len() / 2] as f64
    } else {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) as f64 / 2.0
    };
    println!(
        "{{\"bench\":\"gnostral-cpu-kernels/grouped_expert_ffn\",\"indicative_only\":true,\
\"hidden\":{},\"inter\":{},\"groups\":{},\"rows_per_group\":{},\"isa\":\"{}\",\"threads\":{},\
\"warmup\":{},\"samples_ns\":{:?},\"median_ns\":{},\"checksum\":{}}}",
        c.hidden,
        c.inter,
        c.groups,
        c.rows,
        c.isa.name(),
        c.threads,
        c.warmup,
        samples_ns,
        median,
        checksum
    );
}
