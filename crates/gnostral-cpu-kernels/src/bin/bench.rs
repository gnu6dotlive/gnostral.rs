//! Deterministic, dependency-free throughput harness for the Q8_0 kernels.
//!
//! ```text
//! cargo run --offline --release --bin bench -- [--rows R] [--k K] [--n N]
//!                                             [--iters I] [--warmup W]
//! ```
//!
//! **INDICATIVE ONLY.** This is a local microbenchmark. It has not been
//! verified against `ggml`, no frozen receipt exists for it, and its numbers
//! must not be quoted as a speedup or as a cross-engine comparison. It exists
//! so a later, properly frozen run can reuse the same shapes and counters.
//!
//! Inputs are generated from a fixed xorshift seed, so the printed checksum is
//! identical on every run of the same binary and shape. The harness reports:
//!
//! - Gop/s, counting 2 operations per multiply-add;
//! - GB/s over the compulsory bytes of one call (every block read once, output
//!   written once), which is an upper bound on the attainable bandwidth, not a
//!   cache-traffic measurement.

use std::hint::black_box;
use std::time::Instant;

use gnostral_cpu_kernels::{
    avx2_available, dot_q8_0, gemm_q8_0, quantize_row_q8_0, KernelError, MAX_SAFE_K, QK8_0,
};

const BANNER: &str = "\
============================================================
INDICATIVE ONLY - local CPU microbenchmark, no frozen receipt.
Not verified against ggml or any other engine. Do not cite
these numbers as a speedup.
============================================================";

struct Config {
    rows: usize,
    k: usize,
    n: usize,
    iterations: usize,
    warmup: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            rows: 64,
            k: 1024,
            n: 1024,
            iterations: 20,
            warmup: 3,
        }
    }
}

impl Config {
    fn from_args() -> Result<Self, String> {
        let mut config = Self::default();
        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--rows" => config.rows = parse_usize(&mut args, &flag)?,
                "--k" => config.k = parse_usize(&mut args, &flag)?,
                "--n" => config.n = parse_usize(&mut args, &flag)?,
                "--iters" => config.iterations = parse_usize(&mut args, &flag)?,
                "--warmup" => config.warmup = parse_usize(&mut args, &flag)?,
                "--help" | "-h" => {
                    print_usage();
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument `{other}`")),
            }
        }
        Ok(config)
    }

    fn validate(&self) -> Result<(), String> {
        if self.rows == 0 {
            return Err("--rows must be at least 1".to_owned());
        }
        if !self.k.is_multiple_of(QK8_0) || !self.n.is_multiple_of(QK8_0) {
            return Err(format!(
                "--k and --n must be multiples of {QK8_0} (the Q8_0 block length)"
            ));
        }
        if self.k == 0 || self.n == 0 {
            return Err("--k and --n must be at least 32".to_owned());
        }
        if self.k > MAX_SAFE_K {
            return Err(format!(
                "--k must be at most {MAX_SAFE_K} (i32 accumulation budget)"
            ));
        }
        if self.iterations == 0 {
            return Err("--iters must be at least 1".to_owned());
        }
        Ok(())
    }
}

fn parse_usize(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<usize, String> {
    let text = args
        .next()
        .ok_or_else(|| format!("missing value for `{flag}`"))?;
    text.parse::<usize>()
        .map_err(|_| format!("invalid value `{text}` for `{flag}`"))
}

fn print_usage() {
    println!(
        "usage: bench [--rows R] [--k K] [--n N] [--iters I] [--warmup W]\n\
         defaults: rows=64 k=1024 n=1024 iters=20 warmup=3\n\
         K and N must be multiples of 32; K <= {MAX_SAFE_K}."
    );
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32) / 16_777_216.0
    }

    fn fill(&mut self, values: &mut [f32]) {
        for value in values {
            *value = self.unit() * 2.0 - 1.0;
        }
    }
}

fn main() {
    let config = match Config::from_args() {
        Ok(config) => config,
        Err(error) => {
            eprintln!("error: {error}");
            print_usage();
            std::process::exit(2);
        }
    };
    if let Err(error) = run(&config) {
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}

fn describe(error: KernelError) -> String {
    format!("kernel rejected the benchmark shape: {error:?}")
}

fn time_per_call(warmup: usize, iterations: usize, mut operation: impl FnMut()) -> f64 {
    for _ in 0..warmup {
        operation();
    }
    let start = Instant::now();
    for _ in 0..iterations {
        operation();
    }
    start.elapsed().as_secs_f64() / iterations as f64
}

fn run(config: &Config) -> Result<(), String> {
    config.validate()?;
    let rows = config.rows;
    let k = config.k;
    let n = config.n;
    let k_blocks = k / QK8_0;

    let mut rng = Rng::new(0x0d15_ea5e_beef_0001);
    let mut activation = vec![0.0f32; rows * k];
    let mut weights = vec![0.0f32; n * k];
    rng.fill(&mut activation);
    rng.fill(&mut weights);

    let activation_blocks = quantize_row_q8_0(&activation).map_err(describe)?;
    let weight_blocks = quantize_row_q8_0(&weights).map_err(describe)?;

    let dot_seconds = time_per_call(config.warmup, config.iterations, || {
        let value = dot_q8_0(&activation_blocks[..k_blocks], &weight_blocks[..k_blocks])
            .expect("validated shape");
        black_box(value);
    });

    let mut out = vec![0.0f32; rows * n];
    let gemm_seconds = time_per_call(config.warmup, config.iterations, || {
        gemm_q8_0(&activation_blocks, rows, k, &weight_blocks, n, &mut out)
            .expect("validated shape");
        black_box(&out);
    });

    let dot_ops = 2.0 * k as f64;
    let dot_bytes = 2.0 * k_blocks as f64 * 34.0;
    let gemm_ops = 2.0 * rows as f64 * k as f64 * n as f64;
    let gemm_bytes =
        (activation_blocks.len() + weight_blocks.len()) as f64 * 34.0 + (rows * n) as f64 * 4.0;

    println!("gnostral-cpu-kernels {}", env!("CARGO_PKG_VERSION"));
    println!("{BANNER}");
    println!(
        "host arch={} avx2={}",
        std::env::consts::ARCH,
        if avx2_available() { "yes" } else { "no" }
    );
    println!(
        "shape rows={rows} K={k} N={n} (blocks A={} B={}) warmup={} iterations={}",
        activation_blocks.len(),
        weight_blocks.len(),
        config.warmup,
        config.iterations
    );
    println!(
        "dot      {k} values/call   {:>9.3} Gop/s   {:>8.3} GB/s",
        dot_ops / dot_seconds / 1.0e9,
        dot_bytes / dot_seconds / 1.0e9
    );
    println!(
        "gemm     {rows}x{k}x{n}   {:>9.3} Gop/s   {:>8.3} GB/s",
        gemm_ops / gemm_seconds / 1.0e9,
        gemm_bytes / gemm_seconds / 1.0e9
    );
    let checksum: f64 = out.iter().map(|&value| f64::from(value)).sum();
    println!("checksum {checksum:.6} (deterministic for a fixed shape)");
    println!("GB/s counts compulsory bytes only; it is not a cache-traffic measurement.");
    Ok(())
}
