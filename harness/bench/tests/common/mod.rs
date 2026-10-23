//! Synthetic receipt fixtures. These are test documents, not measurements.

#![allow(dead_code)]

use std::sync::atomic::{AtomicU64, Ordering};

pub const BIN_A: &str = "1111111111111111111111111111111111111111111111111111111111111111";
pub const BIN_B: &str = "2222222222222222222222222222222222222222222222222222222222222222";
pub const BIN_C: &str = "3333333333333333333333333333333333333333333333333333333333333333";
pub const MODEL: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
pub const MODEL_ALT: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
pub const PROMPT: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
pub const PROTOCOL: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

pub struct SampleFixture {
    pub decode: f64,
    /// Raw token for the sample's decode value, when a test needs a value the
    /// fixture arithmetic would not produce (for example an overflowing exponent).
    pub decode_token: Option<String>,
    pub vram: Option<f64>,
}

impl SampleFixture {
    pub fn new(decode: f64) -> Self {
        Self {
            decode,
            decode_token: None,
            vram: Some(7100.0),
        }
    }
}

pub struct Fixture {
    pub schema: String,
    pub run_id: String,
    pub date: String,
    pub protocol_id: String,
    pub protocol_sha256: String,
    pub repetitions: u64,
    pub context_tokens: u64,
    pub prompt_sha256: String,
    pub threads: u64,
    pub temperature: String,
    pub seed: u64,
    pub n_predict: u64,
    pub repeat_penalty: String,
    pub engine: String,
    pub binary_path: String,
    pub binary_sha256: String,
    pub model_path: String,
    pub model_sha256: String,
    pub host_fingerprint: String,
    pub driver_status: String,
    pub warmup_runs: u64,
    pub samples: Vec<SampleFixture>,
    pub ambient: String,
    pub reclaim: String,
    pub parity: String,
    pub median_override: Option<String>,
    pub sample_count_override: Option<u64>,
    pub extra_unknowns: Vec<(String, String)>,
    pub extra_root_field: Option<(String, String)>,
}

impl Fixture {
    pub fn new(engine: &str, binary_sha256: &str, model_sha256: &str) -> Self {
        Self {
            schema: "gnostral.rs/engine-benchmark-receipt/v1".into(),
            run_id: format!("synthetic-{engine}"),
            date: "2026-10-23".into(),
            protocol_id: "Q-025-ENGINE-DIRECTION-01".into(),
            protocol_sha256: PROTOCOL.into(),
            repetitions: 3,
            context_tokens: 1024,
            prompt_sha256: PROMPT.into(),
            threads: 8,
            temperature: "0".into(),
            seed: 42,
            n_predict: 256,
            repeat_penalty: "1".into(),
            engine: engine.into(),
            binary_path: format!("/opt/gnostral-lanes/{engine}/bin/serve"),
            binary_sha256: binary_sha256.into(),
            model_path: "/models/qwen3-30b-a3b-q2_k.gguf".into(),
            model_sha256: model_sha256.into(),
            host_fingerprint: "rtx3070-8gib/32gib-ram/8-threads/graphical".into(),
            driver_status: "NO_XID".into(),
            warmup_runs: 1,
            samples: vec![
                SampleFixture::new(34.0),
                SampleFixture::new(34.5),
                SampleFixture::new(35.0),
            ],
            ambient: "900".into(),
            reclaim: "12".into(),
            parity: "PASS".into(),
            median_override: None,
            sample_count_override: None,
            extra_unknowns: Vec::new(),
            extra_root_field: None,
        }
    }

    pub fn with_decodes(mut self, decodes: &[f64]) -> Self {
        self.samples = decodes.iter().copied().map(SampleFixture::new).collect();
        self
    }

    pub fn with_vram(mut self, index: usize, vram: Option<f64>) -> Self {
        self.samples[index].vram = vram;
        self
    }

    fn median(decodes: &[f64]) -> f64 {
        let mut sorted = decodes.to_vec();
        sorted.sort_by(f64::total_cmp);
        let middle = sorted.len() / 2;
        if sorted.len() % 2 == 1 {
            sorted[middle]
        } else {
            (sorted[middle - 1] + sorted[middle]) / 2.0
        }
    }

    pub fn render(&self) -> String {
        let decodes: Vec<f64> = self.samples.iter().map(|sample| sample.decode).collect();
        let sorted = {
            let mut values = decodes.clone();
            values.sort_by(f64::total_cmp);
            values
        };
        let median_token = self.median_override.clone().unwrap_or_else(|| {
            if sorted.is_empty() {
                "0".into()
            } else {
                format!("{}", Self::median(&decodes))
            }
        });
        let (min, max, p95) = if sorted.is_empty() {
            ("0".to_string(), "0".to_string(), "0".to_string())
        } else {
            let rank = (95 * sorted.len()).div_ceil(100);
            (
                format!("{}", sorted[0]),
                format!("{}", sorted[sorted.len() - 1]),
                format!("{}", sorted[rank.saturating_sub(1).min(sorted.len() - 1)]),
            )
        };

        let samples: Vec<String> = self
            .samples
            .iter()
            .map(|sample| {
                let decode = sample
                    .decode_token
                    .clone()
                    .unwrap_or_else(|| format!("{}", sample.decode));
                let vram = match sample.vram {
                    Some(value) => format!("{value}"),
                    None => "null".to_string(),
                };
                format!(
                    "{{ \"decode_tok_s\": {decode}, \"prefill_tok_s\": 210.5, \"ttft_ms\": 120.0, \"tpot_ms\": 28.0, \"vram_used_mib\": {vram}, \"host_rss_peak_mib\": 21000.0, \"exit_status\": \"OK\" }}"
                )
            })
            .collect();
        let samples = samples.join(",\n    ");

        let mut unknown_entries: Vec<(String, String)> = Vec::new();
        for (index, sample) in self.samples.iter().enumerate() {
            if sample.vram.is_none() {
                unknown_entries.push((
                    format!("samples[{index}].vram_used_mib"),
                    "no whole-card sampler attached in this synthetic fixture".into(),
                ));
            }
        }
        unknown_entries.extend(self.extra_unknowns.iter().cloned());
        let unknowns = unknown_entries
            .iter()
            .map(|(path, reason)| format!("\"{path}\": \"{reason}\""))
            .collect::<Vec<_>>()
            .join(",\n    ");

        let empty_unknowns = unknowns.is_empty();
        let unknowns = if empty_unknowns {
            String::new()
        } else {
            format!("\n    {unknowns}\n  ")
        };

        let mut fields = vec![
            format!("\"schema\": \"{}\"", self.schema),
            format!("\"run_id\": \"{}\"", self.run_id),
            format!("\"date\": \"{}\"", self.date),
            format!(
                "\"protocol\": {{ \"id\": \"{}\", \"sha256\": \"{}\", \"repetitions\": {}, \"context_tokens\": {}, \"prompt_sha256\": \"{}\", \"threads\": {}, \"sampling\": {{ \"temperature\": {}, \"seed\": {}, \"n_predict\": {}, \"repeat_penalty\": {} }} }}",
                self.protocol_id,
                self.protocol_sha256,
                self.repetitions,
                self.context_tokens,
                self.prompt_sha256,
                self.threads,
                self.temperature,
                self.seed,
                self.n_predict,
                self.repeat_penalty
            ),
            format!(
                "\"lane\": {{ \"engine\": \"{}\", \"binary_path\": \"{}\", \"binary_sha256\": \"{}\", \"model_path\": \"{}\", \"model_sha256\": \"{}\", \"host_fingerprint\": \"{}\", \"driver_status\": \"{}\" }}",
                self.engine,
                self.binary_path,
                self.binary_sha256,
                self.model_path,
                self.model_sha256,
                self.host_fingerprint,
                self.driver_status
            ),
            format!("\"warmup_runs\": {}", self.warmup_runs),
            format!("\"samples\": [\n    {samples}\n  ]"),
            format!("\"ambient_vram_mib\": {}", self.ambient),
            format!("\"reclaim_delta_mib\": {}", self.reclaim),
            format!("\"semantic_parity\": \"{}\"", self.parity),
            format!(
                "\"derived\": {{ \"sample_count\": {}, \"median_decode_tok_s\": {median_token}, \"min_decode_tok_s\": {min}, \"max_decode_tok_s\": {max}, \"p95_decode_tok_s\": {p95} }}",
                self.sample_count_override
                    .unwrap_or(self.samples.len() as u64)
            ),
            format!("\"unknowns\": {{{unknowns}}}"),
        ];
        if let Some((key, value)) = &self.extra_root_field {
            fields.push(format!("\"{key}\": {value}"));
        }
        format!("{{\n  {}\n}}\n", fields.join(",\n  "))
    }
}

static COUNTER: AtomicU64 = AtomicU64::new(0);

pub fn write_file(name: &str, body: &str) -> String {
    let serial = COUNTER.fetch_add(1, Ordering::Relaxed);
    let directory = std::env::temp_dir().join(format!(
        "gnostral-bench-tests-{}-{}",
        std::process::id(),
        serial
    ));
    std::fs::create_dir_all(&directory).expect("temporary directory");
    let path = directory.join(name);
    std::fs::write(&path, body).expect("temporary file");
    path.to_string_lossy().into_owned()
}

pub fn write_receipt(body: &str) -> String {
    write_file("receipt.json", body)
}
