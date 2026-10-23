//! The receipt model and its fail-closed validation.
//!
//! A receipt is evidence for exactly one lane under one frozen protocol. The
//! model is closed: unknown fields are rejected, every identity pin is
//! required, every nullable measurement must name itself in `unknowns` with a
//! reason, and the derived statistics must recompute exactly from the samples.

use std::collections::BTreeMap;
use std::fmt;

use crate::json::Json;
use crate::stats::{statistics, Stats};

pub const SCHEMA_ID: &str = "gnostral.rs/engine-benchmark-receipt/v1";
/// The frozen Q-025 protocol measures three repetitions per lane.
pub const MIN_REPETITIONS: u64 = 3;

const ROOT_FIELDS: &[&str] = &[
    "schema",
    "run_id",
    "date",
    "protocol",
    "lane",
    "warmup_runs",
    "samples",
    "ambient_vram_mib",
    "reclaim_delta_mib",
    "semantic_parity",
    "derived",
    "unknowns",
];
const PROTOCOL_FIELDS: &[&str] = &[
    "id",
    "sha256",
    "repetitions",
    "context_tokens",
    "prompt_sha256",
    "threads",
    "sampling",
];
const SAMPLING_FIELDS: &[&str] = &["temperature", "seed", "n_predict", "repeat_penalty"];
const LANE_FIELDS: &[&str] = &[
    "engine",
    "binary_path",
    "binary_sha256",
    "model_path",
    "model_sha256",
    "host_fingerprint",
    "driver_status",
];
const SAMPLE_FIELDS: &[&str] = &[
    "decode_tok_s",
    "prefill_tok_s",
    "ttft_ms",
    "tpot_ms",
    "vram_used_mib",
    "host_rss_peak_mib",
    "exit_status",
];
const DERIVED_FIELDS: &[&str] = &[
    "sample_count",
    "median_decode_tok_s",
    "min_decode_tok_s",
    "max_decode_tok_s",
    "p95_decode_tok_s",
];

#[derive(Debug, Clone, PartialEq)]
pub struct Receipt {
    pub schema: String,
    pub run_id: String,
    pub date: String,
    pub protocol: ProtocolId,
    pub lane: LaneIdentity,
    pub warmup_runs: u64,
    pub samples: Vec<Sample>,
    pub ambient_vram_mib: Option<f64>,
    pub reclaim_delta_mib: Option<f64>,
    pub semantic_parity: Parity,
    pub derived: Derived,
    pub unknowns: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProtocolId {
    pub id: String,
    pub sha256: String,
    pub repetitions: u64,
    pub context_tokens: u64,
    pub prompt_sha256: String,
    pub threads: u64,
    pub sampling: Sampling,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sampling {
    pub temperature: f64,
    pub seed: u64,
    pub n_predict: u64,
    pub repeat_penalty: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LaneIdentity {
    pub engine: String,
    pub binary_path: String,
    pub binary_sha256: String,
    pub model_path: String,
    pub model_sha256: String,
    pub host_fingerprint: String,
    pub driver_status: DriverStatus,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    pub decode_tok_s: f64,
    pub prefill_tok_s: Option<f64>,
    pub ttft_ms: Option<f64>,
    pub tpot_ms: Option<f64>,
    pub vram_used_mib: Option<f64>,
    pub host_rss_peak_mib: Option<f64>,
    pub exit_status: ExitStatus,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Derived {
    pub sample_count: u64,
    pub median_decode_tok_s: f64,
    pub min_decode_tok_s: f64,
    pub max_decode_tok_s: f64,
    pub p95_decode_tok_s: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Parity {
    Pass,
    Fail,
    Unknown,
}

impl Parity {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Fail => "FAIL",
            Self::Unknown => "UNKNOWN",
        }
    }

    fn parse(text: &str) -> Option<Self> {
        match text {
            "PASS" => Some(Self::Pass),
            "FAIL" => Some(Self::Fail),
            "UNKNOWN" => Some(Self::Unknown),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriverStatus {
    NoXid,
    Xid,
    Unknown,
}

impl DriverStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NoXid => "NO_XID",
            Self::Xid => "XID",
            Self::Unknown => "UNKNOWN",
        }
    }

    fn parse(text: &str) -> Option<Self> {
        match text {
            "NO_XID" => Some(Self::NoXid),
            "XID" => Some(Self::Xid),
            "UNKNOWN" => Some(Self::Unknown),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitStatus {
    Ok,
    NonZero,
    Unknown,
}

impl ExitStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "OK",
            Self::NonZero => "NONZERO",
            Self::Unknown => "UNKNOWN",
        }
    }

    fn parse(text: &str) -> Option<Self> {
        match text {
            "OK" => Some(Self::Ok),
            "NONZERO" => Some(Self::NonZero),
            "UNKNOWN" => Some(Self::Unknown),
            _ => None,
        }
    }
}

impl Receipt {
    /// Recompute the throughput statistics from the retained samples.
    pub fn recomputed(&self) -> Option<Stats> {
        let decode: Vec<f64> = self
            .samples
            .iter()
            .map(|sample| sample.decode_tok_s)
            .collect();
        statistics(&decode)
    }
}

fn join(path: &str, key: &str) -> String {
    if path.is_empty() {
        key.to_string()
    } else {
        format!("{path}.{key}")
    }
}

fn as_object<'a>(value: &'a Json, path: &str) -> Result<&'a [(String, Json)], String> {
    match value {
        Json::Obj(entries) => Ok(entries),
        _ => Err(format!("{path}: expected an object")),
    }
}

fn field<'a>(entries: &'a [(String, Json)], key: &str, path: &str) -> Result<&'a Json, String> {
    entries
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value)
        .ok_or_else(|| format!("{}: required field is missing", join(path, key)))
}

fn reject_unknown(entries: &[(String, Json)], allowed: &[&str], path: &str) -> Result<(), String> {
    for (key, _) in entries {
        if !allowed.contains(&key.as_str()) {
            return Err(format!("{}: unknown field", join(path, key)));
        }
    }
    Ok(())
}

fn as_string(value: &Json, path: &str) -> Result<String, String> {
    match value {
        Json::Str(text) => Ok(text.clone()),
        _ => Err(format!("{path}: expected a string")),
    }
}

fn as_u64(value: &Json, path: &str) -> Result<u64, String> {
    match value {
        Json::Num(number) => {
            // Integers above 2^53 cannot be represented exactly by an f64 and
            // are rejected rather than silently rounded.
            if !number.is_finite()
                || number.fract() != 0.0
                || *number < 0.0
                || *number > 9_007_199_254_740_992.0
            {
                Err(format!(
                    "{path}: expected an exact non-negative integer, found {number}"
                ))
            } else {
                Ok(*number as u64)
            }
        }
        _ => Err(format!("{path}: expected an integer")),
    }
}

fn as_f64(value: &Json, path: &str) -> Result<f64, String> {
    match value {
        Json::Num(number) if number.is_finite() => Ok(*number),
        Json::Num(number) => Err(format!("{path}: number is not finite: {number}")),
        _ => Err(format!("{path}: expected a number")),
    }
}

fn as_f64_or_null(value: &Json, path: &str) -> Result<Option<f64>, String> {
    if matches!(value, Json::Null) {
        Ok(None)
    } else {
        as_f64(value, path).map(Some)
    }
}

fn string_field(entries: &[(String, Json)], key: &str, path: &str) -> Result<String, String> {
    as_string(field(entries, key, path)?, &join(path, key))
}

fn u64_field(entries: &[(String, Json)], key: &str, path: &str) -> Result<u64, String> {
    as_u64(field(entries, key, path)?, &join(path, key))
}

fn f64_field(entries: &[(String, Json)], key: &str, path: &str) -> Result<f64, String> {
    as_f64(field(entries, key, path)?, &join(path, key))
}

fn f64_or_null_field(
    entries: &[(String, Json)],
    key: &str,
    path: &str,
) -> Result<Option<f64>, String> {
    as_f64_or_null(field(entries, key, path)?, &join(path, key))
}

fn object_field<'a>(
    entries: &'a [(String, Json)],
    key: &str,
    path: &str,
) -> Result<&'a [(String, Json)], String> {
    as_object(field(entries, key, path)?, &join(path, key))
}

fn array_field<'a>(
    entries: &'a [(String, Json)],
    key: &str,
    path: &str,
) -> Result<&'a [Json], String> {
    match field(entries, key, path)? {
        Json::Arr(items) => Ok(items),
        _ => Err(format!("{}: expected an array", join(path, key))),
    }
}

fn enum_field<T>(
    entries: &[(String, Json)],
    key: &str,
    path: &str,
    allowed: &[&str],
    parse: fn(&str) -> Option<T>,
) -> Result<T, String> {
    let text = string_field(entries, key, path)?;
    parse(&text).ok_or_else(|| {
        format!(
            "{}: expected one of {:?}, found {:?}",
            join(path, key),
            allowed,
            text
        )
    })
}

pub fn decode(value: &Json) -> Result<Receipt, String> {
    let root = as_object(value, "receipt")?;
    reject_unknown(root, ROOT_FIELDS, "receipt")?;

    let protocol_entries = object_field(root, "protocol", "receipt")?;
    reject_unknown(protocol_entries, PROTOCOL_FIELDS, "protocol")?;
    let sampling_entries = object_field(protocol_entries, "sampling", "protocol")?;
    reject_unknown(sampling_entries, SAMPLING_FIELDS, "protocol.sampling")?;
    let sampling = Sampling {
        temperature: f64_field(sampling_entries, "temperature", "protocol.sampling")?,
        seed: u64_field(sampling_entries, "seed", "protocol.sampling")?,
        n_predict: u64_field(sampling_entries, "n_predict", "protocol.sampling")?,
        repeat_penalty: f64_field(sampling_entries, "repeat_penalty", "protocol.sampling")?,
    };
    let protocol = ProtocolId {
        id: string_field(protocol_entries, "id", "protocol")?,
        sha256: string_field(protocol_entries, "sha256", "protocol")?,
        repetitions: u64_field(protocol_entries, "repetitions", "protocol")?,
        context_tokens: u64_field(protocol_entries, "context_tokens", "protocol")?,
        prompt_sha256: string_field(protocol_entries, "prompt_sha256", "protocol")?,
        threads: u64_field(protocol_entries, "threads", "protocol")?,
        sampling,
    };

    let lane_entries = object_field(root, "lane", "receipt")?;
    reject_unknown(lane_entries, LANE_FIELDS, "lane")?;
    let lane = LaneIdentity {
        engine: string_field(lane_entries, "engine", "lane")?,
        binary_path: string_field(lane_entries, "binary_path", "lane")?,
        binary_sha256: string_field(lane_entries, "binary_sha256", "lane")?,
        model_path: string_field(lane_entries, "model_path", "lane")?,
        model_sha256: string_field(lane_entries, "model_sha256", "lane")?,
        host_fingerprint: string_field(lane_entries, "host_fingerprint", "lane")?,
        driver_status: enum_field(
            lane_entries,
            "driver_status",
            "lane",
            &["NO_XID", "XID", "UNKNOWN"],
            DriverStatus::parse,
        )?,
    };

    let sample_values = array_field(root, "samples", "receipt")?;
    let mut samples = Vec::with_capacity(sample_values.len());
    for (index, sample_value) in sample_values.iter().enumerate() {
        let path = format!("samples[{index}]");
        let entries = as_object(sample_value, &path)?;
        reject_unknown(entries, SAMPLE_FIELDS, &path)?;
        samples.push(Sample {
            decode_tok_s: f64_field(entries, "decode_tok_s", &path)?,
            prefill_tok_s: f64_or_null_field(entries, "prefill_tok_s", &path)?,
            ttft_ms: f64_or_null_field(entries, "ttft_ms", &path)?,
            tpot_ms: f64_or_null_field(entries, "tpot_ms", &path)?,
            vram_used_mib: f64_or_null_field(entries, "vram_used_mib", &path)?,
            host_rss_peak_mib: f64_or_null_field(entries, "host_rss_peak_mib", &path)?,
            exit_status: enum_field(
                entries,
                "exit_status",
                &path,
                &["OK", "NONZERO", "UNKNOWN"],
                ExitStatus::parse,
            )?,
        });
    }

    let derived_entries = object_field(root, "derived", "receipt")?;
    reject_unknown(derived_entries, DERIVED_FIELDS, "derived")?;
    let derived = Derived {
        sample_count: u64_field(derived_entries, "sample_count", "derived")?,
        median_decode_tok_s: f64_field(derived_entries, "median_decode_tok_s", "derived")?,
        min_decode_tok_s: f64_field(derived_entries, "min_decode_tok_s", "derived")?,
        max_decode_tok_s: f64_field(derived_entries, "max_decode_tok_s", "derived")?,
        p95_decode_tok_s: f64_field(derived_entries, "p95_decode_tok_s", "derived")?,
    };

    let unknown_entries = object_field(root, "unknowns", "receipt")?;
    let mut unknowns = BTreeMap::new();
    for (key, reason_value) in unknown_entries {
        let reason = as_string(reason_value, &format!("unknowns.{key}"))?;
        unknowns.insert(key.clone(), reason);
    }

    Ok(Receipt {
        schema: string_field(root, "schema", "receipt")?,
        run_id: string_field(root, "run_id", "receipt")?,
        date: string_field(root, "date", "receipt")?,
        protocol,
        lane,
        warmup_runs: u64_field(root, "warmup_runs", "receipt")?,
        samples,
        ambient_vram_mib: f64_or_null_field(root, "ambient_vram_mib", "receipt")?,
        reclaim_delta_mib: f64_or_null_field(root, "reclaim_delta_mib", "receipt")?,
        semantic_parity: enum_field(
            root,
            "semantic_parity",
            "receipt",
            &["PASS", "FAIL", "UNKNOWN"],
            Parity::parse,
        )?,
        derived,
        unknowns,
    })
}

/// A SHA-256 pin is exactly 64 lowercase hexadecimal characters.
pub fn is_sha256_hex(text: &str) -> bool {
    text.len() == 64
        && text
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_absolute_path(text: &str) -> bool {
    text.starts_with('/') && text.len() > 1
}

fn is_valid_date(text: &str) -> bool {
    let bytes = text.as_bytes();
    if bytes.len() != 10 || bytes[4] != b'-' || bytes[7] != b'-' {
        return false;
    }
    let digits = |range: std::ops::Range<usize>| -> Option<u32> {
        text.get(range)
            .filter(|part| part.bytes().all(|byte| byte.is_ascii_digit()))
            .and_then(|part| part.parse().ok())
    };
    let (Some(year), Some(month), Some(day)) = (digits(0..4), digits(5..7), digits(8..10)) else {
        return false;
    };
    if year < 2000 || !(1..=12).contains(&month) {
        return false;
    }
    let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    let days = match month {
        2 if leap => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    };
    (1..=days).contains(&day)
}

fn non_empty(errors: &mut Vec<String>, path: &str, value: &str) {
    if value.trim().is_empty() {
        errors.push(format!("{path}: must not be empty"));
    }
}

fn check_positive(errors: &mut Vec<String>, path: &str, value: Option<f64>) {
    if let Some(value) = value {
        if !value.is_finite() || value <= 0.0 {
            errors.push(format!(
                "{path}: expected a finite positive number, found {value}"
            ));
        }
    }
}

/// Fail closed. An empty vector means the receipt is internally consistent;
/// it does not claim the run happened or that the result is good.
pub fn validate(receipt: &Receipt, expected_protocol_sha256: Option<&str>) -> Vec<String> {
    let mut errors = Vec::new();

    if receipt.schema != SCHEMA_ID {
        errors.push(format!(
            "schema: expected {SCHEMA_ID:?}, found {:?}",
            receipt.schema
        ));
    }
    non_empty(&mut errors, "run_id", &receipt.run_id);
    if !is_valid_date(&receipt.date) {
        errors.push(format!(
            "date: expected YYYY-MM-DD, found {:?}",
            receipt.date
        ));
    }

    let protocol = &receipt.protocol;
    non_empty(&mut errors, "protocol.id", &protocol.id);
    if !is_sha256_hex(&protocol.sha256) {
        errors.push(format!(
            "protocol.sha256: expected 64 lowercase hex characters, found {:?}",
            protocol.sha256
        ));
    }
    if let Some(expected) = expected_protocol_sha256 {
        if protocol.sha256 != expected {
            errors.push(format!(
                "protocol.sha256: receipt records {:?}, expected {:?}",
                protocol.sha256, expected
            ));
        }
    }
    if protocol.repetitions < MIN_REPETITIONS {
        errors.push(format!(
            "protocol.repetitions: the frozen protocol requires at least {MIN_REPETITIONS} measured samples, found {}",
            protocol.repetitions
        ));
    }
    if protocol.context_tokens == 0 {
        errors.push("protocol.context_tokens: must be greater than zero".into());
    }
    if !is_sha256_hex(&protocol.prompt_sha256) {
        errors.push(format!(
            "protocol.prompt_sha256: expected 64 lowercase hex characters, found {:?}",
            protocol.prompt_sha256
        ));
    }
    if !(1..=1024).contains(&protocol.threads) {
        errors.push(format!(
            "protocol.threads: expected 1..=1024, found {}",
            protocol.threads
        ));
    }
    let sampling = &protocol.sampling;
    if !sampling.temperature.is_finite() || sampling.temperature < 0.0 {
        errors.push(format!(
            "protocol.sampling.temperature: expected a finite non-negative value, found {}",
            sampling.temperature
        ));
    }
    if sampling.n_predict == 0 {
        errors.push("protocol.sampling.n_predict: must be greater than zero".into());
    }
    if !sampling.repeat_penalty.is_finite() || sampling.repeat_penalty <= 0.0 {
        errors.push(format!(
            "protocol.sampling.repeat_penalty: expected a finite positive value, found {}",
            sampling.repeat_penalty
        ));
    }

    let lane = &receipt.lane;
    non_empty(&mut errors, "lane.engine", &lane.engine);
    non_empty(&mut errors, "lane.binary_path", &lane.binary_path);
    if !is_absolute_path(&lane.binary_path) {
        errors.push(format!(
            "lane.binary_path: expected an absolute path, found {:?}",
            lane.binary_path
        ));
    }
    if !is_sha256_hex(&lane.binary_sha256) {
        errors.push(format!(
            "lane.binary_sha256: expected 64 lowercase hex characters, found {:?}",
            lane.binary_sha256
        ));
    }
    non_empty(&mut errors, "lane.model_path", &lane.model_path);
    if !is_absolute_path(&lane.model_path) {
        errors.push(format!(
            "lane.model_path: expected an absolute path, found {:?}",
            lane.model_path
        ));
    }
    if !is_sha256_hex(&lane.model_sha256) {
        errors.push(format!(
            "lane.model_sha256: expected 64 lowercase hex characters, found {:?}",
            lane.model_sha256
        ));
    }
    non_empty(&mut errors, "lane.host_fingerprint", &lane.host_fingerprint);

    if receipt.warmup_runs < 1 {
        errors.push(format!(
            "warmup_runs: at least one discarded warmup run is required, found {}",
            receipt.warmup_runs
        ));
    }

    if receipt.samples.len() as u64 != protocol.repetitions {
        errors.push(format!(
            "samples: protocol.repetitions requires {} measured samples, found {}",
            protocol.repetitions,
            receipt.samples.len()
        ));
    }
    for (index, sample) in receipt.samples.iter().enumerate() {
        let path = format!("samples[{index}]");
        if !sample.decode_tok_s.is_finite() || sample.decode_tok_s <= 0.0 {
            errors.push(format!(
                "{path}.decode_tok_s: expected a finite positive number, found {}",
                sample.decode_tok_s
            ));
        }
        check_positive(
            &mut errors,
            &format!("{path}.prefill_tok_s"),
            sample.prefill_tok_s,
        );
        check_positive(&mut errors, &format!("{path}.ttft_ms"), sample.ttft_ms);
        check_positive(&mut errors, &format!("{path}.tpot_ms"), sample.tpot_ms);
        check_positive(
            &mut errors,
            &format!("{path}.vram_used_mib"),
            sample.vram_used_mib,
        );
        check_positive(
            &mut errors,
            &format!("{path}.host_rss_peak_mib"),
            sample.host_rss_peak_mib,
        );
        if sample.exit_status != ExitStatus::Ok {
            errors.push(format!(
                "{path}.exit_status: a measured sample must have exit status OK, found {}",
                sample.exit_status.as_str()
            ));
        }
    }
    check_positive(&mut errors, "ambient_vram_mib", receipt.ambient_vram_mib);
    if let Some(delta) = receipt.reclaim_delta_mib {
        if !delta.is_finite() {
            errors.push(format!(
                "reclaim_delta_mib: expected a finite number, found {delta}"
            ));
        }
    }

    match receipt.recomputed() {
        Some(stats) => {
            let derived = &receipt.derived;
            if derived.sample_count != receipt.samples.len() as u64 {
                errors.push(format!(
                    "derived.sample_count: claimed {}, recomputed {}",
                    derived.sample_count,
                    receipt.samples.len()
                ));
            }
            if derived.median_decode_tok_s != stats.median {
                errors.push(format!(
                    "derived.median_decode_tok_s: claimed {}, recomputed {} from {} samples",
                    derived.median_decode_tok_s,
                    stats.median,
                    receipt.samples.len()
                ));
            }
            if derived.min_decode_tok_s != stats.min {
                errors.push(format!(
                    "derived.min_decode_tok_s: claimed {}, recomputed {}",
                    derived.min_decode_tok_s, stats.min
                ));
            }
            if derived.max_decode_tok_s != stats.max {
                errors.push(format!(
                    "derived.max_decode_tok_s: claimed {}, recomputed {}",
                    derived.max_decode_tok_s, stats.max
                ));
            }
            if derived.p95_decode_tok_s != stats.p95 {
                errors.push(format!(
                    "derived.p95_decode_tok_s: claimed {}, recomputed {}",
                    derived.p95_decode_tok_s, stats.p95
                ));
            }
        }
        None => errors.push("samples: no finite decode samples to derive statistics from".into()),
    }

    validate_unknowns(receipt, &mut errors);

    errors
}

fn nullable_paths(receipt: &Receipt) -> Vec<String> {
    let mut paths = Vec::new();
    if receipt.ambient_vram_mib.is_none() {
        paths.push("ambient_vram_mib".to_string());
    }
    if receipt.reclaim_delta_mib.is_none() {
        paths.push("reclaim_delta_mib".to_string());
    }
    for (index, sample) in receipt.samples.iter().enumerate() {
        let fields = [
            ("prefill_tok_s", sample.prefill_tok_s),
            ("ttft_ms", sample.ttft_ms),
            ("tpot_ms", sample.tpot_ms),
            ("vram_used_mib", sample.vram_used_mib),
            ("host_rss_peak_mib", sample.host_rss_peak_mib),
        ];
        for (name, value) in fields {
            if value.is_none() {
                paths.push(format!("samples[{index}].{name}"));
            }
        }
    }
    paths
}

fn validate_unknowns(receipt: &Receipt, errors: &mut Vec<String>) {
    let nulls = nullable_paths(receipt);
    for path in &nulls {
        match receipt.unknowns.get(path) {
            None => errors.push(format!(
                "{path}: value is null but has no entry in unknowns (UNKNOWN must be explicit)"
            )),
            Some(reason) if reason.trim().is_empty() => {
                errors.push(format!("unknowns.{path}: reason must not be empty"));
            }
            Some(_) => {}
        }
    }
    for (path, reason) in &receipt.unknowns {
        if !nulls.contains(path) {
            errors.push(format!(
                "unknowns.{path}: reason given for a field that is not UNKNOWN ({reason:?})"
            ));
        }
    }
}

impl fmt::Display for LaneIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} (binary {}, model {}, driver {})",
            self.engine,
            self.binary_sha256,
            self.model_sha256,
            self.driver_status.as_str()
        )
    }
}
