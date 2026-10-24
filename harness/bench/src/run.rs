//! Frozen-protocol lane driver (Q-031).
//!
//! A plan names the protocol and the lanes. For each lane the driver hashes the
//! binary and the model and compares them with the plan's expected pins; a lane
//! whose pin is missing or different is `BLOCKED`. Execution order is frozen:
//! one warmup per lane in plan order, then `repetitions` measured rounds, each
//! round running every runnable lane once in plan order (interleaved). Every run,
//! failed or not, is kept with its raw output.
//!
//! A lane reports metrics by printing one line
//! `GNOSTRAL_METRICS {"decode_tok_s": .., "prefill_tok_s": .., ...}` and may print
//! `GNOSTRAL_PARITY PASS|FAIL`. Anything it does not report is `UNKNOWN` with a
//! reason in the receipt. Receipts are built only from these lines; the Pareto
//! table is rendered only from validated receipts, and a blocked or failed lane
//! aborts the comparison, not the run.
//!
//! Plan v2 adds two identity checks, both learned from a real run whose layer
//! placement changed between loads (docs/FINDINGS.md, F-010):
//!
//! - every run prints `GNOSTRAL_PLACEMENT <canonical>` and it must equal the lane's
//!   `expected_placement`, warmup included; a drift fails the lane;
//! - the warmup prints `GNOSTRAL_SELF_REPRO REFERENCE` and keeps its output; every
//!   measured run prints `GNOSTRAL_SELF_REPRO PASS|FAIL` after comparing its tokens
//!   and logprob bits with that reference. A lane that does not reproduce itself
//!   cannot be ranked.

use std::collections::BTreeMap;

use crate::compare::compare;
use crate::json::{self, Json};
use crate::receipt::{
    self, Derived, DriverStatus, ExitStatus, LaneIdentity, Parity, ParityClass, ProtocolId,
    Receipt, Sample, Sampling, SCHEMA_ID,
};
use crate::stats::statistics;

pub const PLAN_SCHEMA: &str = "gnostral.rs/engine-benchmark-plan/v2";
pub const METRICS_PREFIX: &str = "GNOSTRAL_METRICS ";
pub const PARITY_PREFIX: &str = "GNOSTRAL_PARITY ";
pub const DRIVER_PREFIX: &str = "GNOSTRAL_DRIVER ";
pub const PLACEMENT_PREFIX: &str = "GNOSTRAL_PLACEMENT ";
pub const SELF_REPRO_PREFIX: &str = "GNOSTRAL_SELF_REPRO ";

#[derive(Debug, Clone, PartialEq)]
pub struct LanePlan {
    pub engine: String,
    pub binary_path: String,
    pub expected_binary_sha256: String,
    pub model_path: String,
    pub expected_model_sha256: String,
    pub host_fingerprint: String,
    pub argv: Vec<String>,
    /// Canonical placement every run of this lane must report.
    pub expected_placement: String,
    pub parity_class: ParityClass,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    pub run_id: String,
    pub date: String,
    pub protocol: ProtocolId,
    pub warmup_runs: u64,
    pub lanes: Vec<LanePlan>,
    /// Command printing `GNOSTRAL_DRIVER NO_XID|XID`; absent means `UNKNOWN`.
    pub driver_probe: Option<Vec<String>>,
}

fn fields<'a>(v: &'a Json, path: &str, allowed: &[&str]) -> Result<&'a [(String, Json)], String> {
    let Json::Obj(entries) = v else {
        return Err(format!("{path}: expected an object"));
    };
    for (k, _) in entries {
        if !allowed.contains(&k.as_str()) {
            return Err(format!("{path}.{k}: unknown field"));
        }
    }
    Ok(entries)
}

fn get<'a>(e: &'a [(String, Json)], key: &str, path: &str) -> Result<&'a Json, String> {
    e.iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v)
        .ok_or_else(|| format!("{path}.{key}: required field is missing"))
}

fn text(e: &[(String, Json)], key: &str, path: &str) -> Result<String, String> {
    match get(e, key, path)? {
        Json::Str(s) => Ok(s.clone()),
        _ => Err(format!("{path}.{key}: expected a string")),
    }
}

fn number(e: &[(String, Json)], key: &str, path: &str) -> Result<f64, String> {
    match get(e, key, path)? {
        Json::Num(n) if n.is_finite() => Ok(*n),
        _ => Err(format!("{path}.{key}: expected a finite number")),
    }
}

fn uint(e: &[(String, Json)], key: &str, path: &str) -> Result<u64, String> {
    let n = number(e, key, path)?;
    if n < 0.0 || n.fract() != 0.0 || n > 9_007_199_254_740_992.0 {
        return Err(format!("{path}.{key}: expected a non-negative integer"));
    }
    Ok(n as u64)
}

fn argv(v: &Json, path: &str) -> Result<Vec<String>, String> {
    let Json::Arr(items) = v else {
        return Err(format!("{path}: expected an array of strings"));
    };
    let out: Vec<String> = items
        .iter()
        .map(|i| match i {
            Json::Str(s) => Ok(s.clone()),
            _ => Err(format!("{path}: expected an array of strings")),
        })
        .collect::<Result<_, _>>()?;
    if out.is_empty() {
        return Err(format!("{path}: must not be empty"));
    }
    Ok(out)
}

pub fn decode_plan(v: &Json) -> Result<Plan, String> {
    let root = fields(
        v,
        "plan",
        &[
            "schema",
            "run_id",
            "date",
            "protocol",
            "warmup_runs",
            "lanes",
            "driver_probe",
        ],
    )?;
    if text(root, "schema", "plan")? != PLAN_SCHEMA {
        return Err(format!("plan.schema: expected {PLAN_SCHEMA:?}"));
    }
    let p = fields(
        get(root, "protocol", "plan")?,
        "plan.protocol",
        &[
            "id",
            "sha256",
            "repetitions",
            "context_tokens",
            "prompt_sha256",
            "threads",
            "sampling",
        ],
    )?;
    let s = fields(
        get(p, "sampling", "plan.protocol")?,
        "plan.protocol.sampling",
        &["temperature", "seed", "n_predict", "repeat_penalty"],
    )?;
    let protocol = ProtocolId {
        id: text(p, "id", "plan.protocol")?,
        sha256: text(p, "sha256", "plan.protocol")?,
        repetitions: uint(p, "repetitions", "plan.protocol")?,
        context_tokens: uint(p, "context_tokens", "plan.protocol")?,
        prompt_sha256: text(p, "prompt_sha256", "plan.protocol")?,
        threads: uint(p, "threads", "plan.protocol")?,
        sampling: Sampling {
            temperature: number(s, "temperature", "plan.protocol.sampling")?,
            seed: uint(s, "seed", "plan.protocol.sampling")?,
            n_predict: uint(s, "n_predict", "plan.protocol.sampling")?,
            repeat_penalty: number(s, "repeat_penalty", "plan.protocol.sampling")?,
        },
    };
    let Json::Arr(lane_items) = get(root, "lanes", "plan")? else {
        return Err("plan.lanes: expected an array".into());
    };
    let mut lanes = Vec::new();
    for (i, item) in lane_items.iter().enumerate() {
        let path = format!("plan.lanes[{i}]");
        let l = fields(
            item,
            &path,
            &[
                "engine",
                "binary_path",
                "expected_binary_sha256",
                "model_path",
                "expected_model_sha256",
                "host_fingerprint",
                "argv",
                "expected_placement",
                "parity_class",
            ],
        )?;
        let expected_placement = text(l, "expected_placement", &path)?;
        if expected_placement.trim().is_empty() {
            return Err(format!("{path}.expected_placement: must not be empty"));
        }
        let class = text(l, "parity_class", &path)?;
        let parity_class = ParityClass::parse(&class).ok_or_else(|| {
            format!("{path}.parity_class: expected \"exact\" or \"self\", found {class:?}")
        })?;
        lanes.push(LanePlan {
            engine: text(l, "engine", &path)?,
            binary_path: text(l, "binary_path", &path)?,
            expected_binary_sha256: text(l, "expected_binary_sha256", &path)?,
            model_path: text(l, "model_path", &path)?,
            expected_model_sha256: text(l, "expected_model_sha256", &path)?,
            host_fingerprint: text(l, "host_fingerprint", &path)?,
            argv: argv(get(l, "argv", &path)?, &format!("{path}.argv"))?,
            expected_placement,
            parity_class,
        });
    }
    let mut seen = std::collections::BTreeSet::new();
    if lanes.len() < 2 || !lanes.iter().all(|l| seen.insert(l.engine.clone())) {
        return Err(
            "plan.lanes: at least two lanes with distinct engine names are required".into(),
        );
    }
    let driver_probe = match root.iter().find(|(k, _)| k == "driver_probe") {
        None | Some((_, Json::Null)) => None,
        Some((_, v)) => Some(argv(v, "plan.driver_probe")?),
    };
    Ok(Plan {
        run_id: text(root, "run_id", "plan")?,
        date: text(root, "date", "plan")?,
        protocol,
        warmup_runs: uint(root, "warmup_runs", "plan")?,
        lanes,
        driver_probe,
    })
}

/// Output of one command.
#[derive(Debug, Clone, PartialEq)]
pub struct Output {
    pub exit_ok: bool,
    pub stdout: String,
    pub stderr: String,
    pub wall_ms: f64,
}

/// Runs commands and hashes files. The real one uses processes; tests inject fakes.
pub trait Executor {
    fn run(&mut self, argv: &[String]) -> Result<Output, String>;
    fn sha256_file(&mut self, path: &str) -> Result<String, String>;
}

pub struct ProcessExecutor;

impl Executor for ProcessExecutor {
    fn run(&mut self, argv: &[String]) -> Result<Output, String> {
        let start = std::time::Instant::now();
        let out = std::process::Command::new(&argv[0])
            .args(&argv[1..])
            .output()
            .map_err(|e| format!("spawn {:?}: {e}", argv[0]))?;
        Ok(Output {
            exit_ok: out.status.success(),
            stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
            wall_ms: start.elapsed().as_secs_f64() * 1e3,
        })
    }
    fn sha256_file(&mut self, path: &str) -> Result<String, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        Ok(crate::sha256::hex(&crate::sha256::sha256(&bytes)))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Pin {
    Pinned,
    Blocked(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunPhase {
    Warmup,
    Measured,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RunRecord {
    pub lane: String,
    pub phase: RunPhase,
    pub index: u64,
    pub output: Result<Output, String>,
    pub metrics: Result<Metrics, String>,
    pub parity: Parity,
    /// What the run printed after `GNOSTRAL_PLACEMENT `, if anything.
    pub placement: Option<String>,
    pub self_repro: SelfRepro,
}

/// A run's `GNOSTRAL_SELF_REPRO` report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelfRepro {
    /// This run's output is the lane's reference (the warmup).
    Reference,
    Pass,
    Fail,
    Unknown,
}

impl SelfRepro {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Reference => "REFERENCE",
            Self::Pass => "PASS",
            Self::Fail => "FAIL",
            Self::Unknown => "UNKNOWN",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Metrics {
    pub decode_tok_s: f64,
    pub prefill_tok_s: Option<f64>,
    pub ttft_ms: Option<f64>,
    pub tpot_ms: Option<f64>,
    pub vram_used_mib: Option<f64>,
    pub host_rss_peak_mib: Option<f64>,
}

const METRIC_FIELDS: &[&str] = &[
    "decode_tok_s",
    "prefill_tok_s",
    "ttft_ms",
    "tpot_ms",
    "vram_used_mib",
    "host_rss_peak_mib",
];

pub fn parse_metrics(stdout: &str) -> Result<Metrics, String> {
    let lines: Vec<&str> = stdout
        .lines()
        .filter_map(|l| l.strip_prefix(METRICS_PREFIX))
        .collect();
    let [line] = lines[..] else {
        return Err(format!(
            "expected exactly one {METRICS_PREFIX:?} line, found {}",
            lines.len()
        ));
    };
    let v = json::parse(line).map_err(|e| e.to_string())?;
    let e = fields(&v, "metrics", METRIC_FIELDS)?;
    let opt = |k: &str| -> Result<Option<f64>, String> {
        match e.iter().find(|(n, _)| n == k) {
            None | Some((_, Json::Null)) => Ok(None),
            Some((_, Json::Num(n))) if n.is_finite() => Ok(Some(*n)),
            _ => Err(format!("metrics.{k}: expected a finite number or null")),
        }
    };
    Ok(Metrics {
        decode_tok_s: opt("decode_tok_s")?.ok_or("metrics.decode_tok_s: required")?,
        prefill_tok_s: opt("prefill_tok_s")?,
        ttft_ms: opt("ttft_ms")?,
        tpot_ms: opt("tpot_ms")?,
        vram_used_mib: opt("vram_used_mib")?,
        host_rss_peak_mib: opt("host_rss_peak_mib")?,
    })
}

/// The single value after `prefix`, or `None` if the line is absent or repeated.
fn single_line<'a>(stdout: &'a str, prefix: &str) -> Option<&'a str> {
    let found: Vec<&str> = stdout
        .lines()
        .filter_map(|l| l.strip_prefix(prefix))
        .map(str::trim)
        .collect();
    match found[..] {
        [one] => Some(one),
        _ => None,
    }
}

pub fn parse_placement(stdout: &str) -> Option<String> {
    single_line(stdout, PLACEMENT_PREFIX)
        .filter(|p| !p.is_empty() && *p != "UNKNOWN")
        .map(str::to_owned)
}

/// The first word decides; anything after it is detail kept in the raw output.
pub fn parse_self_repro(stdout: &str) -> SelfRepro {
    match single_line(stdout, SELF_REPRO_PREFIX).and_then(|v| v.split_whitespace().next()) {
        Some("REFERENCE") => SelfRepro::Reference,
        Some("PASS") => SelfRepro::Pass,
        Some("FAIL") => SelfRepro::Fail,
        _ => SelfRepro::Unknown,
    }
}

fn parse_parity(stdout: &str) -> Parity {
    let found: Vec<&str> = stdout
        .lines()
        .filter_map(|l| l.strip_prefix(PARITY_PREFIX))
        .collect();
    match found[..] {
        ["PASS"] => Parity::Pass,
        ["FAIL"] => Parity::Fail,
        _ => Parity::Unknown,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Bundle {
    pub plan: Plan,
    pub pins: BTreeMap<String, Pin>,
    pub driver: DriverStatus,
    /// Frozen execution order actually used.
    pub order: Vec<(String, RunPhase, u64)>,
    pub records: Vec<RunRecord>,
    pub receipts: BTreeMap<String, Result<Receipt, String>>,
    pub comparison: Result<String, Vec<String>>,
}

pub fn pin_lane(lane: &LanePlan, exec: &mut dyn Executor) -> Pin {
    for (what, path, expected) in [
        ("binary", &lane.binary_path, &lane.expected_binary_sha256),
        ("model", &lane.model_path, &lane.expected_model_sha256),
    ] {
        if !receipt::is_sha256_hex(expected) {
            return Pin::Blocked(format!(
                "{what}: no valid expected sha256 pinned in the plan"
            ));
        }
        match exec.sha256_file(path) {
            Ok(actual) if &actual == expected => {}
            Ok(actual) => {
                return Pin::Blocked(format!("{what}: sha256 {actual} != pinned {expected}"))
            }
            Err(e) => return Pin::Blocked(format!("{what}: cannot hash: {e}")),
        }
    }
    Pin::Pinned
}

/// Frozen order: warmups lane-major, then measured rounds, each round every lane once.
pub fn frozen_order(plan: &Plan, runnable: &[String]) -> Vec<(String, RunPhase, u64)> {
    let mut order = Vec::new();
    for w in 0..plan.warmup_runs {
        for lane in runnable {
            order.push((lane.clone(), RunPhase::Warmup, w));
        }
    }
    for r in 0..plan.protocol.repetitions {
        for lane in runnable {
            order.push((lane.clone(), RunPhase::Measured, r));
        }
    }
    order
}

pub fn execute(plan: &Plan, exec: &mut dyn Executor) -> Bundle {
    let mut pins = BTreeMap::new();
    for lane in &plan.lanes {
        pins.insert(lane.engine.clone(), pin_lane(lane, exec));
    }
    let runnable: Vec<String> = plan
        .lanes
        .iter()
        .filter(|l| pins[&l.engine] == Pin::Pinned)
        .map(|l| l.engine.clone())
        .collect();
    let driver = match &plan.driver_probe {
        None => DriverStatus::Unknown,
        Some(argv) => match exec.run(argv) {
            Ok(o) if o.exit_ok => {
                let found: Vec<&str> = o
                    .stdout
                    .lines()
                    .filter_map(|l| l.strip_prefix(DRIVER_PREFIX))
                    .collect();
                match found[..] {
                    ["NO_XID"] => DriverStatus::NoXid,
                    ["XID"] => DriverStatus::Xid,
                    _ => DriverStatus::Unknown,
                }
            }
            _ => DriverStatus::Unknown,
        },
    };
    let order = frozen_order(plan, &runnable);
    let mut records = Vec::new();
    for (engine, phase, index) in &order {
        let lane = plan
            .lanes
            .iter()
            .find(|l| &l.engine == engine)
            .expect("runnable lane");
        let output = exec.run(&lane.argv);
        let (metrics, parity) = match &output {
            Ok(o) if o.exit_ok => (parse_metrics(&o.stdout), parse_parity(&o.stdout)),
            Ok(_) => (Err("non-zero exit".into()), Parity::Unknown),
            Err(e) => (Err(e.clone()), Parity::Unknown),
        };
        let (placement, self_repro) = match &output {
            Ok(o) => (parse_placement(&o.stdout), parse_self_repro(&o.stdout)),
            Err(_) => (None, SelfRepro::Unknown),
        };
        records.push(RunRecord {
            lane: engine.clone(),
            phase: *phase,
            index: *index,
            output,
            metrics,
            parity,
            placement,
            self_repro,
        });
    }

    let mut receipts = BTreeMap::new();
    for lane in &plan.lanes {
        let built = match &pins[&lane.engine] {
            Pin::Blocked(why) => Err(format!("BLOCKED: {why}")),
            Pin::Pinned => build_receipt(plan, lane, driver, &records),
        };
        receipts.insert(lane.engine.clone(), built);
    }

    let mut refusals = Vec::new();
    let mut ok = Vec::new();
    for lane in &plan.lanes {
        match &receipts[&lane.engine] {
            Ok(r) => ok.push((lane.engine.clone(), r.clone())),
            Err(e) => refusals.push(format!("{}: {e}", lane.engine)),
        }
    }
    let comparison = if refusals.is_empty() {
        compare(&ok)
    } else {
        Err(refusals)
    };
    Bundle {
        plan: plan.clone(),
        pins,
        driver,
        order,
        records,
        receipts,
        comparison,
    }
}

fn build_receipt(
    plan: &Plan,
    lane: &LanePlan,
    driver: DriverStatus,
    records: &[RunRecord],
) -> Result<Receipt, String> {
    let mine: Vec<&RunRecord> = records.iter().filter(|r| r.lane == lane.engine).collect();
    let measured: Vec<&RunRecord> = mine
        .iter()
        .copied()
        .filter(|r| r.phase == RunPhase::Measured)
        .collect();
    let failed: Vec<String> = measured
        .iter()
        .filter_map(|r| {
            r.metrics
                .as_ref()
                .err()
                .map(|e| format!("run {}: {e}", r.index))
        })
        .collect();
    if !failed.is_empty() {
        return Err(format!(
            "FAILED runs kept in the bundle: {}",
            failed.join("; ")
        ));
    }
    // Placement is identity: every completed run, warmup included, must report the pinned placement.
    let drift: Vec<String> = mine
        .iter()
        .filter(|r| r.output.as_ref().is_ok_and(|o| o.exit_ok))
        .filter(|r| r.placement.as_deref() != Some(lane.expected_placement.as_str()))
        .map(|r| {
            format!(
                "{} {} reported {}",
                phase_name(r.phase),
                r.index,
                r.placement.as_deref().unwrap_or("no placement")
            )
        })
        .collect();
    if !drift.is_empty() {
        return Err(format!(
            "PLACEMENT: plan pins {:?}; {}",
            lane.expected_placement,
            drift.join("; ")
        ));
    }
    let mut unknowns = BTreeMap::new();
    unknowns.insert(
        "ambient_vram_mib".to_string(),
        "not measured by the lane driver".to_string(),
    );
    unknowns.insert(
        "reclaim_delta_mib".to_string(),
        "not measured by the lane driver".to_string(),
    );
    let mut samples = Vec::new();
    for (i, r) in measured.iter().enumerate() {
        let m = r.metrics.as_ref().expect("checked above");
        for (name, v) in [
            ("prefill_tok_s", m.prefill_tok_s),
            ("ttft_ms", m.ttft_ms),
            ("tpot_ms", m.tpot_ms),
            ("vram_used_mib", m.vram_used_mib),
            ("host_rss_peak_mib", m.host_rss_peak_mib),
        ] {
            if v.is_none() {
                unknowns.insert(
                    format!("samples[{i}].{name}"),
                    "lane did not report it".to_string(),
                );
            }
        }
        samples.push(Sample {
            decode_tok_s: m.decode_tok_s,
            prefill_tok_s: m.prefill_tok_s,
            ttft_ms: m.ttft_ms,
            tpot_ms: m.tpot_ms,
            vram_used_mib: m.vram_used_mib,
            host_rss_peak_mib: m.host_rss_peak_mib,
            exit_status: ExitStatus::Ok,
        });
    }
    let decode: Vec<f64> = samples.iter().map(|s| s.decode_tok_s).collect();
    let stats = statistics(&decode).ok_or("no measured samples")?;
    let parities: Vec<Parity> = measured.iter().map(|r| r.parity).collect();
    let semantic_parity = match lane.parity_class {
        ParityClass::SelfOnly => Parity::NotApplicable,
        ParityClass::Exact if parities.iter().all(|p| *p == Parity::Pass) => Parity::Pass,
        ParityClass::Exact if parities.contains(&Parity::Fail) => Parity::Fail,
        ParityClass::Exact => Parity::Unknown,
    };
    let has_reference = mine
        .iter()
        .any(|r| r.phase == RunPhase::Warmup && r.self_repro == SelfRepro::Reference);
    let repro: Vec<SelfRepro> = measured.iter().map(|r| r.self_repro).collect();
    let self_reproducibility = if repro.contains(&SelfRepro::Fail) {
        Parity::Fail
    } else if has_reference && repro.iter().all(|r| *r == SelfRepro::Pass) {
        Parity::Pass
    } else {
        Parity::Unknown
    };
    let receipt = Receipt {
        schema: SCHEMA_ID.to_string(),
        run_id: plan.run_id.clone(),
        date: plan.date.clone(),
        protocol: plan.protocol.clone(),
        lane: LaneIdentity {
            engine: lane.engine.clone(),
            binary_path: lane.binary_path.clone(),
            binary_sha256: lane.expected_binary_sha256.clone(),
            model_path: lane.model_path.clone(),
            model_sha256: lane.expected_model_sha256.clone(),
            host_fingerprint: lane.host_fingerprint.clone(),
            driver_status: driver,
            placement: Some(lane.expected_placement.clone()),
            parity_class: lane.parity_class,
        },
        warmup_runs: plan.warmup_runs,
        samples,
        ambient_vram_mib: None,
        reclaim_delta_mib: None,
        semantic_parity,
        self_reproducibility,
        derived: Derived {
            sample_count: decode.len() as u64,
            median_decode_tok_s: stats.median,
            min_decode_tok_s: stats.min,
            max_decode_tok_s: stats.max,
            p95_decode_tok_s: stats.p95,
        },
        unknowns,
    };
    let errors = receipt::validate(&receipt, None);
    if errors.is_empty() {
        Ok(receipt)
    } else {
        Err(format!("receipt does not validate: {}", errors.join("; ")))
    }
}

fn phase_name(p: RunPhase) -> &'static str {
    match p {
        RunPhase::Warmup => "warmup",
        RunPhase::Measured => "measured",
    }
}

/// The run bundle as JSON: plan identity, pins, order and every run with raw output.
pub fn bundle_json(b: &Bundle) -> Json {
    let s = |v: &str| Json::Str(v.to_string());
    let runs = b
        .records
        .iter()
        .map(|r| {
            let (exit, stdout, stderr, wall) = match &r.output {
                Ok(o) => (
                    s(if o.exit_ok { "OK" } else { "NONZERO" }),
                    s(&o.stdout),
                    s(&o.stderr),
                    Json::Num(o.wall_ms),
                ),
                Err(e) => (s("SPAWN_ERROR"), Json::Null, s(e), Json::Null),
            };
            Json::Obj(vec![
                ("lane".into(), s(&r.lane)),
                ("phase".into(), s(phase_name(r.phase))),
                ("index".into(), Json::Num(r.index as f64)),
                ("exit".into(), exit),
                ("wall_ms".into(), wall),
                (
                    "metrics_error".into(),
                    r.metrics.as_ref().err().map_or(Json::Null, |e| s(e)),
                ),
                ("parity".into(), s(r.parity.as_str())),
                (
                    "placement".into(),
                    r.placement.as_deref().map_or(Json::Null, s),
                ),
                ("self_reproducibility".into(), s(r.self_repro.as_str())),
                ("stdout".into(), stdout),
                ("stderr".into(), stderr),
            ])
        })
        .collect();
    Json::Obj(vec![
        ("schema".into(), s("gnostral.rs/engine-benchmark-bundle/v1")),
        ("run_id".into(), s(&b.plan.run_id)),
        ("protocol_sha256".into(), s(&b.plan.protocol.sha256)),
        ("driver_status".into(), s(b.driver.as_str())),
        (
            "pins".into(),
            Json::Obj(
                b.pins
                    .iter()
                    .map(|(k, v)| {
                        (
                            k.clone(),
                            match v {
                                Pin::Pinned => s("PINNED"),
                                Pin::Blocked(why) => s(&format!("BLOCKED: {why}")),
                            },
                        )
                    })
                    .collect(),
            ),
        ),
        (
            "order".into(),
            Json::Arr(
                b.order
                    .iter()
                    .map(|(l, p, i)| Json::Arr(vec![s(l), s(phase_name(*p)), Json::Num(*i as f64)]))
                    .collect(),
            ),
        ),
        ("runs".into(), Json::Arr(runs)),
        (
            "comparison".into(),
            match &b.comparison {
                Ok(_) => s("COMPARED"),
                Err(reasons) => Json::Arr(reasons.iter().map(|r| s(r)).collect()),
            },
        ),
    ])
}
