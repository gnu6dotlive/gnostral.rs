//! Q-031 lane driver against synthetic lanes. Every number here is a fixture.

use std::collections::BTreeMap;

use gnostral_bench::json;
use gnostral_bench::receipt::{self, DriverStatus, Parity, ProtocolId, Sampling};
use gnostral_bench::run::{
    bundle_json, decode_plan, execute, frozen_order, parse_metrics, Executor, LanePlan, Output,
    Pin, Plan, RunPhase,
};

const H1: &str = "1111111111111111111111111111111111111111111111111111111111111111";
const H2: &str = "2222222222222222222222222222222222222222222222222222222222222222";
const HM: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

struct Fake {
    hashes: BTreeMap<String, String>,
    calls: Vec<String>,
    decode: BTreeMap<String, f64>,
    fail: Option<(String, usize)>,
    driver: &'static str,
}

impl Executor for Fake {
    fn run(&mut self, argv: &[String]) -> Result<Output, String> {
        let lane = argv[0].clone();
        self.calls.push(lane.clone());
        if lane == "probe" {
            return Ok(Output {
                exit_ok: true,
                stdout: format!("GNOSTRAL_DRIVER {}\n", self.driver),
                stderr: String::new(),
                wall_ms: 1.0,
            });
        }
        let n = self.calls.iter().filter(|c| **c == lane).count();
        if self.fail.as_ref() == Some(&(lane.clone(), n)) {
            return Ok(Output {
                exit_ok: false,
                stdout: String::new(),
                stderr: "boom".into(),
                wall_ms: 2.0,
            });
        }
        let d = self.decode[&lane] + n as f64;
        Ok(Output {
            exit_ok: true,
            stdout: format!("log line\nGNOSTRAL_METRICS {{\"decode_tok_s\":{d},\"ttft_ms\":120.5,\"vram_used_mib\":null}}\nGNOSTRAL_PARITY PASS\n"),
            stderr: String::new(),
            wall_ms: 10.0,
        })
    }
    fn sha256_file(&mut self, path: &str) -> Result<String, String> {
        self.hashes
            .get(path)
            .cloned()
            .ok_or_else(|| format!("{path}: no such file"))
    }
}

fn lane(engine: &str, bin_hash: &str) -> LanePlan {
    LanePlan {
        engine: engine.into(),
        binary_path: format!("/opt/{engine}"),
        expected_binary_sha256: bin_hash.into(),
        model_path: "/models/m.gguf".into(),
        expected_model_sha256: HM.into(),
        host_fingerprint: "lab-3070".into(),
        argv: vec![engine.into(), "--bench".into()],
    }
}

fn plan(lanes: Vec<LanePlan>) -> Plan {
    Plan {
        run_id: "engine-perf-20261023".into(),
        date: "2026-10-23".into(),
        protocol: ProtocolId {
            id: "q025-frozen".into(),
            sha256: "a".repeat(64),
            repetitions: 3,
            context_tokens: 1024,
            prompt_sha256: "d".repeat(64),
            threads: 8,
            sampling: Sampling {
                temperature: 0.0,
                seed: 42,
                n_predict: 128,
                repeat_penalty: 1.0,
            },
        },
        warmup_runs: 1,
        lanes,
        driver_probe: Some(vec!["probe".into()]),
    }
}

fn fake() -> Fake {
    Fake {
        hashes: BTreeMap::from([
            ("/opt/llama".to_string(), H1.to_string()),
            ("/opt/strata".to_string(), H2.to_string()),
            ("/models/m.gguf".to_string(), HM.to_string()),
        ]),
        calls: Vec::new(),
        decode: BTreeMap::from([("llama".to_string(), 30.0), ("strata".to_string(), 20.0)]),
        fail: None,
        driver: "NO_XID",
    }
}

#[test]
fn order_is_warmup_then_interleaved_rounds_and_every_run_is_kept() {
    let p = plan(vec![lane("llama", H1), lane("strata", H2)]);
    let mut f = fake();
    let b = execute(&p, &mut f);
    let lanes: Vec<&str> = f
        .calls
        .iter()
        .filter(|c| *c != "probe")
        .map(String::as_str)
        .collect();
    assert_eq!(
        lanes,
        ["llama", "strata", "llama", "strata", "llama", "strata", "llama", "strata"]
    );
    assert_eq!(
        b.order,
        frozen_order(&p, &["llama".into(), "strata".into()])
    );
    assert_eq!(b.order[0].1, RunPhase::Warmup);
    assert_eq!(b.records.len(), 8);
    let r = b.receipts["llama"].as_ref().unwrap();
    // Warmup (call 1) is discarded; measured calls 2..4 are kept in order.
    let decode: Vec<f64> = r.samples.iter().map(|s| s.decode_tok_s).collect();
    assert_eq!(decode, [32.0, 33.0, 34.0]);
    assert_eq!(r.derived.median_decode_tok_s, 33.0);
    assert_eq!(r.lane.driver_status, DriverStatus::NoXid);
    assert_eq!(r.semantic_parity, Parity::Pass);
    assert!(r.unknowns.contains_key("samples[0].vram_used_mib"));
    assert!(r.unknowns.contains_key("samples[2].prefill_tok_s"));
    assert!(receipt::validate(r, None).is_empty());
    let table = b.comparison.as_ref().unwrap();
    assert!(table.contains("llama") && table.contains("strata"));
}

#[test]
fn unpinned_or_mismatched_lane_is_blocked_and_aborts_only_the_comparison() {
    let p = plan(vec![
        lane("llama", H1),
        lane("strata", H1),
        lane("other", "not-a-hash"),
    ]);
    let mut f = fake();
    f.decode.insert("other".into(), 1.0);
    let b = execute(&p, &mut f);
    assert_eq!(b.pins["llama"], Pin::Pinned);
    assert!(matches!(&b.pins["strata"], Pin::Blocked(why) if why.contains("!= pinned")));
    assert!(
        matches!(&b.pins["other"], Pin::Blocked(why) if why.contains("no valid expected sha256"))
    );
    assert!(
        f.calls.iter().all(|c| c != "strata" && c != "other"),
        "blocked lanes never run"
    );
    assert!(b.receipts["llama"].is_ok(), "the run itself continues");
    let refusals = b.comparison.unwrap_err();
    assert_eq!(refusals.len(), 2);
}

#[test]
fn a_failed_run_is_kept_and_fails_the_lane_not_the_bundle() {
    let p = plan(vec![lane("llama", H1), lane("strata", H2)]);
    let mut f = fake();
    f.fail = Some(("strata".into(), 3));
    let b = execute(&p, &mut f);
    assert_eq!(b.records.len(), 8);
    let failed = b
        .records
        .iter()
        .find(|r| r.lane == "strata" && r.phase == RunPhase::Measured && r.index == 1)
        .unwrap();
    assert!(failed.metrics.is_err());
    assert!(b.receipts["strata"]
        .as_ref()
        .unwrap_err()
        .contains("FAILED"));
    assert!(b.comparison.is_err());
    let text = json::write(&bundle_json(&b));
    assert!(text.contains("\"stderr\":\"boom\""));
    assert!(json::parse(&text).is_ok());
}

#[test]
fn unknown_driver_status_refuses_the_comparison() {
    let mut p = plan(vec![lane("llama", H1), lane("strata", H2)]);
    p.driver_probe = None;
    let b = execute(&p, &mut fake());
    assert_eq!(b.driver, DriverStatus::Unknown);
    assert!(b.receipts.values().all(|r| r.is_ok()));
    assert!(b
        .comparison
        .unwrap_err()
        .iter()
        .any(|r| r.contains("NO_XID")));
}

#[test]
fn receipts_round_trip_through_the_encoder() {
    let b = execute(
        &plan(vec![lane("llama", H1), lane("strata", H2)]),
        &mut fake(),
    );
    for r in b.receipts.values() {
        let r = r.as_ref().unwrap();
        let text = json::write(&receipt::encode(r));
        let back = receipt::decode(&json::parse(&text).unwrap()).unwrap();
        assert_eq!(&back, r);
    }
}

#[test]
fn metrics_line_is_strict() {
    assert!(parse_metrics("GNOSTRAL_METRICS {\"decode_tok_s\":1}").is_ok());
    assert!(parse_metrics("nothing").is_err());
    assert!(parse_metrics(
        "GNOSTRAL_METRICS {\"decode_tok_s\":1}\nGNOSTRAL_METRICS {\"decode_tok_s\":2}"
    )
    .is_err());
    assert!(parse_metrics("GNOSTRAL_METRICS {\"tpot_ms\":1}").is_err());
    assert!(parse_metrics("GNOSTRAL_METRICS {\"decode_tok_s\":1,\"made_up\":3}").is_err());
}

#[test]
fn plan_decoding_is_closed_and_needs_two_distinct_lanes() {
    let text = r#"{"schema":"gnostral.rs/engine-benchmark-plan/v1","run_id":"r","date":"2026-10-23",
      "protocol":{"id":"q025","sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","repetitions":3,
        "context_tokens":1024,"prompt_sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","threads":8,
        "sampling":{"temperature":0,"seed":1,"n_predict":64,"repeat_penalty":1}},
      "warmup_runs":1,
      "lanes":[
        {"engine":"a","binary_path":"/a","expected_binary_sha256":"x","model_path":"/m","expected_model_sha256":"y","host_fingerprint":"h","argv":["/a"]},
        {"engine":"b","binary_path":"/b","expected_binary_sha256":"x","model_path":"/m","expected_model_sha256":"y","host_fingerprint":"h","argv":["/b"]}]}"#;
    let p = decode_plan(&json::parse(text).unwrap()).unwrap();
    assert_eq!(p.lanes.len(), 2);
    assert_eq!(p.driver_probe, None);
    let dup = text.replace("\"engine\":\"b\"", "\"engine\":\"a\"");
    assert!(decode_plan(&json::parse(&dup).unwrap()).is_err());
    let extra = text.replace(
        "\"warmup_runs\":1",
        "\"warmup_runs\":1,\"median_only\":true",
    );
    assert!(decode_plan(&json::parse(&extra).unwrap()).is_err());
}

#[test]
fn cli_runs_real_processes_and_writes_bundle_receipts_and_table() {
    use std::process::Command;
    let dir = std::env::temp_dir().join(format!("gnostral-bench-run-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let hash = |bytes: &[u8]| gnostral_bench::sha256::hex(&gnostral_bench::sha256::sha256(bytes));
    let mut lanes = Vec::new();
    for (name, decode) in [("alpha", 12.5), ("beta", 10.0)] {
        let bin = dir.join(format!("{name}.bin"));
        std::fs::write(&bin, name.as_bytes()).unwrap();
        let script = format!(
            "echo 'GNOSTRAL_METRICS {{\"decode_tok_s\":{decode},\"ttft_ms\":50}}'; echo 'GNOSTRAL_PARITY PASS'"
        );
        lanes.push(format!(
            r#"{{"engine":"{name}","binary_path":"{}","expected_binary_sha256":"{}","model_path":"{}","expected_model_sha256":"{}","host_fingerprint":"ci","argv":["/bin/sh","-c",{}]}}"#,
            bin.display(),
            hash(name.as_bytes()),
            dir.join("m.gguf").display(),
            hash(b"model"),
            json::write(&json::Json::Str(script)),
        ));
    }
    std::fs::write(dir.join("m.gguf"), b"model").unwrap();
    let plan = format!(
        r#"{{"schema":"gnostral.rs/engine-benchmark-plan/v1","run_id":"cli","date":"2026-10-23",
        "protocol":{{"id":"q025","sha256":"{}","repetitions":3,"context_tokens":1024,"prompt_sha256":"{}","threads":8,
          "sampling":{{"temperature":0,"seed":1,"n_predict":64,"repeat_penalty":1}}}},
        "warmup_runs":1,"driver_probe":["/bin/sh","-c","echo GNOSTRAL_DRIVER NO_XID"],"lanes":[{}]}}"#,
        "a".repeat(64),
        "d".repeat(64),
        lanes.join(",")
    );
    let plan_path = dir.join("plan.json");
    std::fs::write(&plan_path, plan).unwrap();
    let out = dir.join("out");
    let status = Command::new(env!("CARGO_BIN_EXE_gnostral-bench"))
        .args([
            "run",
            plan_path.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        status.status.success(),
        "{}",
        String::from_utf8_lossy(&status.stderr)
    );
    for receipt_file in ["receipts/alpha.json", "receipts/beta.json"] {
        let v = Command::new(env!("CARGO_BIN_EXE_gnostral-bench"))
            .args(["validate", out.join(receipt_file).to_str().unwrap()])
            .output()
            .unwrap();
        assert!(v.status.success(), "{}", String::from_utf8_lossy(&v.stderr));
    }
    let bundle = std::fs::read_to_string(out.join("bundle.json")).unwrap();
    assert_eq!(bundle.matches("\"phase\":\"measured\"").count(), 6);
    assert!(std::fs::read_to_string(out.join("pareto.md"))
        .unwrap()
        .contains("alpha"));
    let _ = std::fs::remove_dir_all(&dir);
}
