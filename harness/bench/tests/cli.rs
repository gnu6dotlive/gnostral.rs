//! End-to-end CLI behaviour against synthetic receipts.

mod common;

use std::process::{Command, Output};

use common::{write_file, write_receipt, Fixture, BIN_A, BIN_B, BIN_C, MODEL, MODEL_ALT};

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_gnostral-bench"))
        .args(args)
        .output()
        .expect("run gnostral-bench")
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn assert_valid(path: &str) -> String {
    let output = run(&["validate", path]);
    assert!(
        output.status.success(),
        "expected valid: {}",
        stderr(&output)
    );
    assert_eq!(output.status.code(), Some(0));
    stdout(&output)
}

fn assert_invalid(path: &str, needle: &str) -> String {
    let output = run(&["validate", path]);
    assert_eq!(
        output.status.code(),
        Some(1),
        "expected exit 1, stdout: {} stderr: {}",
        stdout(&output),
        stderr(&output)
    );
    let message = stderr(&output);
    assert!(
        message.contains(needle),
        "expected {needle:?} in diagnostics: {message}"
    );
    message
}

#[test]
fn validates_a_complete_receipt() {
    let body = Fixture::new("llama.cpp", BIN_A, MODEL).render();
    let path = write_receipt(&body);
    let output = assert_valid(&path);
    assert!(output.contains("VALID"));
    assert!(output.contains("median_decode_tok_s=34.500"), "{output}");
    assert!(output.contains("samples=3"));
}

#[test]
fn validate_accepts_unknown_vram_with_a_reason() {
    let body = Fixture::new("llama.cpp", BIN_A, MODEL)
        .with_vram(1, None)
        .render();
    let path = write_receipt(&body);
    assert_valid(&path);
}

#[test]
fn validate_rejects_null_without_an_unknowns_entry() {
    let body = Fixture::new("llama.cpp", BIN_A, MODEL)
        .with_vram(0, None)
        .render()
        .replace(
            "\"samples[0].vram_used_mib\": \"no whole-card sampler attached in this synthetic fixture\"",
            "",
        );
    let path = write_receipt(&body);
    assert_invalid(&path, "has no entry in unknowns");
}

#[test]
fn validate_rejects_an_unknowns_reason_for_a_known_field() {
    let mut fixture = Fixture::new("llama.cpp", BIN_A, MODEL);
    fixture
        .extra_unknowns
        .push(("lane.engine".into(), "not actually unknown".into()));
    let path = write_receipt(&fixture.render());
    assert_invalid(&path, "reason given for a field that is not UNKNOWN");
}

#[test]
fn validate_rejects_an_unpinned_identity() {
    let mut fixture = Fixture::new("llama.cpp", BIN_A, MODEL);
    fixture.binary_sha256 = String::new();
    let path = write_receipt(&fixture.render());
    assert_invalid(&path, "lane.binary_sha256");
}

#[test]
fn validate_rejects_warmup_below_one() {
    let mut fixture = Fixture::new("llama.cpp", BIN_A, MODEL);
    fixture.warmup_runs = 0;
    let path = write_receipt(&fixture.render());
    assert_invalid(&path, "warmup_runs");
}

#[test]
fn validate_rejects_fewer_than_three_repetitions() {
    let mut fixture = Fixture::new("llama.cpp", BIN_A, MODEL).with_decodes(&[34.0, 34.5]);
    fixture.repetitions = 2;
    let path = write_receipt(&fixture.render());
    assert_invalid(&path, "protocol.repetitions");
}

#[test]
fn validate_rejects_a_sample_count_that_differs_from_the_protocol() {
    let fixture = Fixture::new("llama.cpp", BIN_A, MODEL).with_decodes(&[34.0, 34.5]);
    let path = write_receipt(&fixture.render());
    assert_invalid(
        &path,
        "protocol.repetitions requires 3 measured samples, found 2",
    );
}

#[test]
fn validate_rejects_a_non_positive_sample() {
    let fixture = Fixture::new("llama.cpp", BIN_A, MODEL).with_decodes(&[0.0, 34.0, 35.0]);
    let path = write_receipt(&fixture.render());
    assert_invalid(&path, "samples[0].decode_tok_s");
}

#[test]
fn validate_rejects_a_non_finite_sample() {
    let mut fixture = Fixture::new("llama.cpp", BIN_A, MODEL);
    fixture.samples[0].decode_token = Some("1e999".into());
    let path = write_receipt(&fixture.render());
    assert_invalid(&path, "finite");
}

#[test]
fn validate_rejects_a_tampered_median() {
    let mut fixture = Fixture::new("llama.cpp", BIN_A, MODEL);
    fixture.median_override = Some("99.0".into());
    let path = write_receipt(&fixture.render());
    let message = assert_invalid(&path, "derived.median_decode_tok_s");
    assert!(message.contains("recomputed"), "{message}");
}

#[test]
fn validate_rejects_a_protocol_hash_mismatch() {
    let body = Fixture::new("llama.cpp", BIN_A, MODEL).render();
    let path = write_receipt(&body);
    let expected = "b".repeat(64);
    let output = run(&["validate", &path, "--protocol-hash", &expected]);
    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).contains("protocol.sha256"));
}

#[test]
fn protocol_file_pins_the_exact_protocol_bytes() {
    let protocol_bytes = b"Q-025 frozen protocol, synthetic fixture\n";
    let protocol_path = write_file("protocol.txt", "Q-025 frozen protocol, synthetic fixture\n");
    let mut fixture = Fixture::new("llama.cpp", BIN_A, MODEL);
    fixture.protocol_sha256 =
        gnostral_bench::sha256::hex(&gnostral_bench::sha256::sha256(protocol_bytes));
    let path = write_receipt(&fixture.render());

    let output = run(&["validate", &path, "--protocol-file", &protocol_path]);
    assert!(
        output.status.success(),
        "expected protocol pin to pass: {}",
        stderr(&output)
    );

    std::fs::write(&protocol_path, "Q-025 frozen protocol, edited\n").expect("edit protocol");
    let output = run(&["validate", &path, "--protocol-file", &protocol_path]);
    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).contains("protocol.sha256"));
}

#[test]
fn validate_rejects_malformed_json() {
    let path = write_file("broken.json", "{\"schema\": \"x\",}");
    let output = run(&["validate", &path]);
    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).contains("malformed JSON"));
}

#[test]
fn validate_rejects_duplicate_keys() {
    let path = write_file("duplicate.json", "{\"run_id\": \"a\", \"run_id\": \"b\"}");
    let output = run(&["validate", &path]);
    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).contains("duplicate object key"));
}

fn lane(engine: &str, binary: &str, decodes: &[f64], vram: f64) -> Fixture {
    let mut fixture = Fixture::new(engine, binary, MODEL).with_decodes(decodes);
    for index in 0..fixture.samples.len() {
        fixture.samples[index].vram = Some(vram);
    }
    fixture
}

#[test]
fn compare_renders_a_three_lane_pareto_table() {
    let a = write_receipt(&lane("llama.cpp", BIN_A, &[34.0, 34.5, 35.0], 7400.0).render());
    let b = write_receipt(&lane("strata", BIN_B, &[9.4, 9.5, 9.6], 7200.0).render());
    let c = write_receipt(&lane("candidate", BIN_C, &[20.0, 20.2, 20.4], 7100.0).render());
    let output = run(&["compare", &a, &b, &c]);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let report = stdout(&output);
    assert!(
        report.contains("Baseline (first lane): llama.cpp"),
        "{report}"
    );
    assert!(report.contains("llama.cpp: median 34.500 tok/s over 3 samples"));
    assert!(report.contains("strata: median 9.500 tok/s over 3 samples"));
    assert!(report.contains("candidate: median 20.200 tok/s over 3 samples"));
    assert!(
        report.contains("| llama.cpp | exact | 0-47:cuda:0;nm=cuda:0 | 34.500 | 34.000-35.000 | 3 | baseline | 7400.0 |"),
        "{report}"
    );
    assert!(
        report.contains(
            "| strata | exact | 0-47:cuda:0;nm=cuda:0 | 9.500 | 9.400-9.600 | 3 | BELOW |"
        ),
        "{report}"
    );
    assert!(report.contains("1.000x"), "{report}");
    assert!(report.contains("0.586x"), "{report}");
    assert!(report.contains("FRONT"), "{report}");
    assert!(report.contains("DOMINATED"), "{report}");
}

#[test]
fn compare_refuses_a_shared_model_mismatch() {
    let a = write_receipt(&lane("llama.cpp", BIN_A, &[34.0, 34.5, 35.0], 7400.0).render());
    let mut other = lane("candidate", BIN_C, &[20.0, 20.2, 20.4], 7100.0);
    other.model_sha256 = MODEL_ALT.into();
    let c = write_receipt(&other.render());
    let output = run(&["compare", &a, &c]);
    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).contains("lane.model_sha256"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn compare_refuses_a_sampling_mismatch() {
    let a = write_receipt(&lane("llama.cpp", BIN_A, &[34.0, 34.5, 35.0], 7400.0).render());
    let mut other = lane("candidate", BIN_C, &[20.0, 20.2, 20.4], 7100.0);
    other.n_predict = 512;
    let c = write_receipt(&other.render());
    let output = run(&["compare", &a, &c]);
    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).contains("protocol.sampling.n_predict"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn compare_refuses_an_unpinned_lane() {
    let a = write_receipt(&lane("llama.cpp", BIN_A, &[34.0, 34.5, 35.0], 7400.0).render());
    let mut other = lane("candidate", BIN_C, &[20.0, 20.2, 20.4], 7100.0);
    other.binary_sha256 = "not-a-hash".into();
    let c = write_receipt(&other.render());
    let output = run(&["compare", &a, &c]);
    assert_eq!(output.status.code(), Some(1));
    let message = stderr(&output);
    assert!(message.contains("REFUSED"), "{message}");
    assert!(message.contains("lane.binary_sha256"), "{message}");
}

#[test]
fn compare_prints_unknown_instead_of_a_ratio_for_unknown_vram() {
    let a = write_receipt(&lane("llama.cpp", BIN_A, &[34.0, 34.5, 35.0], 7400.0).render());
    let mut other = lane("candidate", BIN_C, &[20.0, 20.2, 20.4], 7100.0);
    other.samples[1].vram = None;
    let c = write_receipt(&other.render());
    let output = run(&["compare", &a, &c]);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let report = stdout(&output);
    assert!(report.contains("UNKNOWN"), "{report}");
    assert!(!report.to_lowercase().contains("nan"), "{report}");
    assert!(!report.to_lowercase().contains("inf"), "{report}");
    assert!(
        report.contains("0.586x"),
        "speed ratio must still print: {report}"
    );
}

#[test]
fn compare_refuses_a_lane_without_semantic_parity() {
    let a = write_receipt(&lane("llama.cpp", BIN_A, &[34.0, 34.5, 35.0], 7400.0).render());
    let mut other = lane("candidate", BIN_C, &[20.0, 20.2, 20.4], 7100.0);
    other.parity = "FAIL".into();
    let c = write_receipt(&other.render());
    let output = run(&["compare", &a, &c]);
    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).contains("semantic parity is FAIL"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn usage_errors_exit_two() {
    let output = run(&["compare"]);
    assert_eq!(output.status.code(), Some(2));
    let output = run(&["validate"]);
    assert_eq!(output.status.code(), Some(2));
    let output = run(&["nonsense"]);
    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn validate_rejects_a_self_class_lane_that_claims_token_parity() {
    let mut fixture = Fixture::new("llama.cpp", BIN_A, MODEL);
    fixture.parity_class = "self".into();
    let path = write_receipt(&fixture.render());
    assert_invalid(&path, "expected NOT_APPLICABLE");
    fixture.parity = "NOT_APPLICABLE".into();
    assert_valid(&write_receipt(&fixture.render()));
}

#[test]
fn validate_rejects_an_exact_lane_without_a_parity_verdict() {
    let mut fixture = Fixture::new("strata", BIN_B, MODEL);
    fixture.parity = "NOT_APPLICABLE".into();
    assert_invalid(&write_receipt(&fixture.render()), "exact-class lane");
}

#[test]
fn validate_requires_unknown_placement_to_be_explained() {
    let mut fixture = Fixture::new("strata", BIN_B, MODEL);
    fixture.placement = None;
    assert_invalid(&write_receipt(&fixture.render()), "lane.placement");
    fixture.extra_unknowns.push((
        "lane.placement".into(),
        "engine built without the placement report".into(),
    ));
    assert_valid(&write_receipt(&fixture.render()));
}

#[test]
fn compare_refuses_unknown_placement_and_unreproducible_lanes() {
    let a = write_receipt(&lane("llama.cpp", BIN_A, &[34.0, 34.5, 35.0], 7400.0).render());
    let mut unplaced = lane("strata", BIN_B, &[9.4, 9.5, 9.6], 7200.0);
    unplaced.placement = None;
    unplaced
        .extra_unknowns
        .push(("lane.placement".into(), "not reported".into()));
    let mut flaky = lane("candidate", BIN_C, &[20.0, 20.2, 20.4], 7100.0);
    flaky.self_repro = "FAIL".into();
    let output = run(&[
        "compare",
        &a,
        &write_receipt(&unplaced.render()),
        &write_receipt(&flaky.render()),
    ]);
    assert_eq!(output.status.code(), Some(1));
    let message = stderr(&output);
    // The CLI labels lanes by receipt path.
    assert!(message.contains("layer placement is UNKNOWN"), "{message}");
    assert!(
        message.contains("self-reproducibility is FAIL"),
        "{message}"
    );
}

#[test]
fn compare_marks_overlapping_ranges_instead_of_ranking_them() {
    let a = write_receipt(&lane("per-route", BIN_A, &[112.0, 125.0, 127.0], 3000.0).render());
    let b = write_receipt(&lane("grouped", BIN_B, &[97.0, 113.0, 131.0], 3000.0).render());
    let c = write_receipt(&lane("gnostral", BIN_C, &[227.0, 259.0, 273.0], 3000.0).render());
    let output = run(&["compare", &a, &b, &c]);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let report = stdout(&output);
    assert!(
        report.contains(
            "| grouped | exact | 0-47:cuda:0;nm=cuda:0 | 113.000 | 97.000-131.000 | 3 | OVERLAP |"
        ),
        "{report}"
    );
    assert!(
        report.contains(
            "| gnostral | exact | 0-47:cuda:0;nm=cuda:0 | 259.000 | 227.000-273.000 | 3 | ABOVE |"
        ),
        "{report}"
    );
}
