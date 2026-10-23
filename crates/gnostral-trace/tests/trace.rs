use gnostral_trace::{measure_overhead, Identity, ManualClock, MonotonicClock, Phase, Recorder};

fn identity() -> Identity {
    Identity {
        binary_sha256: Some("abc".into()),
        model_sha256: None,
        command: "gnostral serve --trace".into(),
    }
}

fn traced(clock: &ManualClock, enabled: bool) -> Recorder<'_> {
    let mut r = Recorder::new(clock, enabled, 16);
    r.begin_request();
    for (phase, ns) in [
        (Phase::Schedule, 10),
        (Phase::Forward, 500),
        (Phase::CpuExpert, 200),
        (Phase::Sample, 20),
    ] {
        let s = r.start();
        clock.advance(ns);
        r.stop(phase, s);
        clock.advance(5); // untraced work between phases
    }
    r.end_request();
    r
}

#[test]
fn every_phase_reports_a_duration_or_unknown_and_gaps_are_reconciled() {
    let clock = ManualClock::default();
    let t = traced(&clock, true).timeline(identity());
    assert_eq!(t.phases.len(), Phase::ALL.len());
    let get = |name: &str| t.phases.iter().find(|p| p.phase == name).unwrap().total_ns;
    assert_eq!(get("forward"), Some(500));
    assert_eq!(get("upload"), None);
    assert_eq!(get("post_process"), None);
    assert_eq!(t.wall_ns, Some(750));
    assert_eq!(t.attributed_ns, 730);
    assert_eq!(t.unknown_ns, Some(20));
    assert_eq!(t.overlap_ns, 0);
    let s = t.summary();
    assert!(s.contains("UNKNOWN") && s.contains("unattributed"));
}

#[test]
fn disabled_recorder_records_nothing() {
    let clock = ManualClock::default();
    let r = traced(&clock, false);
    assert!(r.spans().is_empty());
    let t = r.timeline(identity());
    assert_eq!(t.wall_ns, None);
    assert!(t.phases.iter().all(|p| p.total_ns.is_none()));
    assert!(!Recorder::new(&clock, false, 8).enabled());
}

#[test]
fn full_buffer_drops_instead_of_allocating() {
    let clock = ManualClock::default();
    let mut r = Recorder::new(&clock, true, 2);
    let cap = r.spans().as_ptr();
    for _ in 0..5 {
        let s = r.start();
        clock.advance(1);
        r.stop(Phase::Forward, s);
    }
    assert_eq!(r.spans().len(), 2);
    assert_eq!(r.spans().as_ptr(), cap);
    assert_eq!(r.timeline(identity()).dropped_spans, 3);
}

#[test]
fn overlapping_spans_are_reported_not_double_counted() {
    let clock = ManualClock::default();
    let mut r = Recorder::new(&clock, true, 8);
    r.begin_request();
    let outer = r.start();
    clock.advance(10);
    let inner = r.start();
    clock.advance(30);
    r.stop(Phase::CpuExpert, inner);
    clock.advance(10);
    r.stop(Phase::Forward, outer);
    r.end_request();
    let t = r.timeline(identity());
    assert_eq!(t.wall_ns, Some(50));
    assert_eq!(t.overlap_ns, 30);
    assert_eq!(t.unknown_ns, Some(0));
}

#[test]
fn json_receipt_is_deterministic_and_marks_unknown() {
    let a = traced(&ManualClock::default(), true)
        .timeline(identity())
        .to_json();
    let b = traced(&ManualClock::default(), true)
        .timeline(identity())
        .to_json();
    assert_eq!(a, b);
    assert!(a.starts_with("{\"schema\":\"gnostral-trace/1\""));
    assert!(a.contains("\"model_sha256\":\"UNKNOWN\""));
    assert!(a.contains("\"phase\":\"upload\",\"count\":0,\"total_ns\":\"UNKNOWN\""));
    assert!(a.contains("\"overhead_ns_per_span\":\"UNKNOWN\""));
    assert!(a.contains("[\"forward\",15,515]"));
}

#[test]
fn overhead_is_measured_on_the_real_clock() {
    let clock = MonotonicClock::default();
    let ns = measure_overhead(&clock, 10_000);
    // A measurement, not an assertion of negligibility: it must exist and be finite.
    assert!(ns < 1_000_000);
    let mut t = Recorder::new(&clock, true, 1).timeline(identity());
    t.overhead_ns_per_span = Some(ns);
    assert!(t
        .to_json()
        .contains(&format!("\"overhead_ns_per_span\":{ns}")));
}
