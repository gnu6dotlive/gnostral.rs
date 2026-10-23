use gnostral_memory_plan::kv::KvLayout;
use gnostral_scheduler::{
    percentile_nearest_rank, simulate, Class, CostModel, Outcome, Policy, Rejection, Request,
    Scheduler, SchedulerConfig,
};

fn layout() -> KvLayout {
    KvLayout {
        layers: 2,
        kv_heads: 1,
        head_dim: 8,
        elem_bytes: 2,
        block_tokens: 16,
    }
}

fn cost() -> CostModel {
    CostModel {
        prefill_us_per_token: 100,
        decode_base_us: 20_000,
        decode_us_per_seq: 2_000,
    }
}

fn config(policy: Policy, max_running: usize) -> SchedulerConfig {
    SchedulerConfig {
        policy,
        max_running,
        max_step_tokens: 512,
        kv_watermark_blocks: 1,
    }
}

fn req(id: u64, class: Class, arrival_us: u64, prompt: u64, out: u64) -> Request {
    Request {
        id,
        class,
        arrival_us,
        prompt_tokens: prompt,
        max_new_tokens: out,
        ttft_deadline_us: None,
    }
}

fn workload(n: u64) -> Vec<Request> {
    (0..n)
        .map(|i| {
            req(
                i,
                if i % 3 == 0 {
                    Class::Interactive
                } else {
                    Class::Standard
                },
                i * 7_000,
                64 + (i * 37) % 300,
                16 + (i * 11) % 48,
            )
        })
        .collect()
}

#[test]
fn simulation_is_deterministic_and_complete() {
    let w = workload(40);
    let a = simulate(
        config(Policy::Slo { aging_us: 200_000 }, 4),
        cost(),
        layout(),
        256,
        &w,
        1_000_000,
    )
    .unwrap();
    let b = simulate(
        config(Policy::Slo { aging_us: 200_000 }, 4),
        cost(),
        layout(),
        256,
        &w,
        1_000_000,
    )
    .unwrap();
    assert_eq!(a, b);
    assert_eq!(a.unfinished, 0);
    assert!(a.records.iter().all(|r| r.outcome == Outcome::Completed));
    let expected: u64 = w.iter().map(|r| r.max_new_tokens).sum();
    assert_eq!(a.generated_tokens, expected);
    // Every completed request has its own samples, not just an aggregate.
    assert!(a
        .records
        .iter()
        .all(|r| r.ttft_us.is_some() && r.queue_wait_us.is_some()));
}

#[test]
fn background_work_is_not_starved_under_slo_with_aging() {
    // Interactive arrivals outpace service, so without aging the background request waits for all of them.
    let mut w: Vec<Request> = (0..200)
        .map(|i| req(i, Class::Interactive, i * 30_000, 400, 8))
        .collect();
    w.push(req(1_000, Class::Background, 0, 400, 8));
    let bg_ttft_and_max_interactive = |aging_us| {
        let r = simulate(
            config(Policy::Slo { aging_us }, 1),
            cost(),
            layout(),
            128,
            &w,
            100_000,
        )
        .unwrap();
        assert_eq!(r.unfinished, 0);
        let bg = r
            .records
            .iter()
            .find(|x| x.id == 1_000)
            .unwrap()
            .ttft_us
            .unwrap();
        let interactive_first_tokens: Vec<u64> = r
            .records
            .iter()
            .filter(|x| x.class == Class::Interactive)
            .map(|x| x.ttft_us.unwrap() + w[x.id as usize].arrival_us)
            .collect();
        (bg, *interactive_first_tokens.iter().max().unwrap())
    };
    let (starved, last_interactive) = bg_ttft_and_max_interactive(u64::MAX);
    assert!(
        starved >= last_interactive,
        "no aging: background goes last"
    );
    let (aged, last_interactive) = bg_ttft_and_max_interactive(300_000);
    assert!(
        aged < last_interactive,
        "aging lets background overtake queued interactive work"
    );
}

#[test]
fn fifo_completes_equal_requests_in_arrival_order() {
    let w: Vec<Request> = (0..6).map(|i| req(i, Class::Standard, i, 32, 4)).collect();
    let mut s = Scheduler::new(config(Policy::Fifo, 1), Some(cost()), layout(), 64).unwrap();
    for r in &w {
        s.submit(*r).unwrap();
    }
    let mut now = 10;
    while !s.is_idle() {
        let b = s.schedule(now);
        now += cost()
            .step_us(b.prefill_tokens(), b.decode.len() as u64)
            .max(1);
        s.complete(&b, now);
    }
    let order: Vec<u64> = s.finished().iter().map(|f| f.request.id).collect();
    assert_eq!(order, vec![0, 1, 2, 3, 4, 5]);
    assert_eq!(s.kv().free_blocks(), 64);
    s.kv().check_invariants().unwrap();
}

#[test]
fn deadlines_are_rejected_explicitly_with_the_estimate() {
    let mut s = Scheduler::new(
        config(Policy::Slo { aging_us: 1 }, 4),
        Some(cost()),
        layout(),
        1024,
    )
    .unwrap();
    for i in 0..10 {
        s.submit(req(i, Class::Interactive, 0, 2_000, 4)).unwrap();
    }
    let mut late = req(99, Class::Interactive, 0, 100, 4);
    late.ttft_deadline_us = Some(50_000);
    match s.submit(late) {
        Err(Rejection::DeadlineUnmeetable {
            estimate_us,
            deadline_us,
        }) => {
            assert!(estimate_us > deadline_us);
            assert_eq!(deadline_us, 50_000);
        }
        other => panic!("{other:?}"),
    }
    let mut no_cost =
        Scheduler::new(config(Policy::Slo { aging_us: 1 }, 4), None, layout(), 64).unwrap();
    assert_eq!(no_cost.submit(late), Err(Rejection::CostModelUnknown));
    assert_eq!(
        s.submit(req(0, Class::Standard, 0, 1, 1)),
        Err(Rejection::DuplicateId)
    );
    assert!(matches!(
        s.submit(req(500, Class::Standard, 0, 100_000, 1)),
        Err(Rejection::TooLargeForKv { .. })
    ));
}

#[test]
fn kv_pressure_preempts_by_recompute_and_still_completes() {
    let w: Vec<Request> = (0..6).map(|i| req(i, Class::Standard, 0, 48, 64)).collect();
    let r = simulate(config(Policy::Fifo, 6), cost(), layout(), 16, &w, 100_000).unwrap();
    assert_eq!(r.unfinished, 0);
    assert!(r.records.iter().any(|x| x.preemptions > 0));
    assert!(r.records.iter().all(|x| x.generated == 64));
}

#[test]
fn batching_raises_modelled_aggregate_throughput() {
    // Under the linear cost model only; the real C4/C1 ratio is the M2 measurement.
    let w: Vec<Request> = (0..8)
        .map(|i| req(i, Class::Standard, 0, 128, 64))
        .collect();
    let c1 = simulate(
        config(Policy::Fifo, 1),
        cost(),
        layout(),
        512,
        &w,
        1_000_000,
    )
    .unwrap();
    let c4 = simulate(
        config(Policy::Fifo, 4),
        cost(),
        layout(),
        512,
        &w,
        1_000_000,
    )
    .unwrap();
    assert!(c4.throughput_tok_s() > 1.5 * c1.throughput_tok_s());
    let tpot =
        |r: &gnostral_scheduler::Receipt| r.per_class[&Class::Standard].tpot_p50_p95_p99.unwrap().1;
    assert!(tpot(&c4) <= tpot(&c1) * 3 / 2);
}

#[test]
fn nearest_rank_percentiles() {
    let s: Vec<u64> = (1..=100).collect();
    assert_eq!(percentile_nearest_rank(&s, 50), Some(50));
    assert_eq!(percentile_nearest_rank(&s, 95), Some(95));
    assert_eq!(percentile_nearest_rank(&s, 99), Some(99));
    assert_eq!(percentile_nearest_rank(&[7], 99), Some(7));
    assert_eq!(percentile_nearest_rank(&[], 50), None);
}
