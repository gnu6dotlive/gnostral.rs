use gnostral_expert_residency::dynamic::{
    Decision, DynamicCache, DynamicConfig, Refusal, HIT_SCORE,
};
use gnostral_expert_residency::{ExpertKey, ResidencyError};

fn key(layer: u32, expert: u32) -> ExpertKey {
    ExpertKey { layer, expert }
}

fn config(capacity: u64) -> DynamicConfig {
    DynamicConfig {
        capacity_bytes: capacity,
        half_life_steps: 8,
        admit_min_score: HIT_SCORE,
        hysteresis_num: 2,
        hysteresis_den: 1,
        max_upload_bytes_per_step: 0,
    }
}

fn cache(capacity: u64, experts: u32) -> DynamicCache {
    let sizes: Vec<_> = (0..experts).map(|e| (key(0, e), 100)).collect();
    DynamicCache::new(config(capacity), &sizes).unwrap()
}

/// Apply decisions as an engine would: every admitted upload succeeds.
fn settle(cache: &mut DynamicCache) -> Vec<Decision> {
    let decisions = cache.decide().unwrap();
    for d in &decisions {
        if let Decision::Admit { ticket, .. } = d {
            cache.complete_upload(ticket.clone(), true).unwrap();
        }
    }
    decisions
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0 >> 33
    }
}

fn run(history: &[Vec<ExpertKey>], capacity: u64) -> (Vec<Vec<Decision>>, Vec<ExpertKey>) {
    let mut c = cache(capacity, 16);
    let mut all = Vec::new();
    for step in history {
        c.observe(step).unwrap();
        all.push(settle(&mut c));
    }
    (all, c.resident_set())
}

#[test]
fn same_history_gives_same_decisions_and_resident_set() {
    let mut rng = Lcg(5);
    let history: Vec<Vec<ExpertKey>> = (0..200)
        .map(|_| (0..4).map(|_| key(0, (rng.next() % 16) as u32)).collect())
        .collect();
    let a = run(&history, 500);
    let b = run(&history, 500);
    assert_eq!(a.0, b.0);
    assert_eq!(a.1, b.1);
}

#[test]
fn a_miss_is_admitted_or_refused_never_served_from_pending() {
    let mut c = cache(100, 2);
    c.observe(&[key(0, 0)]).unwrap();
    let decisions = c.decide().unwrap();
    let ticket = match &decisions[..] {
        [Decision::Admit { key: k, ticket, .. }] if *k == key(0, 0) => ticket.clone(),
        other => panic!("{other:?}"),
    };
    // Pending: acquire is a miss, not a stale hit.
    assert!(c.acquire(key(0, 0)).unwrap().is_none());
    c.complete_upload(ticket.clone(), true).unwrap();
    assert!(c.acquire(key(0, 0)).unwrap().is_some());
    // The same ticket cannot complete twice.
    assert_eq!(
        c.complete_upload(ticket, true),
        Err(ResidencyError::InvalidTicket)
    );
}

#[test]
fn failed_upload_returns_bytes_and_is_counted() {
    let mut c = cache(100, 1);
    c.observe(&[key(0, 0)]).unwrap();
    let Decision::Admit { ticket, .. } = c.decide().unwrap().remove(0) else {
        panic!()
    };
    assert_eq!(c.used_bytes(), 100);
    assert_eq!(
        c.complete_upload(ticket, false),
        Err(ResidencyError::UploadFailed)
    );
    assert_eq!(c.used_bytes(), 0);
    assert_eq!(c.telemetry().failed_uploads, 1);
}

#[test]
fn leased_or_pinned_slots_are_never_evicted() {
    let mut c = cache(100, 3);
    c.observe(&[key(0, 0)]).unwrap();
    settle(&mut c);
    let lease = c.acquire(key(0, 0)).unwrap().unwrap();
    for _ in 0..40 {
        c.observe(&[key(0, 1), key(0, 1), key(0, 1)]).unwrap();
    }
    let d = c.decide().unwrap();
    assert!(d.contains(&Decision::Refuse {
        key: key(0, 1),
        reason: Refusal::NoVictim
    }));
    assert!(c.is_resident(key(0, 0)));
    c.release(lease).unwrap();
    c.pin(key(0, 0)).unwrap();
    assert!(c
        .decide()
        .unwrap()
        .iter()
        .all(|d| !matches!(d, Decision::Evict { .. })));
    c.unpin(key(0, 0)).unwrap();
    let d = settle(&mut c);
    assert_eq!(
        d[0],
        Decision::Evict {
            key: key(0, 0),
            bytes: 100
        }
    );
    assert!(c.is_resident(key(0, 1)));
}

#[test]
fn warm_reference_working_set_is_stable() {
    // Four hot experts fit exactly; a rare fifth expert must not churn them.
    let mut c = cache(400, 8);
    let hot = [key(0, 0), key(0, 1), key(0, 2), key(0, 3)];
    for _ in 0..20 {
        c.observe(&hot).unwrap();
        settle(&mut c);
    }
    let evictions_before = c.telemetry().evictions;
    for step in 0..500 {
        let mut keys = hot.to_vec();
        if step % 50 == 0 {
            keys.push(key(0, 5));
        }
        c.observe(&keys).unwrap();
        settle(&mut c);
    }
    assert_eq!(c.telemetry().evictions, evictions_before);
    assert_eq!(c.resident_set(), hot.to_vec());
}

#[test]
fn hysteresis_admits_a_clearly_hotter_expert() {
    let mut c = cache(100, 2);
    c.observe(&[key(0, 0)]).unwrap();
    settle(&mut c);
    for _ in 0..30 {
        c.observe(&[key(0, 1), key(0, 1)]).unwrap();
    }
    let d = settle(&mut c);
    assert_eq!(
        d[0],
        Decision::Evict {
            key: key(0, 0),
            bytes: 100
        }
    );
    assert!(matches!(d[1], Decision::Admit { key: k, .. } if k == key(0, 1)));
}

#[test]
fn refusals_are_typed() {
    let sizes = [(key(0, 0), 50), (key(0, 1), 500), (key(0, 2), 50)];
    let mut cfg = config(100);
    cfg.admit_min_score = 2 * HIT_SCORE;
    cfg.max_upload_bytes_per_step = 60;
    let mut c = DynamicCache::new(cfg, &sizes).unwrap();
    c.observe(&[
        key(0, 0),
        key(0, 0),
        key(0, 1),
        key(0, 1),
        key(0, 2),
        key(0, 2),
    ])
    .unwrap();
    let d = c.decide().unwrap();
    assert!(d.contains(&Decision::Refuse {
        key: key(0, 1),
        reason: Refusal::TooLarge
    }));
    assert!(d.contains(&Decision::Refuse {
        key: key(0, 2),
        reason: Refusal::UploadBudget
    }));
    c.observe(&[key(0, 2)]).unwrap();
    let mut cold = DynamicCache::new(cfg, &sizes).unwrap();
    cold.observe(&[key(0, 0)]).unwrap();
    assert_eq!(
        cold.decide().unwrap(),
        vec![Decision::Refuse {
            key: key(0, 0),
            reason: Refusal::Cold
        }]
    );
}

#[test]
fn shrink_returns_memory_but_not_below_the_leased_set() {
    let mut c = cache(300, 4);
    c.observe(&[key(0, 0), key(0, 1), key(0, 2)]).unwrap();
    settle(&mut c);
    let lease = c.acquire(key(0, 1)).unwrap().unwrap();
    c.pin(key(0, 2)).unwrap();
    assert_eq!(c.shrink(100), Err(ResidencyError::Busy));
    assert_eq!(c.used_bytes(), 300);
    let evicted = c.shrink(200).unwrap();
    assert_eq!(
        evicted,
        vec![Decision::Evict {
            key: key(0, 0),
            bytes: 100
        }]
    );
    assert_eq!(c.used_bytes(), 200);
    c.release(lease).unwrap();
    c.grow(300);
    assert_eq!(c.active_leases(), 0);
}

#[test]
fn telemetry_counts_hits_misses_and_unknown_measurement() {
    let mut c = cache(100, 2);
    c.observe(&[key(0, 0)]).unwrap();
    settle(&mut c);
    c.observe(&[key(0, 0), key(0, 1)]).unwrap();
    let t = c.telemetry();
    assert_eq!((t.steps, t.hits, t.misses), (2, 1, 2));
    assert_eq!(t.measured_bytes, None);
    c.report_measured_bytes(4096);
    assert_eq!(c.telemetry().measured_bytes, Some(4096));
}

#[test]
fn scores_decay_with_half_life_and_reject_unknown_keys() {
    let mut c = cache(100, 2);
    c.observe(&[key(0, 0)]).unwrap();
    for _ in 0..8 {
        c.observe(&[]).unwrap();
    }
    assert_eq!(c.score(key(0, 0)).unwrap(), HIT_SCORE / 2);
    assert_eq!(c.observe(&[key(1, 0)]), Err(ResidencyError::InvalidProfile));
    assert!(DynamicCache::new(config(1), &[(key(0, 0), 0)]).is_err());
    assert!(DynamicCache::new(config(1), &[(key(0, 0), 1), (key(0, 0), 1)]).is_err());
}

#[test]
fn foreign_lease_is_rejected() {
    let mut a = cache(100, 1);
    a.observe(&[key(0, 0)]).unwrap();
    settle(&mut a);
    let lease = a.acquire(key(0, 0)).unwrap().unwrap();
    a.release(lease.clone()).unwrap();
    assert_eq!(a.release(lease), Err(ResidencyError::InvalidTicket));
}
