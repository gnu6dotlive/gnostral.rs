//! Q-043 broker gates: no overbooking at any plan step, class rules, typed refusals, receipts.

use gnostral_cohost::{
    replay, Broker, Class, CohostError, Elastic, Kind, Residency, Step, TenantId, TenantSpec,
};
use gnostral_memory_plan::envelope::Envelope;
use gnostral_memory_plan::{GIB, MIB};

fn spec(
    name: &str,
    kind: Kind,
    class: Class,
    core: u64,
    elastic: Option<(u64, u64)>,
    parkable: bool,
) -> TenantSpec {
    TenantSpec {
        name: name.into(),
        kind,
        class,
        core_bytes: core,
        elastic: elastic.map(|(min, max)| Elastic { min, max }),
        parkable,
    }
}

/// RTX 3070 desktop: 8 GiB card, 1 GiB desktop, 512 MiB reserve, 512 MiB free floor.
fn desktop() -> Envelope {
    Envelope {
        device_bytes: 8 * GIB,
        desktop_ambient: Some(GIB),
        desktop_reserve: 512 * MIB,
        min_free: 512 * MIB,
    }
}

fn llm() -> TenantSpec {
    spec(
        "qwen3-30b-a3b",
        Kind::Language,
        Class::Interactive,
        2560 * MIB,
        Some((512 * MIB, 4 * GIB)),
        false,
    )
}
fn stt() -> TenantSpec {
    spec(
        "stt",
        Kind::SpeechToText,
        Class::Realtime,
        800 * MIB,
        None,
        true,
    )
}
fn tts() -> TenantSpec {
    spec(
        "tts",
        Kind::TextToSpeech,
        Class::Realtime,
        400 * MIB,
        None,
        true,
    )
}
fn draft() -> TenantSpec {
    spec(
        "draft",
        Kind::Draft,
        Class::Background,
        600 * MIB,
        None,
        true,
    )
}

fn assert_safe(
    b: &mut Broker,
    f: impl FnOnce(&mut Broker) -> Result<Vec<Step>, CohostError>,
) -> Result<Vec<Step>, CohostError> {
    let (used, cap) = (b.used(), b.capacity());
    let before = b.views();
    let r = f(b);
    match &r {
        Ok(plan) => {
            let trace = replay(used, cap, plan).expect("plan overbooks the card");
            assert_eq!(trace.last().copied().unwrap_or(used), b.used());
            assert!(b.used() <= b.capacity());
        }
        Err(_) => assert_eq!(before, b.views(), "a refusal must change nothing"),
    }
    r
}

fn admit_safe(b: &mut Broker, s: TenantSpec) -> TenantId {
    let mut id = None;
    assert_safe(b, |b| {
        b.admit(s).map(|(i, plan)| {
            id = Some(i);
            plan
        })
    })
    .unwrap();
    id.unwrap()
}

#[test]
fn unknown_desktop_ambient_is_refused() {
    let mut e = desktop();
    e.desktop_ambient = None;
    assert!(matches!(Broker::new(&e), Err(CohostError::Memory(_))));
}

#[test]
fn assistant_on_one_desktop_card() {
    let mut b = Broker::new(&desktop()).unwrap();
    assert_eq!(b.capacity(), 6 * GIB);

    let (l, plan) = b.admit(llm()).unwrap();
    // Alone, the language model takes core + as much expert cache as fits.
    assert_eq!(
        plan,
        vec![Step::Load {
            id: l,
            core: 2560 * MIB,
            elastic: 6 * GIB - 2560 * MIB
        }]
    );

    let s = admit_safe(&mut b, stt());
    let t = admit_safe(&mut b, tts());
    assert_eq!(b.used(), 6 * GIB);
    assert_eq!(
        b.view(l).unwrap().bytes,
        6 * GIB - 1200 * MIB,
        "speech came out of the expert cache"
    );

    // Background draft may not squeeze the interactive model's cache.
    let refused = assert_safe(&mut b, |b| b.admit(draft()).map(|x| x.1));
    assert!(
        matches!(refused, Err(CohostError::Refused { needed, reclaimable: 0 }) if needed == 600 * MIB)
    );

    // Speech leaves: its bytes go back to the expert cache.
    assert_safe(&mut b, |b| b.release(s)).unwrap();
    assert_safe(&mut b, |b| b.release(t)).unwrap();
    assert_eq!(
        b.view(l).unwrap().residency,
        Residency::Loaded {
            elastic: 6 * GIB - 2560 * MIB
        }
    );
    assert_eq!(b.receipts().len(), 6);
}

#[test]
fn realtime_parks_idle_background_but_not_busy() {
    let mut b = Broker::with_capacity(3 * GIB);
    let (_l, _) = b
        .admit(spec(
            "lm",
            Kind::Language,
            Class::Interactive,
            2 * GIB,
            None,
            false,
        ))
        .unwrap();
    let (d, _) = b
        .admit(spec(
            "draft",
            Kind::Draft,
            Class::Background,
            900 * MIB,
            None,
            true,
        ))
        .unwrap();
    b.set_busy(d, true).unwrap();
    let r = assert_safe(&mut b, |b| b.admit(stt()).map(|x| x.1));
    assert!(
        matches!(r, Err(CohostError::Refused { .. })),
        "busy tenants are never parked"
    );
    b.set_busy(d, false).unwrap();
    let plan = assert_safe(&mut b, |b| b.admit(stt()).map(|x| x.1)).unwrap();
    assert!(matches!(plan[0], Step::Park { id, bytes } if id == d && bytes == 900 * MIB));
    // Waking the draft needs room it cannot take from a higher class.
    let woke = assert_safe(&mut b, |b| b.wake(d));
    assert!(matches!(woke, Err(CohostError::Refused { .. })));
    assert!(matches!(
        b.wake(TenantId(0)),
        Err(CohostError::WrongState(_))
    ));
}

#[test]
fn equal_class_shares_elastic_but_never_parks() {
    let mut b = Broker::with_capacity(4 * GIB);
    let (a, _) = b
        .admit(spec(
            "a",
            Kind::Language,
            Class::Interactive,
            GIB,
            Some((256 * MIB, 3 * GIB)),
            true,
        ))
        .unwrap();
    let (_c, plan) = b
        .admit(spec(
            "c",
            Kind::Embedding,
            Class::Interactive,
            2 * GIB,
            None,
            false,
        ))
        .unwrap();
    assert!(matches!(plan[0], Step::Shrink { id, .. } if id == a));
    assert_eq!(
        b.view(a).unwrap().residency,
        Residency::Loaded { elastic: GIB }
    );
    let r = b.admit(spec("e", Kind::Other, Class::Interactive, GIB, None, false));
    assert!(
        matches!(r, Err(CohostError::Refused { needed, reclaimable }) if needed == GIB && reclaimable == 768 * MIB)
    );
}

#[test]
fn desktop_pressure_shrinks_lowest_class_first_or_refuses() {
    let mut b = Broker::with_capacity(6 * GIB);
    let (l, _) = b.admit(llm()).unwrap();
    let (s, _) = b.admit(stt()).unwrap();
    let plan = b.set_capacity(5 * GIB).unwrap();
    assert!(matches!(plan[..], [Step::Shrink { id, .. }] if id == l));
    assert!(b.used() <= 5 * GIB);
    // Below every core and minimum: refused, nothing moves (speech is busy).
    b.set_busy(s, true).unwrap();
    let before = b.views();
    assert!(matches!(
        b.set_capacity(3 * GIB),
        Err(CohostError::Refused { .. })
    ));
    assert_eq!(before, b.views());
    // Pressure gone: the cache grows back.
    let plan = b.set_capacity(6 * GIB).unwrap();
    assert!(matches!(plan[..], [Step::Grow { id, .. }] if id == l));
    assert_eq!(b.used(), 6 * GIB);
}

#[test]
fn invalid_specs_and_unknown_ids_are_typed() {
    let mut b = Broker::with_capacity(GIB);
    assert_eq!(
        b.admit(spec("z", Kind::Other, Class::Background, 0, None, false))
            .unwrap_err(),
        CohostError::InvalidSpec
    );
    assert_eq!(
        b.admit(spec(
            "x",
            Kind::Other,
            Class::Background,
            1,
            Some((5, 4)),
            false
        ))
        .unwrap_err(),
        CohostError::InvalidSpec
    );
    assert_eq!(
        b.release(TenantId(9)).unwrap_err(),
        CohostError::NotFound(TenantId(9))
    );
    assert_eq!(b.receipts().len(), 3);
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

#[test]
fn random_sequences_never_overbook_and_respect_classes() {
    for seed in 0..200 {
        let mut rng = Lcg(seed);
        let mut b = Broker::with_capacity((2 + rng.below(6)) * GIB);
        for _ in 0..60 {
            let ids: Vec<TenantId> = b.views().iter().map(|v| v.id).collect();
            let pick = |rng: &mut Lcg| {
                ids.get(rng.below(ids.len().max(1) as u64) as usize)
                    .copied()
            };
            let before = b.views();
            match rng.below(6) {
                0 | 1 => {
                    let class = [Class::Realtime, Class::Interactive, Class::Background]
                        [rng.below(3) as usize];
                    let core = (1 + rng.below(16)) * 128 * MIB;
                    let elastic = (rng.below(2) == 0).then(|| {
                        let min = rng.below(4) * 128 * MIB;
                        (min, min + rng.below(16) * 128 * MIB)
                    });
                    let s = spec("t", Kind::Other, class, core, elastic, rng.below(2) == 0);
                    let r = assert_safe(&mut b, |b| b.admit(s).map(|x| x.1));
                    if let Ok(plan) = r {
                        for step in &plan {
                            let victim = match step {
                                Step::Shrink { id, .. } => Some((id, false)),
                                Step::Park { id, .. } => Some((id, true)),
                                _ => None,
                            };
                            if let Some((id, parked)) = victim {
                                let v = before.iter().find(|v| v.id == *id).unwrap();
                                if parked {
                                    assert!(v.class > class && !v.busy);
                                } else {
                                    assert!(v.class >= class);
                                }
                            }
                        }
                    }
                }
                2 => {
                    if let Some(id) = pick(&mut rng) {
                        assert_safe(&mut b, |b| b.release(id)).unwrap();
                    }
                }
                3 => {
                    if let Some(id) = pick(&mut rng) {
                        let _ = assert_safe(&mut b, |b| b.wake(id));
                    }
                }
                4 => {
                    if let Some(id) = pick(&mut rng) {
                        b.set_busy(id, rng.below(2) == 0).unwrap();
                    }
                }
                _ => {
                    let cap = (1 + rng.below(8)) * GIB;
                    let before = b.views();
                    match b.set_capacity(cap) {
                        Ok(_) => assert!(b.used() <= b.capacity()),
                        Err(_) => assert_eq!(before, b.views()),
                    }
                }
            }
            for v in b.views() {
                if let Residency::Loaded { .. } = v.residency {
                    assert!(v.bytes > 0);
                }
            }
            assert!(b.used() <= b.capacity());
        }
    }
}
