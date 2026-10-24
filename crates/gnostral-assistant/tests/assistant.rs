//! Q-044/Q-045 gates: admitted audio never misses, barge-in cancels, speech starts at the first sentence.

use gnostral_assistant::chunk::Chunker;
use gnostral_assistant::share::{admit, simulate, Policy, ShareError, Stream};
use gnostral_assistant::turn::{Effect, Event, LatencyBudget, State, Turn, TurnError};

struct Lcg(u64);
impl Lcg {
    fn below(&mut self, n: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) % n
    }
}

/// 20 ms speech-to-text chunks and 40 ms text-to-speech chunks on one card.
fn voice() -> Vec<Stream> {
    vec![
        Stream {
            name: "stt",
            period_us: 20_000,
            cost_us: 3_000,
            phase_us: 0,
        },
        Stream {
            name: "tts",
            period_us: 40_000,
            cost_us: 8_000,
            phase_us: 5_000,
        },
    ]
}

#[test]
fn admission_bounds_the_language_model_step() {
    let a = admit(&voice(), 10_000).unwrap();
    assert!((a.utilization - 0.35).abs() < 1e-12);
    assert_eq!(a.max_step_us, 17_000);
    assert!(a.admitted);
    assert!(!admit(&voice(), 17_001).unwrap().admitted);
    assert_eq!(
        admit(
            &[Stream {
                name: "bad",
                period_us: 10,
                cost_us: 11,
                phase_us: 0
            }],
            1
        ),
        Err(ShareError::InvalidStream("bad"))
    );
    assert!(admit(&[], 1_000_000).unwrap().admitted);
    // Utilisation a hair under 1 with a 1 us stream: refused instead of an enormous check.
    let full = [
        Stream {
            name: "f",
            period_us: 1_000_000,
            cost_us: 999_999,
            phase_us: 0,
        },
        Stream {
            name: "g",
            period_us: 1,
            cost_us: 0,
            phase_us: 0,
        },
    ];
    assert!(!admit(&full, 1).unwrap().admitted);
}

#[test]
fn admitted_sets_never_miss_under_realtime_first() {
    let mut rng = Lcg(42);
    let mut fifo_missed_where_edf_did_not = false;
    let mut checked = 0;
    for _ in 0..1500 {
        let n = 1 + rng.below(3) as usize;
        let streams: Vec<Stream> = (0..n)
            .map(|i| {
                let period = (10 + rng.below(50)) * 1_000;
                Stream {
                    name: ["a", "b", "c"][i],
                    period_us: period,
                    cost_us: 500 + rng.below(period / 5),
                    phase_us: rng.below(period),
                }
            })
            .collect();
        let Ok(a) = admit(&streams, 1) else { continue };
        if !a.admitted || a.max_step_us == 0 {
            continue;
        }
        let step = 1 + rng.below(a.max_step_us);
        assert!(admit(&streams, step).unwrap().admitted);
        let edf = simulate(&streams, step, Policy::RealtimeFirst, 2_000_000);
        assert_eq!(edf.deadline_misses, 0, "{streams:?} step {step} {edf:?}");
        assert!(edf.lm_steps > 0);
        checked += 1;
        let fifo = simulate(&streams, step, Policy::Fifo, 2_000_000);
        fifo_missed_where_edf_did_not |= fifo.deadline_misses > 0;
    }
    assert!(checked > 500, "only {checked} admitted sets exercised");
    assert!(
        fifo_missed_where_edf_did_not,
        "deadline order matters somewhere"
    );
}

#[test]
fn an_oversized_step_makes_audio_late() {
    let streams = voice();
    let ok = simulate(&streams, 17_000, Policy::RealtimeFirst, 1_000_000);
    assert_eq!(ok.deadline_misses, 0);
    let late = simulate(&streams, 30_000, Policy::RealtimeFirst, 1_000_000);
    assert!(late.deadline_misses > 0 && late.worst_lateness_us > 0);
    assert!(ok.lm_share_per_mille > 500, "{ok:?}");
}

#[test]
fn voice_turn_happy_path_and_latency() {
    let mut t = Turn::new();
    assert_eq!(t.on(0, Event::SpeechStart).unwrap(), Effect::None);
    assert_eq!(
        t.on(1_800, Event::SpeechEnd).unwrap(),
        Effect::StartTranscription
    );
    assert_eq!(
        t.on(2_000, Event::Transcript("quelle heure est-il ?".into()))
            .unwrap(),
        Effect::StartGeneration
    );
    assert_eq!(t.on(2_450, Event::FirstPiece).unwrap(), Effect::StartSpeech);
    assert_eq!(t.on(6_000, Event::Done).unwrap(), Effect::None);
    assert_eq!(t.state(), State::Idle);
    assert_eq!(t.first_audio_latency_ms(), Some(650));
    assert_eq!(t.transcript(), Some("quelle heure est-il ?"));
    assert!(t.log().iter().all(|x| !x.barge_in));
}

#[test]
fn barge_in_cancels_output_from_any_busy_state() {
    for interrupt_in in [State::Transcribing, State::Thinking, State::Speaking] {
        let mut t = Turn::new();
        t.on(0, Event::SpeechStart).unwrap();
        t.on(10, Event::SpeechEnd).unwrap();
        if interrupt_in != State::Transcribing {
            t.on(20, Event::Transcript("x".into())).unwrap();
        }
        if interrupt_in == State::Speaking {
            t.on(30, Event::FirstPiece).unwrap();
        }
        assert_eq!(t.state(), interrupt_in);
        assert_eq!(t.on(40, Event::SpeechStart).unwrap(), Effect::CancelOutput);
        assert_eq!(t.state(), State::Listening);
        assert!(t.log().last().unwrap().barge_in);
    }
}

#[test]
fn out_of_order_events_are_typed_errors() {
    let mut t = Turn::new();
    assert!(matches!(
        t.on(0, Event::Done),
        Err(TurnError::Unexpected {
            state: State::Idle,
            ..
        })
    ));
    t.on(5, Event::SpeechStart).unwrap();
    assert_eq!(
        t.on(4, Event::SpeechEnd),
        Err(TurnError::Clock {
            last_ms: 5,
            at_ms: 4
        })
    );
    assert!(matches!(
        t.on(6, Event::FirstPiece),
        Err(TurnError::Unexpected { .. })
    ));
    assert_eq!(t.on(7, Event::Cancel).unwrap(), Effect::CancelOutput);
    assert_eq!(t.state(), State::Idle);
}

#[test]
fn first_audio_budget_adds_up() {
    let b = LatencyBudget {
        stt_final_ms: 120.0,
        prefill_ms: 180.0,
        tpot_ms: 28.0,
        tokens_to_first_piece: 9.0,
        tts_first_ms: 90.0,
    };
    assert!((b.first_audio_ms() - 642.0).abs() < 1e-9);
}

#[test]
fn chunker_speaks_sentence_by_sentence() {
    let mut c = Chunker::new(80);
    let mut pieces = Vec::new();
    for token in [
        "Il est ",
        "3.14 h",
        " à Montréal.",
        " Veux-tu",
        " autre chose ?",
        " Oui",
    ] {
        pieces.extend(c.push(token));
    }
    assert_eq!(
        pieces,
        vec!["Il est 3.14 h à Montréal. ", "Veux-tu autre chose ? "]
    );
    assert_eq!(c.finish().as_deref(), Some("Oui"));
    assert_eq!(c.finish(), None);
}

#[test]
fn chunker_cuts_long_runs_and_never_loses_text() {
    let mut rng = Lcg(7);
    let alphabet: Vec<char> = "abcdé fgh. ij! k?\nlm;nop: 12.5 ".chars().collect();
    for _ in 0..300 {
        let max = 5 + rng.below(30) as usize;
        let mut c = Chunker::new(max);
        let text: String = (0..rng.below(300))
            .map(|_| alphabet[rng.below(alphabet.len() as u64) as usize])
            .collect();
        let mut out = Vec::new();
        let mut i = 0;
        let chars: Vec<char> = text.chars().collect();
        while i < chars.len() {
            let n = 1 + rng.below(6) as usize;
            let piece: String = chars[i..(i + n).min(chars.len())].iter().collect();
            out.extend(c.push(&piece));
            i += n;
        }
        out.extend(c.finish());
        assert_eq!(out.concat(), text);
        for p in &out[..out.len().saturating_sub(1)] {
            let content = p.trim_end().chars().count();
            assert!(content <= max, "{p:?} > {max}");
            assert!(!p.is_empty());
        }
    }
}
