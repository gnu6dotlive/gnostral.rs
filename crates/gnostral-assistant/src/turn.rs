//! One voice turn as an explicit state machine.
//!
//! `Idle -> Listening -> Transcribing -> Thinking -> Speaking -> Idle`, where
//! `Thinking` is the language model before its first speakable piece and
//! `Speaking` overlaps generation with playback. The person may start talking
//! again at any time after `Listening` (barge-in): generation and speech are
//! cancelled at once and the machine goes back to `Listening`. Every transition
//! is recorded with its time, and an event that does not fit the state is a
//! typed error, not a silent fix.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    Idle,
    Listening,
    Transcribing,
    Thinking,
    Speaking,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    /// Voice activity started.
    SpeechStart,
    /// Voice activity ended; the audio goes to speech-to-text.
    SpeechEnd,
    /// Final transcript; the language model starts.
    Transcript(String),
    /// The first speakable piece is ready; text-to-speech starts.
    FirstPiece,
    /// Generation finished and the last audio played.
    Done,
    /// The person or the app cancelled the turn.
    Cancel,
}

/// What the engine must do on a transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effect {
    None,
    StartTranscription,
    StartGeneration,
    StartSpeech,
    /// Stop generation and speech now; drop queued audio.
    CancelOutput,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transition {
    pub at_ms: u64,
    pub from: State,
    pub to: State,
    pub effect: Effect,
    pub barge_in: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TurnError {
    Unexpected {
        state: State,
        event: Event,
    },
    /// Time went backwards.
    Clock {
        last_ms: u64,
        at_ms: u64,
    },
}

#[derive(Debug, Clone)]
pub struct Turn {
    state: State,
    last_ms: u64,
    log: Vec<Transition>,
    transcript: Option<String>,
}

impl Default for Turn {
    fn default() -> Self {
        Self::new()
    }
}

impl Turn {
    pub fn new() -> Self {
        Self {
            state: State::Idle,
            last_ms: 0,
            log: Vec::new(),
            transcript: None,
        }
    }

    pub fn state(&self) -> State {
        self.state
    }

    pub fn log(&self) -> &[Transition] {
        &self.log
    }

    pub fn transcript(&self) -> Option<&str> {
        self.transcript.as_deref()
    }

    pub fn on(&mut self, at_ms: u64, event: Event) -> Result<Effect, TurnError> {
        if at_ms < self.last_ms {
            return Err(TurnError::Clock {
                last_ms: self.last_ms,
                at_ms,
            });
        }
        use State::*;
        let (to, effect, barge_in) = match (self.state, &event) {
            (Idle, Event::SpeechStart) => (Listening, Effect::None, false),
            (Listening, Event::SpeechEnd) => (Transcribing, Effect::StartTranscription, false),
            (Transcribing, Event::Transcript(text)) => {
                self.transcript = Some(text.clone());
                (Thinking, Effect::StartGeneration, false)
            }
            (Thinking, Event::FirstPiece) => (Speaking, Effect::StartSpeech, false),
            (Speaking, Event::Done) => (Idle, Effect::None, false),
            (Transcribing, Event::SpeechStart) => (Listening, Effect::CancelOutput, true),
            (Thinking | Speaking, Event::SpeechStart) => (Listening, Effect::CancelOutput, true),
            (Listening | Transcribing | Thinking | Speaking, Event::Cancel) => {
                (Idle, Effect::CancelOutput, false)
            }
            (state, _) => return Err(TurnError::Unexpected { state, event }),
        };
        self.log.push(Transition {
            at_ms,
            from: self.state,
            to,
            effect,
            barge_in,
        });
        self.state = to;
        self.last_ms = at_ms;
        Ok(effect)
    }

    /// Milliseconds from the end of the person's speech to the first audio, for the last turn.
    pub fn first_audio_latency_ms(&self) -> Option<u64> {
        let end = self
            .log
            .iter()
            .rev()
            .find(|t| t.to == State::Transcribing)?;
        let speak = self
            .log
            .iter()
            .find(|t| t.to == State::Speaking && t.at_ms >= end.at_ms)?;
        Some(speak.at_ms - end.at_ms)
    }
}

/// Inputs of the first-audio budget, all measured by Q-042 timers in the real engine.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatencyBudget {
    /// Speech-to-text time after the end of speech.
    pub stt_final_ms: f64,
    /// Language model prefill of the transcript and history.
    pub prefill_ms: f64,
    /// Time per generated token.
    pub tpot_ms: f64,
    /// Tokens until the first speakable piece.
    pub tokens_to_first_piece: f64,
    /// Text-to-speech time for the first piece.
    pub tts_first_ms: f64,
}

impl LatencyBudget {
    pub fn first_audio_ms(&self) -> f64 {
        self.stt_final_ms
            + self.prefill_ms
            + self.tokens_to_first_piece * self.tpot_ms
            + self.tts_first_ms
    }
}
