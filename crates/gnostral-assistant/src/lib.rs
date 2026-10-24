//! Policy for running an assistant (ears, voice, brain) on one desktop card.
//!
//! - [`share`]: one GPU, several models. Audio work has deadlines, a language
//!   model step cannot be interrupted once launched. An admission test bounds
//!   the longest non-audio step so audio never misses, and a simulator checks it
//!   (Q-045).
//! - [`turn`]: the voice turn as an explicit state machine, with barge-in (the
//!   person talks over the assistant) cancelling generation and speech (Q-044).
//! - [`chunk`]: cutting streamed text into speakable pieces so speech starts
//!   at the first sentence, not at the end of the answer (Q-044).
//!
//! Everything here is policy and simulation: no device call, no model, no
//! measurement. Costs are inputs that Q-042 timers will supply.

pub mod chunk;
pub mod share;
pub mod turn;
