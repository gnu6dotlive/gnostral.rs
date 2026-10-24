//! Speculative decoding with a small draft model (Q-046).
//!
//! The vendored engine already carries speculative machinery (MTP, DFlash,
//! draft proposers and verifiers). This crate does not replace it; it supplies
//! what the program needs around it:
//!
//! - [`reference`]: a plain greedy speculation loop over abstract models. Its
//!   output is token-identical to plain greedy decoding for **any** draft, which
//!   makes it the oracle the engine's speculative path is gated against (Q-030).
//! - [`policy`]: acceptance tracking and a pure cost model choosing the draft
//!   length `k`, including `k = 0` (speculation off) when the draft does not pay
//!   for itself, which is also when the co-hosting broker (Q-043) may park it.

pub mod policy;
pub mod reference;
