//! Dependency-free receipt validation and comparison for Q-031 `ENGINE-PERF-01`.
//!
//! This crate reads machine-readable benchmark receipts, checks that every
//! identity pin and every frozen protocol field is present and internally
//! consistent, recomputes the derived statistics from the individual samples,
//! and renders a Pareto table. The [`run`] module drives a frozen-protocol plan
//! (identity pins, warmup, interleaved repetitions, every run kept). Nothing
//! here invents a number: anything a lane does not report is either rejected
//! or carried as an explicit `UNKNOWN`.

pub mod compare;
pub mod json;
pub mod receipt;
pub mod run;
pub mod sha256;
pub mod stats;

pub use compare::compare;
pub use receipt::{decode, DriverStatus, ExitStatus, LaneIdentity, Parity, Receipt, Sample};
pub use stats::{statistics, Stats};
