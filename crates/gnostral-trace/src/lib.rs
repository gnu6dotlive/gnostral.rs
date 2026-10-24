//! Per-phase request timeline (Q-042).
//!
//! - **Off by default.** [`Recorder::from_env`] is enabled only when
//!   `GNOSTRAL_TRACE=1`; a disabled recorder never reads the clock.
//! - **Allocation-free hot path.** Spans go into a buffer preallocated at
//!   construction; when it is full, spans are counted as dropped, never
//!   reallocated.
//! - **Monotonic clock.** [`MonotonicClock`] wraps `std::time::Instant`; its
//!   name is recorded in every receipt. Tests inject [`ManualClock`].
//! - **Reconciliation.** [`Recorder::timeline`] compares the sum of recorded
//!   phases with the measured wall time; overlaps are reported and the gap is an
//!   explicit `UNKNOWN` phase, never zero by assumption.
//! - **Overhead honesty.** [`measure_overhead`] times empty spans on the real
//!   clock so the receipt carries the measured instrumentation cost.

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Environment flag that turns tracing on.
pub const TRACE_ENV: &str = "GNOSTRAL_TRACE";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Phase {
    Schedule,
    Forward,
    CpuExpert,
    Upload,
    Sample,
    PostProcess,
    /// Waiting for queued device work to finish before a host read (not a transfer).
    GpuWait,
    /// Device-to-host and host-to-device copies in the MoE path, after any wait.
    HostTransfer,
    /// Routing plan, gather and ordered scatter on the host.
    MoeBookkeeping,
}

impl Phase {
    pub const ALL: [Phase; 9] = [
        Phase::Schedule,
        Phase::Forward,
        Phase::CpuExpert,
        Phase::Upload,
        Phase::Sample,
        Phase::PostProcess,
        Phase::GpuWait,
        Phase::HostTransfer,
        Phase::MoeBookkeeping,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Phase::Schedule => "schedule",
            Phase::Forward => "forward",
            Phase::CpuExpert => "cpu_expert",
            Phase::Upload => "upload",
            Phase::Sample => "sample",
            Phase::PostProcess => "post_process",
            Phase::GpuWait => "gpu_wait",
            Phase::HostTransfer => "host_transfer",
            Phase::MoeBookkeeping => "moe_bookkeeping",
        }
    }
}

pub trait Clock: Sync {
    fn now_ns(&self) -> u64;
    fn source(&self) -> &'static str;
}

/// `std::time::Instant` relative to construction (CLOCK_MONOTONIC on Linux).
pub struct MonotonicClock {
    base: Instant,
}

impl Default for MonotonicClock {
    fn default() -> Self {
        Self {
            base: Instant::now(),
        }
    }
}

impl Clock for MonotonicClock {
    fn now_ns(&self) -> u64 {
        self.base.elapsed().as_nanos() as u64
    }
    fn source(&self) -> &'static str {
        "std::time::Instant (monotonic)"
    }
}

/// Test clock advanced by hand.
#[derive(Default)]
pub struct ManualClock(AtomicU64);

impl ManualClock {
    pub fn advance(&self, ns: u64) {
        self.0.fetch_add(ns, Ordering::SeqCst);
    }
}

impl Clock for ManualClock {
    fn now_ns(&self) -> u64 {
        self.0.load(Ordering::SeqCst)
    }
    fn source(&self) -> &'static str {
        "manual (test)"
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub phase: Phase,
    pub start_ns: u64,
    pub end_ns: u64,
}

pub struct Recorder<'c> {
    clock: &'c dyn Clock,
    enabled: bool,
    spans: Vec<Span>,
    dropped: u64,
    start_ns: Option<u64>,
    end_ns: Option<u64>,
}

impl<'c> Recorder<'c> {
    pub fn new(clock: &'c dyn Clock, enabled: bool, capacity: usize) -> Self {
        Self {
            clock,
            enabled,
            spans: Vec::with_capacity(if enabled { capacity } else { 0 }),
            dropped: 0,
            start_ns: None,
            end_ns: None,
        }
    }

    /// Enabled only when `GNOSTRAL_TRACE=1`.
    pub fn from_env(clock: &'c dyn Clock, capacity: usize) -> Self {
        let enabled = std::env::var(TRACE_ENV).is_ok_and(|v| v == "1");
        Self::new(clock, enabled, capacity)
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Mark the request's wall-clock start.
    pub fn begin_request(&mut self) {
        if self.enabled {
            self.start_ns = Some(self.clock.now_ns());
        }
    }

    pub fn end_request(&mut self) {
        if self.enabled {
            self.end_ns = Some(self.clock.now_ns());
        }
    }

    /// Start timing a phase; record it with [`Recorder::stop`]. Returns 0 when disabled.
    #[inline]
    pub fn start(&self) -> u64 {
        if self.enabled {
            self.clock.now_ns()
        } else {
            0
        }
    }

    /// Record `phase` from `start_ns` to now. Never allocates.
    #[inline]
    pub fn stop(&mut self, phase: Phase, start_ns: u64) {
        if !self.enabled {
            return;
        }
        let end_ns = self.clock.now_ns();
        if self.spans.len() == self.spans.capacity() {
            self.dropped += 1;
        } else {
            self.spans.push(Span {
                phase,
                start_ns,
                end_ns,
            });
        }
    }

    pub fn spans(&self) -> &[Span] {
        &self.spans
    }

    pub fn timeline(&self, identity: Identity) -> Timeline {
        let wall_ns = match (self.start_ns, self.end_ns) {
            (Some(s), Some(e)) if e >= s => Some(e - s),
            _ => None,
        };
        let mut phases: Vec<PhaseTotal> = Phase::ALL
            .iter()
            .map(|&p| {
                let mine: Vec<&Span> = self.spans.iter().filter(|s| s.phase == p).collect();
                PhaseTotal {
                    phase: p.name(),
                    count: mine.len() as u64,
                    total_ns: (!mine.is_empty())
                        .then(|| mine.iter().map(|s| s.end_ns - s.start_ns).sum()),
                }
            })
            .collect();
        phases.sort_by_key(|p| p.phase);
        let attributed: u64 = phases.iter().filter_map(|p| p.total_ns).sum();
        let mut sorted = self.spans.clone();
        sorted.sort_by_key(|s| (s.start_ns, s.end_ns));
        // Covered time is the union of all spans; nested or overlapping spans are counted once.
        let mut covered_ns = 0u64;
        let mut cursor = 0u64;
        for s in &sorted {
            let from = s.start_ns.max(cursor);
            if s.end_ns > from {
                covered_ns += s.end_ns - from;
            }
            cursor = cursor.max(s.end_ns);
        }
        let overlap_ns = attributed - covered_ns;
        Timeline {
            identity,
            clock: self.clock.source(),
            enabled: self.enabled,
            wall_ns,
            phases,
            attributed_ns: attributed,
            unknown_ns: wall_ns.map(|w| w.saturating_sub(covered_ns)),
            overlap_ns,
            dropped_spans: self.dropped,
            spans: self.spans.clone(),
            overhead_ns_per_span: None,
        }
    }
}

/// Binary and model identity carried by every exported timeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Identity {
    pub binary_sha256: Option<String>,
    pub model_sha256: Option<String>,
    pub command: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhaseTotal {
    pub phase: &'static str,
    pub count: u64,
    /// `None`: the phase was never recorded (reported as `UNKNOWN`, not zero).
    pub total_ns: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Timeline {
    pub identity: Identity,
    pub clock: &'static str,
    pub enabled: bool,
    pub wall_ns: Option<u64>,
    pub phases: Vec<PhaseTotal>,
    pub attributed_ns: u64,
    /// Wall time not covered by any span; `None` without a wall measurement.
    pub unknown_ns: Option<u64>,
    pub overlap_ns: u64,
    pub dropped_spans: u64,
    pub spans: Vec<Span>,
    pub overhead_ns_per_span: Option<u64>,
}

fn json_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn json_opt_u64(out: &mut String, v: Option<u64>) {
    match v {
        Some(v) => {
            let _ = write!(out, "{v}");
        }
        None => out.push_str("\"UNKNOWN\""),
    }
}

fn json_opt_str(out: &mut String, v: &Option<String>) {
    match v {
        Some(v) => json_str(out, v),
        None => out.push_str("\"UNKNOWN\""),
    }
}

impl Timeline {
    /// Deterministic JSON receipt (keys in fixed order, `UNKNOWN` strings for missing values).
    pub fn to_json(&self) -> String {
        let mut o = String::new();
        o.push_str("{\"schema\":\"gnostral-trace/1\",\"clock\":");
        json_str(&mut o, self.clock);
        let _ = write!(
            o,
            ",\"enabled\":{},\"identity\":{{\"binary_sha256\":",
            self.enabled
        );
        json_opt_str(&mut o, &self.identity.binary_sha256);
        o.push_str(",\"model_sha256\":");
        json_opt_str(&mut o, &self.identity.model_sha256);
        o.push_str(",\"command\":");
        json_str(&mut o, &self.identity.command);
        o.push_str("},\"wall_ns\":");
        json_opt_u64(&mut o, self.wall_ns);
        o.push_str(",\"phases\":[");
        for (i, p) in self.phases.iter().enumerate() {
            if i > 0 {
                o.push(',');
            }
            o.push_str("{\"phase\":");
            json_str(&mut o, p.phase);
            let _ = write!(o, ",\"count\":{},\"total_ns\":", p.count);
            json_opt_u64(&mut o, p.total_ns);
            o.push('}');
        }
        let _ = write!(
            o,
            "],\"attributed_ns\":{},\"unknown_ns\":",
            self.attributed_ns
        );
        json_opt_u64(&mut o, self.unknown_ns);
        let _ = write!(
            o,
            ",\"overlap_ns\":{},\"dropped_spans\":{},\"overhead_ns_per_span\":",
            self.overlap_ns, self.dropped_spans
        );
        json_opt_u64(&mut o, self.overhead_ns_per_span);
        o.push_str(",\"spans\":[");
        for (i, s) in self.spans.iter().enumerate() {
            if i > 0 {
                o.push(',');
            }
            let _ = write!(o, "[\"{}\",{},{}]", s.phase.name(), s.start_ns, s.end_ns);
        }
        o.push_str("]}");
        o
    }

    /// Small fixed-width text summary.
    pub fn summary(&self) -> String {
        let mut o = String::new();
        let ms = |ns: u64| ns as f64 / 1e6;
        let _ = writeln!(o, "{:<14} {:>6} {:>12}", "phase", "count", "total_ms");
        for p in &self.phases {
            match p.total_ns {
                Some(ns) => {
                    let _ = writeln!(o, "{:<14} {:>6} {:>12.3}", p.phase, p.count, ms(ns));
                }
                None => {
                    let _ = writeln!(o, "{:<14} {:>6} {:>12}", p.phase, p.count, "UNKNOWN");
                }
            }
        }
        match (self.wall_ns, self.unknown_ns) {
            (Some(w), Some(u)) => {
                let _ = writeln!(o, "{:<14} {:>6} {:>12.3}", "unattributed", "", ms(u));
                let _ = writeln!(o, "{:<14} {:>6} {:>12.3}", "wall", "", ms(w));
            }
            _ => {
                let _ = writeln!(o, "{:<14} {:>6} {:>12}", "wall", "", "UNKNOWN");
            }
        }
        o
    }
}

/// Mean cost of one start/stop pair on `clock`, in nanoseconds, over `n` spans.
pub fn measure_overhead(clock: &dyn Clock, n: usize) -> u64 {
    let mut r = Recorder::new(clock, true, n);
    let t0 = clock.now_ns();
    for _ in 0..n {
        let s = r.start();
        r.stop(Phase::Schedule, s);
    }
    let t1 = clock.now_ns();
    (t1 - t0) / n.max(1) as u64
}
