//! Memory budget arithmetic for the gnostral engine program.
//!
//! - [`envelope`]: one device envelope shared by workspace, KV and expert cache,
//!   with the desktop reserve and a free floor taken off the top (Q-027).
//! - [`segments`]: segmented allocator with elastic return and a fragmentation
//!   metric (Q-027).
//! - [`kv`]: paged KV block layout, block pool and per-sequence block tables with
//!   copy-on-write forks (Q-028).
//! - [`stream`]: long-context KV tiering and per-step streaming plan, with
//!   ticketed block moves (Q-038).
//! - [`reserve`]: desktop-reserve state machine with receipted transitions (Q-041).
//!
//! Everything here is bookkeeping: no allocation, no device call, no
//! measurement. Unknown inputs stay `None` (`UNKNOWN`) and are refused where a
//! decision would depend on them.

pub mod envelope;
pub mod kv;
pub mod reserve;
pub mod segments;
pub mod stream;

/// A rejected memory operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryError {
    /// An input that a decision depends on is `UNKNOWN`.
    Unknown { what: &'static str },
    /// Reserve and free floor leave nothing (or less than zero) for the engine.
    EnvelopeTooSmall { device: u64, held_back: u64 },
    /// The components' minima do not fit.
    MinimaExceedEnvelope { needed: u64, available: u64 },
    /// A request is malformed (duplicate component, desired below minimum, zero size).
    InvalidRequest,
    /// The allocation is larger than one segment.
    TooLarge { bytes: u64, segment: u64 },
    /// No segment can hold the allocation and no new segment may be opened.
    OutOfSegments,
    /// Not enough free KV blocks.
    NoBlocks { needed: usize, free: usize },
    /// Unknown sequence, block, handle or ticket.
    NotFound,
    /// The object is in use (leased, referenced or in flight).
    Busy,
    /// A size computation overflowed.
    Overflow,
}

impl std::fmt::Display for MemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "memory plan: {self:?}")
    }
}

impl std::error::Error for MemoryError {}

/// GiB in bytes.
pub const GIB: u64 = 1 << 30;
/// MiB in bytes.
pub const MIB: u64 = 1 << 20;
