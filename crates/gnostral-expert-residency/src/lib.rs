//! Native expert-residency bookkeeping. No inference or physical-memory claims.
// Mechanisms informed by Niko1221/Strata, MIT, copyright 2026 contributors.
pub mod dynamic;
mod plan;
pub use plan::{CachePlan, ExpertKey, ExpertSize, SlotSpec};
mod routes;
mod table;
pub use routes::{partition_routes, Route, RoutePartition};
pub use table::{Lease, ResidencyTable, SlotId, UploadTicket};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidencyError {
    InvalidGeometry,
    InvalidProfile,
    Overflow,
    InvalidTicket,
    Busy,
    UploadFailed,
    Closed,
}
impl std::fmt::Display for ResidencyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "expert residency: {self:?}")
    }
}
impl std::error::Error for ResidencyError {}
