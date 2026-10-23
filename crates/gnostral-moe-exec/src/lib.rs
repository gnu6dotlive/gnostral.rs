//! Deterministic route partitioning and grouped MoE dispatch planning.
//!
//! This crate is the testable, allocation-owning policy half of the engine's MoE
//! executor. It does not allocate GPU memory, does not depend on Candle/CUDA and
//! never claims a speedup: it turns a flat list of routed `(token, expert, weight)`
//! assignments into a deterministic, conserved set of groups, one group per
//! `(placement, expert)`, so an engine can issue one kernel per group instead of
//! one kernel per route.
//!
//! Design rules:
//! - **Deterministic.** The plan depends only on the routes and the placements,
//!   never on hash-map iteration or arrival timing. Same input, same plan.
//! - **Conserved.** Every input route index appears in exactly one group, in one
//!   position. `validate_conservation` proves it for a plan.
//! - **Placement is explicit.** `GpuResident`, `CpuHost` and `Upload` routes never
//!   share a group; the caller decides placement before planning, the engine owns
//!   materialization.
//! - **Bounded batches.** A group can be split at `max_group_routes` so a caller
//!   can cap kernel batch size without reordering routes.
//!
//! The reference executor at the bottom is deliberately simple and used by tests
//! to prove the plan is equivalent to a naive per-route loop.

use std::collections::BTreeMap;

/// Number of elements in one GGML K-quant block (Q2_K/Q3_K/Q4_K/...).
pub const QK_K: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExpertKey {
    pub layer: u32,
    pub expert: u32,
}

impl ExpertKey {
    pub const fn new(layer: u32, expert: u32) -> Self {
        Self { layer, expert }
    }
}

/// Where an expert's weights live for this dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Placement {
    /// Weights already resident on the compute device; one grouped device kernel per expert.
    GpuResident,
    /// Weights on the host; one grouped host call per expert.
    CpuHost,
    /// Slot has been admitted but the upload has not completed; the route must wait
    /// rather than compute on stale weights. Never silently treated as CPU or GPU.
    Upload,
}

impl Placement {
    fn rank(self) -> u8 {
        match self {
            Placement::GpuResident => 0,
            Placement::CpuHost => 1,
            Placement::Upload => 2,
        }
    }
}

/// One routed token->expert assignment.
///
/// `ordinal` disambiguates multiple assignments for the same token (top-k routing);
/// `(token, ordinal)` must be unique within one dispatch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Route {
    pub token: u32,
    pub ordinal: u32,
    pub key: ExpertKey,
    pub weight: f32,
}

/// One execution group: all routes for one expert under one placement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteGroup {
    pub key: ExpertKey,
    pub placement: Placement,
    /// Positions in the sorted route permutation (`DispatchPlan::order`).
    pub start: usize,
    pub len: usize,
}

/// Validated, deterministic dispatch plan.
#[derive(Debug, Clone, PartialEq)]
pub struct DispatchPlan {
    /// A permutation of `0..routes.len()`: the routes sorted into group order.
    order: Vec<usize>,
    groups: Vec<RouteGroup>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanLimits {
    /// Number of tokens in the current forward pass. Every route token must be < token_count.
    pub token_count: u32,
    /// Maximum routes in one group; a larger expert population is split into
    /// consecutive groups. 0 means "no limit".
    pub max_group_routes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanError {
    TokenOutOfRange {
        token: u32,
        token_count: u32,
    },
    NonFiniteWeight {
        token: u32,
        ordinal: u32,
    },
    DuplicateRoute {
        token: u32,
        ordinal: u32,
    },
    /// A group whose weights have not finished uploading must not be executed.
    UploadGroupNotReady {
        layer: u32,
        expert: u32,
    },
    Overflow,
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanError::TokenOutOfRange { token, token_count } => write!(
                formatter,
                "route token {token} is outside the forward pass ({token_count} tokens)"
            ),
            PlanError::NonFiniteWeight { token, ordinal } => write!(
                formatter,
                "route ({token}, {ordinal}) has a non-finite weight"
            ),
            PlanError::DuplicateRoute { token, ordinal } => write!(
                formatter,
                "duplicate route ({token}, {ordinal}); a route may appear once per dispatch"
            ),
            PlanError::UploadGroupNotReady { layer, expert } => write!(
                formatter,
                "expert (layer {layer}, expert {expert}) is still uploading; its routes cannot execute"
            ),
            PlanError::Overflow => formatter.write_str("route index overflow"),
        }
    }
}

impl std::error::Error for PlanError {}

impl DispatchPlan {
    pub fn groups(&self) -> &[RouteGroup] {
        &self.groups
    }

    pub fn order(&self) -> &[usize] {
        &self.order
    }

    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    pub fn route_count(&self) -> usize {
        self.order.len()
    }

    /// Iterate groups for one placement, in deterministic group order.
    pub fn groups_for(&self, placement: Placement) -> impl Iterator<Item = &RouteGroup> {
        self.groups
            .iter()
            .filter(move |group| group.placement == placement)
    }

    /// The sorted positions belonging to one group.
    pub fn positions(&self, group: &RouteGroup) -> &[usize] {
        &self.order[group.start..group.start + group.len]
    }

    /// Internal consistency: group ranges cover `order` exactly, in order.
    pub fn validate_ranges(&self) -> Result<(), PlanError> {
        let mut expected = 0usize;
        for group in &self.groups {
            if group.start != expected || group.len == 0 {
                return Err(PlanError::Overflow);
            }
            expected = group
                .start
                .checked_add(group.len)
                .ok_or(PlanError::Overflow)?;
        }
        if expected != self.order.len() {
            return Err(PlanError::Overflow);
        }
        Ok(())
    }

    /// Prove conservation against the input route count: every input index appears
    /// exactly once in the plan. This is the property the engine relies on.
    pub fn validate_conservation(&self, route_count: usize) -> Result<(), PlanError> {
        if route_count != self.order.len() {
            return Err(PlanError::Overflow);
        }
        let mut seen = vec![false; route_count];
        for &index in &self.order {
            let slot = seen.get_mut(index).ok_or(PlanError::Overflow)?;
            if *slot {
                return Err(PlanError::DuplicateRoute {
                    token: index as u32,
                    ordinal: 0,
                });
            }
            *slot = true;
        }
        self.validate_ranges()
    }
}

/// Build a deterministic plan.
///
/// Groups are ordered by `(placement rank, layer, expert)`; routes inside a group
/// by `(token, ordinal)`; a group larger than `limits.max_group_routes` is split
/// into consecutive groups (`max_group_routes == 0` disables the cap, yielding one
/// group per `(placement, expert)`). Duplicate `(token, ordinal)` pairs are rejected because
/// the engine's supervision receipts key per-token contributions by that pair.
pub fn plan<F>(
    routes: &[Route],
    placement_of: F,
    limits: PlanLimits,
) -> Result<DispatchPlan, PlanError>
where
    F: Fn(ExpertKey) -> Placement,
{
    // Validate first so no partially built plan can be observed.
    let mut unique = std::collections::BTreeSet::new();
    for route in routes {
        if route.token >= limits.token_count {
            return Err(PlanError::TokenOutOfRange {
                token: route.token,
                token_count: limits.token_count,
            });
        }
        if !route.weight.is_finite() {
            return Err(PlanError::NonFiniteWeight {
                token: route.token,
                ordinal: route.ordinal,
            });
        }
        if !unique.insert((route.token, route.ordinal)) {
            return Err(PlanError::DuplicateRoute {
                token: route.token,
                ordinal: route.ordinal,
            });
        }
    }

    // Sort by group key first, then route key: deterministic permutation.
    let mut order: Vec<usize> = (0..routes.len()).collect();
    order.sort_by_key(|&index| {
        let route = &routes[index];
        let placement = placement_of(route.key);
        (
            placement.rank(),
            route.key.layer,
            route.key.expert,
            route.token,
            route.ordinal,
        )
    });

    let mut groups: Vec<RouteGroup> = Vec::new();
    let mut cursor = 0usize;
    while cursor < order.len() {
        let first = &routes[order[cursor]];
        let placement = placement_of(first.key);
        let mut end = cursor + 1;
        while end < order.len() {
            let route = &routes[order[end]];
            if route.key != first.key || placement_of(route.key) != placement {
                break;
            }
            end += 1;
        }
        let mut start = cursor;
        while start < end {
            let remaining = end - start;
            let len = if limits.max_group_routes == 0 {
                remaining
            } else {
                remaining.min(limits.max_group_routes)
            };
            groups.push(RouteGroup {
                key: first.key,
                placement,
                start,
                len,
            });
            start += len;
        }
        cursor = end;
    }

    Ok(DispatchPlan { order, groups })
}

/// Convenience: group by key without caring about placement policy. All routes
/// are treated as CPU-host groups; useful for pure-CPU tests and for engines that
/// already partitioned by device.
pub fn plan_cpu(routes: &[Route], limits: PlanLimits) -> Result<DispatchPlan, PlanError> {
    plan(routes, |_| Placement::CpuHost, limits)
}

/// Number of groups and routes per placement, for receipts.
pub fn summarize(plan: &DispatchPlan) -> BTreeMap<Placement, (usize, usize)> {
    let mut summary: BTreeMap<Placement, (usize, usize)> = BTreeMap::new();
    for group in plan.groups() {
        let entry = summary.entry(group.placement).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += group.len;
    }
    summary
}

// ---------------------------------------------------------------------------
// Reference executor (normative oracle for tests; not the optimized path)
// ---------------------------------------------------------------------------

/// Computes one expert's contribution for a group of token rows.
///
/// `rows` is the ordered list of input positions for this group; `tokens` is the
/// full `[token_count, hidden]` row-major matrix. Return `rows.len() * hidden`
/// values in the same order as `rows`.
pub trait GroupedExpert {
    fn compute(&self, key: ExpertKey, rows: &[usize], tokens: &[f32], hidden: usize) -> Vec<f32>;
}

/// Execute a plan with grouped expert calls and weighted scatter-add.
///
/// This is the reference semantics the engine must match: gather each group, call
/// the expert once with all rows, scatter each result back multiplied by its route
/// weight. `Upload` groups are rejected because their weights are not ready.
pub fn execute_reference<E: GroupedExpert>(
    plan: &DispatchPlan,
    routes: &[Route],
    tokens: &[f32],
    hidden: usize,
    expert: &E,
) -> Result<Vec<f32>, PlanError> {
    plan.validate_conservation(routes.len())?;
    let token_count = tokens.len() / hidden;
    let mut output = vec![0f32; tokens.len()];
    for group in plan.groups() {
        if group.placement == Placement::Upload {
            return Err(PlanError::UploadGroupNotReady {
                layer: group.key.layer,
                expert: group.key.expert,
            });
        }
        let positions = plan.positions(group);
        let rows: Vec<usize> = positions
            .iter()
            .map(|&i| routes[i].token as usize)
            .collect();
        let contributions = expert.compute(group.key, &rows, tokens, hidden);
        if contributions.len() != rows.len() * hidden {
            return Err(PlanError::Overflow);
        }
        for (position_index, &route_index) in positions.iter().enumerate() {
            let route = &routes[route_index];
            let token = route.token as usize;
            if token >= token_count {
                return Err(PlanError::TokenOutOfRange {
                    token: route.token,
                    token_count: token_count as u32,
                });
            }
            let contribution =
                &contributions[position_index * hidden..(position_index + 1) * hidden];
            let destination = &mut output[token * hidden..(token + 1) * hidden];
            for (dst, value) in destination.iter_mut().zip(contribution) {
                *dst += *value * route.weight;
            }
        }
    }
    Ok(output)
}

/// Naive per-route executor used as the oracle in tests and parity checks.
pub fn execute_naive<E: GroupedExpert>(
    routes: &[Route],
    tokens: &[f32],
    hidden: usize,
    expert: &E,
) -> Vec<f32> {
    let mut output = vec![0f32; tokens.len()];
    for route in routes {
        let rows = [route.token as usize];
        let contribution = expert.compute(route.key, &rows, tokens, hidden);
        let destination =
            &mut output[route.token as usize * hidden..(route.token as usize + 1) * hidden];
        for (dst, value) in destination.iter_mut().zip(&contribution) {
            *dst += *value * route.weight;
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(layer: u32, expert: u32) -> ExpertKey {
        ExpertKey::new(layer, expert)
    }

    fn route(token: u32, ordinal: u32, layer: u32, expert: u32, weight: f32) -> Route {
        Route {
            token,
            ordinal,
            key: key(layer, expert),
            weight,
        }
    }

    fn limits(token_count: u32, max_group_routes: usize) -> PlanLimits {
        PlanLimits {
            token_count,
            max_group_routes,
        }
    }

    #[test]
    fn groups_one_per_expert_and_placement() {
        let routes = vec![
            route(0, 0, 0, 1, 0.5),
            route(1, 0, 0, 1, 0.5),
            route(1, 1, 0, 2, 0.25),
            route(2, 0, 1, 0, 1.0),
        ];
        let placements = |k: ExpertKey| {
            if k.layer == 0 {
                Placement::GpuResident
            } else {
                Placement::CpuHost
            }
        };
        let plan = plan(&routes, placements, limits(3, 0)).unwrap();
        assert_eq!(plan.groups().len(), 3);
        assert_eq!(plan.groups_for(Placement::GpuResident).count(), 2);
        assert_eq!(plan.groups_for(Placement::CpuHost).count(), 1);
        plan.validate_conservation(routes.len()).unwrap();
        let summary = summarize(&plan);
        assert_eq!(summary[&Placement::GpuResident], (2, 3));
        assert_eq!(summary[&Placement::CpuHost], (1, 1));
    }

    #[test]
    fn deterministic_under_input_permutation() {
        let routes = vec![
            route(3, 0, 2, 7, 0.1),
            route(1, 0, 0, 0, 0.2),
            route(2, 1, 2, 7, 0.3),
            route(0, 0, 0, 0, 0.4),
        ];
        let mut permuted = routes.clone();
        permuted.reverse();
        let a = plan_cpu(&routes, limits(4, 0)).unwrap();
        let b = plan_cpu(&permuted, limits(4, 0)).unwrap();
        let route_tuple = |plan: &DispatchPlan, source: &[Route]| {
            plan.groups()
                .iter()
                .map(|g| {
                    (
                        g.key,
                        plan.positions(g)
                            .iter()
                            .map(|&i| (source[i].token, source[i].ordinal))
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(route_tuple(&a, &routes), route_tuple(&b, &permuted));
    }

    #[test]
    fn max_group_splits_without_reordering_or_losing_routes() {
        let routes: Vec<Route> = (0..10).map(|token| route(token, 0, 0, 4, 0.5)).collect();
        let plan = plan_cpu(&routes, limits(10, 3)).unwrap();
        assert_eq!(plan.groups().len(), 4);
        assert_eq!(plan.groups()[0].len, 3);
        assert_eq!(plan.groups()[3].len, 1);
        plan.validate_conservation(routes.len()).unwrap();
        // Positions inside a group are sorted by (token, ordinal).
        let positions: Vec<usize> = plan.groups()[0]
            .clone()
            .pipe_positions(&plan)
            .iter()
            .map(|&i| routes[i].token as usize)
            .collect();
        assert_eq!(positions, vec![0, 1, 2]);
    }

    trait PipePositions {
        fn pipe_positions(self, plan: &DispatchPlan) -> Vec<usize>;
    }
    impl PipePositions for RouteGroup {
        fn pipe_positions(self, plan: &DispatchPlan) -> Vec<usize> {
            plan.positions(&self).to_vec()
        }
    }

    #[test]
    fn rejects_out_of_range_token() {
        let routes = vec![route(7, 0, 0, 0, 1.0)];
        let error = plan_cpu(&routes, limits(7, 0)).unwrap_err();
        assert_eq!(
            error,
            PlanError::TokenOutOfRange {
                token: 7,
                token_count: 7
            }
        );
    }

    #[test]
    fn rejects_non_finite_weight() {
        let routes = vec![route(0, 0, 0, 0, f32::NAN)];
        assert!(matches!(
            plan_cpu(&routes, limits(1, 0)).unwrap_err(),
            PlanError::NonFiniteWeight { .. }
        ));
    }

    #[test]
    fn rejects_duplicate_token_ordinal() {
        let routes = vec![route(0, 1, 0, 0, 1.0), route(0, 1, 0, 1, 1.0)];
        assert!(matches!(
            plan_cpu(&routes, limits(1, 0)).unwrap_err(),
            PlanError::DuplicateRoute { .. }
        ));
    }

    #[test]
    fn zero_group_limit_means_no_cap() {
        let empty: Vec<Route> = Vec::new();
        assert!(plan_cpu(&empty, limits(0, 0)).is_ok());
        let routes: Vec<Route> = (0..5).map(|token| route(token, 0, 0, 0, 1.0)).collect();
        let plan = plan_cpu(&routes, limits(5, 0)).unwrap();
        assert_eq!(plan.groups().len(), 1);
        assert_eq!(plan.groups()[0].len, 5);
        plan.validate_conservation(routes.len()).unwrap();
    }

    #[test]
    fn empty_plan_is_valid() {
        let plan = plan_cpu(&[], limits(0, 0)).unwrap();
        assert!(plan.is_empty());
        assert_eq!(plan.route_count(), 0);
        plan.validate_conservation(0).unwrap();
        assert!(plan.groups_for(Placement::GpuResident).next().is_none());
    }

    #[test]
    fn upload_routes_are_isolated_and_never_silently_coerced() {
        let routes = vec![route(0, 0, 0, 0, 1.0), route(1, 0, 0, 0, 1.0)];
        let plan = plan(&routes, |_| Placement::Upload, limits(2, 0)).unwrap();
        assert_eq!(plan.groups().len(), 1);
        assert_eq!(plan.groups()[0].placement, Placement::Upload);
    }

    fn sample_tokens() -> (Vec<f32>, usize) {
        (vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2)
    }

    struct DeterministicExpert;
    impl GroupedExpert for DeterministicExpert {
        fn compute(
            &self,
            key: ExpertKey,
            rows: &[usize],
            tokens: &[f32],
            hidden: usize,
        ) -> Vec<f32> {
            let mut out = Vec::with_capacity(rows.len() * hidden);
            for &row in rows {
                for column in 0..hidden {
                    let x = tokens[row * hidden + column];
                    out.push(x * (1.0 + key.expert as f32) + key.layer as f32);
                }
            }
            out
        }
    }

    #[test]
    fn grouped_execution_matches_the_naive_per_route_oracle() {
        let (tokens, hidden) = sample_tokens();
        let routes = vec![
            route(0, 0, 0, 1, 0.5),
            route(1, 0, 0, 1, 0.25),
            route(1, 1, 0, 2, 0.75),
            route(2, 0, 1, 0, 1.0),
            route(2, 1, 0, 1, 0.5),
        ];
        let plan = plan_cpu(&routes, limits(3, 0)).unwrap();
        let grouped =
            execute_reference(&plan, &routes, &tokens, hidden, &DeterministicExpert).unwrap();
        let naive = execute_naive(&routes, &tokens, hidden, &DeterministicExpert);
        assert_eq!(grouped, naive);
    }

    #[test]
    fn reference_executor_rejects_upload_groups() {
        let (tokens, hidden) = sample_tokens();
        let routes = vec![route(0, 0, 0, 0, 1.0)];
        let plan = plan(&routes, |_| Placement::Upload, limits(3, 0)).unwrap();
        assert_eq!(
            execute_reference(&plan, &routes, &tokens, hidden, &DeterministicExpert).unwrap_err(),
            PlanError::UploadGroupNotReady {
                layer: 0,
                expert: 0
            }
        );
    }
}
