# EngineProvider v1 — draft

Status: **Q-024a draft.** v0 remains the qualified surface
(`contracts/ENGINE_PROVIDER_V0.md`, `harness/engine-provider-contract/src/lib.rs`).
v1 is **additive**: it is a superset that a v0 adapter can satisfy by
construction, so no qualified v0 evidence is invalidated by adopting it.
The reference implementation is `harness/engine-provider-contract/src/v1.rs`; it
implements the types below and the mechanical conformance checks, and nothing in
it is a qualification.

Scope of this draft: the minimum interface change that lets one supervisor drive
**several kinds of local engine plus optional remote engines** without gaining
materialization authority. It is a contract, not an implementation, and it does
not qualify anything by existing.

## 0. Core invariant (unchanged, restated because v1 adds surface area)

> The supervisor controls intent, lifecycle, and resource envelopes.
> The engine controls materialization.

Every addition below is checked against one question: *does it let the
supervisor describe what it wants, or does it let the supervisor decide how the
engine does it?* Anything in the second category is rejected — including when it
would be convenient for multi-engine routing.

## 1. What v0 cannot express (the gap v1 closes)

| Gap | v0 today | Why it blocks versatility |
| --- | --- | --- |
| Several engines, one supervisor | no registry; one trait object | selecting between mistralrs-strata, a C++ Strata/llama.cpp server and a one-shot CLI is call-site code, not an admitted decision with a receipt |
| Negotiation | `DeclaredCapability { key: String, value: String }` | string keys cannot be matched safely; a typo becomes a silent non-match |
| Streaming | `generate` returns `GenerationReceipt { text: String }` | no TTFT/ITL, no cancellation point, no partial output; Q-016 cannot be measured through the provider |
| Async / overlap | synchronous `&mut self` | long generations block the supervisor's only thread |
| Error taxonomy | `ProviderError { message: String }` | the supervisor's fail-closed policy cannot distinguish `NotReady` from `EngineFault` from `TransportFault` |
| Admission facts | `observe()` after load only | admission (Q-015) needs pre-load, freshness-stamped facts |
| Request identity | none | no cancellation, no deduplication, no per-request evidence |

## 2. Provider registry (supervisor lane)

The registry is **selection**, not control. It is static configuration plus
qualified facts; there is no discovery protocol, no provider-to-provider call and
no provider-initiated registration.

```rust
pub struct ProviderId(pub String);

pub struct ProviderEntry {
    pub id: ProviderId,
    /// v0 EngineIdentity: name, revision, binary digest, transport, provenance.
    pub identity: EngineIdentity,
    /// Qualification profile the evidence below was produced under.
    pub profile: String,
    pub qualified: Vec<QualifiedCapability>,
    pub declared: Vec<DeclaredCapability>,
    pub conformance: Option<ConformanceReceipt>,
    pub enabled: bool,
}

pub struct ProviderRegistry {
    entries: Vec<ProviderEntry>,
}
```

Selection produces a receipt, never a silent fallback:

```rust
pub struct SelectionRequest {
    pub required: Vec<CapabilityRequirement>,
    pub preferred: Vec<CapabilityKey>,
    pub model: ModelArtifact,
    pub profile: String,           // e.g. "RTX3070-8GIB-LOCAL"
}

pub struct SelectionReceipt {
    pub chosen: Option<ProviderId>,
    pub rejected: Vec<(ProviderId, SelectionRejection)>,
    pub decided_at_ms: u64,
}
```

Rules:

- A provider is a candidate only if its `enabled` flag is true **and** the
  required capabilities are `Qualified` under the **same** profile. `Declared`
  never satisfies `required`.
- If exactly one candidate survives, it is chosen. If none survives, the receipt
  carries every rejection reason; the supervisor fails closed.
- If several survive, `preferred` orders them, and the receipt records the order.
  Ties are broken by an explicit, recorded rule (registry order) — never by
  timing.
- A `SelectionReceipt` is **not** a load grant. Loading still needs
  `LoadIntent` + admission.

## 3. Capability negotiation (typed, declared vs qualified preserved)

```rust
pub enum CapabilityKey {
    // response surface
    Transport,            // value: EngineTransport
    ServingMode,          // Persistent | OneShot
    Streaming,            // full-duplex token events
    Cancellation,
    // request features
    ConstrainedDecoding,  // regex | json_schema | ebnf
    MultiChoiceN,
    PromptLogprobs,
    TokenLogprobs,
    // resource/limits facts the engine can honour
    MaxContextTokens,
    MaxConcurrency,
    KvBytes,              // observable KV size
    // provenance
    EngineRevision,
}

pub enum CapabilityValue {
    Bool(bool),
    UInt(u64),            // parsed, not stringly
    Text(String),         // only for genuinely textual facts
    Enum(u16),            // transport / mode discriminants
}

pub struct CapabilityRequirement {
    pub key: CapabilityKey,
    pub at_least: Option<CapabilityValue>,   // for numeric minimums
    pub exact: Option<CapabilityValue>,
}

pub enum Negotiation {
    Accepted,
    Unsatisfied(Vec<CapabilityKey>),
    /// Engine said nothing about a required key.
    Unknown(Vec<CapabilityKey>),
    /// Declared and qualified disagree in a way that forbids use.
    Conflict(Vec<CapabilityKey>),
}
```

Rules:

- `Unknown` is a distinct result and **never** satisfies a requirement. This
  mirrors v0's `Fact::Unknown` rule: "a missing observation must never be
  replaced with a plausible default."
- `Declared` and `Qualified` stay separate fields; negotiation reads `Qualified`
  only. v0's example (`declared context = 4096`, `qualified context = 384`) is a
  valid v1 state, not a contradiction.
- Numeric comparisons are numeric. No string-parsed capacities.

## 4. Async and streaming (no async runtime, object-safe)

The contract crate is intentionally dependency-free
(`harness/engine-provider-contract/Cargo.toml` has an empty `[dependencies]`).
`async fn` in a trait is not dyn-compatible without boxing, and adding a runtime
(tokio/async-std) to a reference model would be a heavier change than v1 needs.
v1 therefore uses **start/poll + sink**, which any adapter can implement and a
caller can wrap in `async` at its own edge.

```rust
pub struct RequestId(pub u64);

pub enum StreamEvent {
    Token {
        index: u32,
        token_id: Option<u32>,
        text_delta: String,
        logprob: Option<f32>,
    },
    Usage {
        prompt_tokens: Option<u64>,
        completion_tokens: Option<u64>,
    },
    Finished {
        stop_reason: StopReason,
        receipt: GenerationReceipt,   // terminal, provenance-bearing
    },
    Failed {
        error: ProviderError,
    },
}

pub enum StopReason { Eos, Length, StopSequence, Cancelled, Deadline, EngineFault }

pub enum SinkControl { Continue, Pause }

pub trait TokenSink {
    fn on_event(&mut self, event: StreamEvent) -> SinkControl;
}

pub trait EngineProviderStream: EngineProvider {
    fn start_generate(
        &mut self,
        handle: &ModelHandle,
        request: GenerationRequest,
        sink: &mut dyn TokenSink,
    ) -> Result<RequestId, ProviderError>;

    /// Non-blocking: returns what is currently deliverable and whether the
    /// request is still live. Must not block the calling thread.
    fn poll(&mut self, id: RequestId) -> Result<PollOutcome, ProviderError>;

    fn cancel(&mut self, id: RequestId) -> Result<(), ProviderError>;
}
```

Rules:

- `Streaming` is an **optional qualified capability**. A one-shot CLI engine may
  implement `generate` only; it is still a valid v1 provider.
- `poll` must not block. A transport with no non-blocking mode (a local CLI pipe)
  reports `Streaming = not qualified` and uses the synchronous `generate`.
- `SinkControl::Pause` is honour-system backpressure: the provider stops pushing
  until `resume`. An engine that cannot pause simply ignores it; a supervisor
  must not assume backpressure exists unless `stream_pause` is qualified.
- Token `index` is monotonic per request and starts at 0. Events are never
  reordered. `Finished`/`Failed` is terminal and exclusive.
- `cancel` is idempotent; cancelling an unknown/finished id returns
  `ProviderErrorKind::Cancelled` (`retryable = false`) or `Ok(())`; it never
  faults the instance.
- The terminal `GenerationReceipt` in `Finished` carries the same provenance
  fields as v0 (`Fact<u64>` for token counts and TTFT). A streaming provider that
  cannot measure TTFT reports `Fact::Unknown`, not `0`.

## 5. Error taxonomy

```rust
pub enum ProviderErrorKind {
    IdentityMismatch,          // binary/model digest does not match the loaded instance
    UnknownModel,
    AdmissionRejected,         // resource envelope or admission policy said no
    CapabilityUnsatisfied,
    NotReady,                  // state machine not yet READY(SEMANTIC) for the operation
    Busy,                      // transient capacity refusal
    Cancelled,
    DeadlineExceeded,
    EngineFault,               // engine reports its own failure
    DriverFault,               // host-observed GPU/driver fault (Xid, ECC, reset)
    TransportFault,            // socket/pipe/process boundary
    ProtocolViolation,         // peer answered something the contract forbids
    Internal,
}

pub struct ProviderError {
    pub kind: ProviderErrorKind,
    pub message: String,          // human text; never the only classification
    pub retryable: bool,          // explicit, not inferred from kind by callers
    pub retry_after_ms: Fact<u64>,
    pub observed_at_ms: u64,
}
```

Rules:

- No `Other`. An adapter that cannot classify must say `Internal` and is
  therefore visibly unqualified, rather than quietly passing a string.
- `TransportFault` must never be reported as `NotReady`; "the socket is down" and
  "the model has not loaded" have different remediation and different evidence.
- `DriverFault` requires a host-sourced observation (v0's `Fact` provenance); it
  may not be inferred from an engine string.
- Callers must not decide retry from `message`. The fail-closed rule is: unknown
  kind ⇒ do not retry.

## 6. Admission facts (pre-load, freshness-stamped)

```rust
pub struct AdmissionFacts {
    pub handle: Option<ModelHandle>,     // None before load
    pub sampled_at_ms: u64,
    pub ttl_ms: u64,
    pub gpu_free_bytes: Fact<u64>,
    pub host_free_bytes: Fact<u64>,
    pub driver_health: Fact<DriverHealth>,
    pub admitted_context_tokens: Fact<u64>,
    pub admitted_concurrency: Fact<u32>,
}

pub enum DriverHealth { NoXid, Xid(u32), Reset, Unknown }
```

Rules:

- The **provider reports**; the **supervisor decides**. No admission verdict is
  returned by the provider, because admission is a supervisor authority
  (Q-015).
- Freshness: a fact sampled more than `ttl_ms` ago is stale. The supervisor must
  treat stale as `Unknown` and deny.
- Pre-load, `handle` is `None` and everything except host/driver facts may be
  `Unknown`; that is a valid, deny-by-default state.
- The provider must not reserve, pre-allocate or "helpfully" adjust the envelope.

## 7. Conformance harness (what makes "implements the interface" checkable)

"Implements the trait" is not qualification (v0 says so explicitly). v1 adds a
generic, dependency-free **mechanical** conformance harness so every provider is
checked against the same rules before profile qualification:

```rust
pub struct ConformanceCheck {
    pub id: &'static str,
    pub passed: bool,
    pub detail: String,
}
pub struct ConformanceReport { pub checks: Vec<ConformanceCheck>, pub passed: bool }
pub fn provider_conformance<P: EngineProvider>(p: &mut P) -> ConformanceReport;
```

Implemented checks (no GPU, no model, no load):

1. `identity_stable` — `identity()` is stable across calls and names the engine.
2. `declared_and_qualified_stay_separate` — declared records carry a source;
   qualified records carry a non-empty `evidence_ref` and `profile`.
3. `lifecycle_model_is_enforced` — `Serving` is unreachable except through the v0
   transition graph; `Stopped -> Serving` and `Offline -> Serving` are refused.
4. `unknown_survives_as_fact` — `Fact::Unknown` does not become a default.
5. `error_taxonomy_is_classifiable` — 13 explicit `ProviderErrorKind`s, unique
   labels, no `Other`.
6. `stream_contract_is_checkable` — `validate_stream_events` enforces monotonic
   token indices from 0 and exactly one terminal event (via the public helper).
7. `registry_rejects_duplicate_ids` — the selection surface cannot contain an
   ambiguous identity.

Checks that require a live instance (a real load, semantic probe, drain,
unload/reclaim) stay where v0 put them: **profile qualification through evidence**,
not a trait. There is no generic "the provider is good" result. The harness may
also be applied inside a test with a mock provider; that is what Q-024a's own
tests do.

## 8. Non-goals (v1 adds nothing to this list)

Still deliberately absent, now and in v1:

- `move_layer_to_gpu()`, `set_gpu_layer_count()`, `map_device_page()`,
  `evict_kv_block()`, `promote_radix_node()`, `construct_token_batch()`,
  `select_kernel()`
- any method that selects an expert, allocates KV, chooses a batch composition,
  picks a CUDA graph, or reorders requests inside the engine
- provider-to-provider routing or delegation
- a discovery protocol, dynamic registration, or remote configuration push
- secrets, credentials or tool permissions in the provider surface
- a "remote" transport that is trusted because it is remote. A datacenter
  provider (including a hypothetical DeepInfer wrapper) is selectable and
  observable and **still** owns its own materialization; the supervisor gains no
  placement authority by using it.

## 9. Migration and sequencing

The sequence is split deliberately. **Design before Q-013-D; implement after.**
Writing the Q-013-D adapter against v0 would bake v0's missing async, streaming,
cancellation and error taxonomy into the first real adapter, and the adapter
would then justify itself as "the interface". So the contract lands first. But an
*implemented* registry whose providers' lifecycle is not yet authenticated
multiplies an unauthenticated surface, so no provider implementation is
qualified before Q-013-D.

| Step | Change | When | Gate |
| --- | --- | --- | --- |
| 1 | Land v1 types + conformance harness as an additive module; v0 trait unchanged | **before Q-013-D** | `cargo test` on the dependency-free crate; v0 tests still pass |
| 2 | Write the Q-013-D owned-process adapter **against v1**, as the first v1 provider | at Q-013-D | Q-013-D evidence; conformance group 1 green |
| 3 | Add the one-shot CLI provider (no streaming) | after Q-013-D | conformance + one bounded CLI generation with exact receipt |
| 4 | Add the second local engine (mistralrs-strata in-process or C++ Strata server) | after Q-013-D | per-engine qualification profile; no capability inheritance between engines |
| 5 | Optional remote provider | after Q-013-D | separate trust model; never a fallback for a local denial |

This is the resolution of a real disagreement between the two audit lanes: the
supervisor/engine audit argues v1 must land *before* Q-013-D, while
`reports/vendoring-integration-plan.md` first argued v1 must not ship before it.
Both are right about different objects — interface before, implementation after.

## 10. Qualification requirement (unchanged)

A v1 provider is not qualified because it implements v1. Qualification remains
evidence-driven and per profile: load/registration, semantic readiness,
supported concurrency/context, unload/reclaim, fault recovery, driver health.
Everything in this document that is not evidence is a proposal.
