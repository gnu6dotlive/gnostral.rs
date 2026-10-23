use gnostral_memory_plan::envelope::{arbitrate, ArbitrationLog, Component, Envelope, Request};
use gnostral_memory_plan::kv::{CopyOnWrite, KvLayout, KvManager, SeqId};
use gnostral_memory_plan::reserve::{
    Action, Event, Observation, ReserveMachine, ReserveProfile, ReserveState,
};
use gnostral_memory_plan::segments::SegmentedAllocator;
use gnostral_memory_plan::stream::{BlockMover, Location, StreamConfig};
use gnostral_memory_plan::{MemoryError, GIB, MIB};

/// Qwen3-30B-A3B attention geometry (48 layers, 4 KV heads, head_dim 128), f16 KV.
fn qwen3_30b_kv(block_tokens: u64) -> KvLayout {
    KvLayout {
        layers: 48,
        kv_heads: 4,
        head_dim: 128,
        elem_bytes: 2,
        block_tokens,
    }
}

fn rtx3070(ambient: Option<u64>) -> Envelope {
    Envelope {
        device_bytes: 8 * GIB,
        desktop_ambient: ambient,
        desktop_reserve: 256 * MIB,
        min_free: GIB,
    }
}

#[test]
fn envelope_refuses_unknown_ambient_and_tiny_cards() {
    assert_eq!(
        rtx3070(None).available(),
        Err(MemoryError::Unknown {
            what: "desktop_ambient"
        })
    );
    let tiny = Envelope {
        device_bytes: GIB,
        desktop_ambient: Some(GIB),
        desktop_reserve: 0,
        min_free: 0,
    };
    assert!(matches!(
        tiny.available(),
        Err(MemoryError::EnvelopeTooSmall { .. })
    ));
}

#[test]
fn arbitration_grants_minima_then_priority_order() {
    let env = rtx3070(Some(512 * MIB));
    let available = env.available().unwrap();
    assert_eq!(available, 8 * GIB - 512 * MIB - 256 * MIB - GIB);
    let requests = [
        Request {
            component: Component::Experts,
            min: 0,
            desired: 8 * GIB,
        },
        Request {
            component: Component::Kv,
            min: GIB,
            desired: 2 * GIB,
        },
        Request {
            component: Component::Workspace,
            min: 512 * MIB,
            desired: 768 * MIB,
        },
    ];
    let split = arbitrate(&env, &requests).unwrap();
    assert_eq!(split.grant(Component::Workspace), 768 * MIB);
    assert_eq!(split.grant(Component::Kv), 2 * GIB);
    assert_eq!(
        split.grant(Component::Experts),
        available - 768 * MIB - 2 * GIB
    );
    assert_eq!(split.granted() + split.unallocated, available);
    // Same inputs, same split, regardless of request order.
    let mut reversed = requests;
    reversed.reverse();
    assert_eq!(arbitrate(&env, &reversed).unwrap(), split);
}

#[test]
fn arbitration_refuses_impossible_minima_and_bad_requests() {
    let env = rtx3070(Some(512 * MIB));
    let too_much = [Request {
        component: Component::Kv,
        min: 7 * GIB,
        desired: 7 * GIB,
    }];
    assert!(matches!(
        arbitrate(&env, &too_much),
        Err(MemoryError::MinimaExceedEnvelope { .. })
    ));
    let dup = [
        Request {
            component: Component::Kv,
            min: 0,
            desired: 1,
        },
        Request {
            component: Component::Kv,
            min: 0,
            desired: 1,
        },
    ];
    assert_eq!(arbitrate(&env, &dup), Err(MemoryError::InvalidRequest));
    let inverted = [Request {
        component: Component::Kv,
        min: 2,
        desired: 1,
    }];
    assert_eq!(arbitrate(&env, &inverted), Err(MemoryError::InvalidRequest));
    let mut log = ArbitrationLog::default();
    let _ = log.record(1, &env, &too_much);
    let _ = log.record(2, &rtx3070(None), &[]);
    assert_eq!(log.entries().len(), 2);
    assert!(log.entries().iter().all(|(_, r)| r.is_err()));
}

#[test]
fn segmented_allocator_best_fit_coalesce_and_elastic_return() {
    let mut a = SegmentedAllocator::new(1024, 64, 2).unwrap();
    let x = a.alloc(100).unwrap();
    let y = a.alloc(500).unwrap();
    let z = a.alloc(200).unwrap();
    assert_eq!((x.segment, x.offset, x.bytes), (0, 0, 128));
    assert_eq!((y.offset, y.bytes), (128, 512));
    assert_eq!(z.offset, 640);
    a.free(y).unwrap();
    // 150 rounds to 192: the 128-byte tail is too small, the freed 512-byte hole fits.
    let w = a.alloc(150).unwrap();
    assert_eq!((w.segment, w.offset), (0, 128));
    let big = a.alloc(1000).unwrap();
    assert_eq!(big.segment, 1);
    assert_eq!(a.alloc(1000), Err(MemoryError::OutOfSegments));
    assert!(matches!(a.alloc(2000), Err(MemoryError::TooLarge { .. })));
    let s = a.stats();
    assert_eq!(s.segments, 2);
    assert!(s.fragmentation_ppm > 0);
    a.free(big).unwrap();
    assert_eq!(a.free(big), Err(MemoryError::NotFound));
    assert_eq!(a.release_empty(), vec![1]);
    for alloc in [x, z, w] {
        a.free(alloc).unwrap();
    }
    let s = a.stats();
    assert_eq!(
        (
            s.used_bytes,
            s.free_bytes,
            s.largest_free,
            s.fragmentation_ppm
        ),
        (0, 1024, 1024, 0)
    );
    assert_eq!(a.release_empty(), vec![0]);
    assert_eq!(a.stats().reserved_bytes, 0);
}

#[test]
fn segment_cap_cannot_drop_below_open_segments() {
    let mut a = SegmentedAllocator::new(256, 64, 4).unwrap();
    let _x = a.alloc(256).unwrap();
    let _y = a.alloc(256).unwrap();
    assert_eq!(a.set_max_segments(1), Err(MemoryError::Busy));
    a.set_max_segments(2).unwrap();
    assert_eq!(a.alloc(1), Err(MemoryError::OutOfSegments));
    assert!(SegmentedAllocator::new(100, 3, 1).is_err());
}

#[test]
fn kv_layout_arithmetic_for_the_reference_model() {
    let layout = qwen3_30b_kv(16);
    assert_eq!(layout.bytes_per_token().unwrap(), 96 * 1024);
    assert_eq!(layout.bytes_per_block().unwrap(), 16 * 96 * 1024);
    // 32K tokens of f16 KV for this geometry is exactly 3 GiB (arithmetic, not a measurement).
    assert_eq!(32 * 1024 * layout.bytes_per_token().unwrap(), 3 * GIB);
    assert_eq!(layout.blocks_for(17), 2);
    assert_eq!(layout.blocks_in(GIB).unwrap(), GIB / (16 * 96 * 1024));
}

#[test]
fn paged_kv_allocation_append_fork_and_copy_on_write() {
    let mut kv = KvManager::new(qwen3_30b_kv(4), 10).unwrap();
    kv.allocate(SeqId(1), 6).unwrap();
    assert_eq!(kv.block_table(SeqId(1)).unwrap(), &[0, 1]);
    kv.fork(SeqId(1), SeqId(2)).unwrap();
    kv.check_invariants().unwrap();
    // Seq 2 writes into the shared partial tail: copy-on-write into a fresh block.
    let cow = kv.append(SeqId(2), 1).unwrap();
    assert_eq!(cow, Some(CopyOnWrite { src: 1, dst: 2 }));
    assert_eq!(kv.block_table(SeqId(2)).unwrap(), &[0, 2]);
    assert_eq!(kv.block_table(SeqId(1)).unwrap(), &[0, 1]);
    // Seq 1 now owns its tail alone: no copy.
    assert_eq!(kv.append(SeqId(1), 3).unwrap(), None);
    assert_eq!(kv.block_table(SeqId(1)).unwrap(), &[0, 1, 3]);
    kv.check_invariants().unwrap();
    assert_eq!(
        kv.blocks_needed(Some(SeqId(1)), 100).unwrap(),
        25 // 9 -> 109 tokens: 28 blocks of 4, three held, tail not shared
    );
    assert!(matches!(
        kv.append(SeqId(1), 100),
        Err(MemoryError::NoBlocks { .. })
    ));
    kv.check_invariants().unwrap();
    kv.free(SeqId(1)).unwrap();
    kv.free(SeqId(2)).unwrap();
    assert_eq!(kv.free_blocks(), 10);
    kv.check_invariants().unwrap();
    assert_eq!(kv.free(SeqId(2)), Err(MemoryError::NotFound));
}

#[test]
fn paged_kv_is_deterministic_under_churn() {
    let run = || {
        let mut kv = KvManager::new(qwen3_30b_kv(8), 64).unwrap();
        let mut tables = Vec::new();
        for i in 0..20u64 {
            let _ = kv.allocate(SeqId(i), 1 + i * 7 % 30);
            if i % 3 == 2 {
                let _ = kv.free(SeqId(i - 1));
            }
            for s in kv.sequences().collect::<Vec<_>>() {
                let _ = kv.append(s, 5);
            }
            kv.check_invariants().unwrap();
            tables.push(
                kv.sequences()
                    .map(|s| kv.block_table(s).unwrap().to_vec())
                    .collect::<Vec<_>>(),
            );
        }
        tables
    };
    assert_eq!(run(), run());
}

fn stream(device_blocks: u64, bw: Option<u64>) -> StreamConfig {
    StreamConfig {
        layout: qwen3_30b_kv(16),
        device_blocks,
        sink_blocks: 2,
        prefetch_depth: 2,
        host_to_device_bytes_per_s: bw,
    }
}

#[test]
fn tiering_partitions_every_block_exactly_once() {
    let cfg = stream(64, None);
    for total in [1u64, 63, 64, 65, 2048] {
        let t = cfg.tiering(total).unwrap();
        let mut all: Vec<u64> = t.device.iter().chain(&t.host).copied().collect();
        all.sort_unstable();
        assert_eq!(all, (0..total).collect::<Vec<_>>(), "total {total}");
        assert!(t.device.len() as u64 <= 64);
    }
    let t = cfg.tiering(100).unwrap();
    assert_eq!(&t.device[..2], &[0, 1]);
    assert_eq!(*t.device.last().unwrap(), 99);
    assert_eq!(t.host.first(), Some(&2));
    assert!(stream(2, None).tiering(10).is_err());
}

#[test]
fn step_plan_arithmetic_and_unknown_bandwidth() {
    let cfg = stream(64, None);
    let p = cfg.step_plan(32 * 1024).unwrap();
    assert_eq!(p.tiering.host.len(), 2048 - 64);
    let per_layer_block = cfg.layout.bytes_per_block_layer().unwrap();
    assert_eq!(p.bytes_per_layer, (2048 - 64) * per_layer_block);
    assert_eq!(p.bytes_per_step, p.bytes_per_layer * 48);
    assert_eq!(p.staging_bytes, 2 * p.bytes_per_layer);
    assert_eq!(p.transfer_floor_us, None);
    let measured = stream(64, Some(12 * GIB)).step_plan(32 * 1024).unwrap();
    assert!(measured.transfer_floor_us.unwrap() > 0);
    // Short context: everything stays on device, nothing streams.
    assert_eq!(cfg.step_plan(1024).unwrap().bytes_per_step, 0);
    assert!(cfg.max_tokens_for(2 * GIB, 24 * GIB).unwrap() >= 64 * 16);
    assert_eq!(cfg.max_tokens_for(1, 24 * GIB).unwrap(), 0);
}

#[test]
fn block_mover_tickets_and_leases() {
    let mut m = BlockMover::new(0..4);
    let lease = m.lease(1).unwrap();
    assert_eq!(m.begin_evict(1), Err(MemoryError::Busy));
    m.release(lease.clone()).unwrap();
    assert_eq!(m.release(lease), Err(MemoryError::NotFound));
    let t = m.begin_evict(1).unwrap();
    assert!(matches!(m.location(1).unwrap(), Location::ToHost(_)));
    assert_eq!(m.lease(1), Err(MemoryError::Busy));
    m.complete(t.clone(), true).unwrap();
    assert_eq!(m.location(1).unwrap(), Location::Host);
    assert_eq!(m.complete(t, true), Err(MemoryError::NotFound));
    let f = m.begin_fetch(1).unwrap();
    m.complete(f, false).unwrap();
    assert_eq!(m.location(1).unwrap(), Location::Host);
    let f = m.begin_fetch(1).unwrap();
    m.complete(f, true).unwrap();
    assert!(m.lease(1).is_ok());
    assert_eq!(m.begin_fetch(9), Err(MemoryError::NotFound));
}

fn profile() -> ReserveProfile {
    ReserveProfile {
        release_bytes: 512 * MIB,
        release_below_free: 256 * MIB,
        reacquire_above_free: GIB,
        hold_steps: 3,
    }
}

fn obs(step: u64, free: Option<u64>) -> Observation {
    Observation {
        step,
        desktop_free: free,
        driver_clean: Some(true),
    }
}

#[test]
fn reserve_releases_under_pressure_and_reacquires_after_hold() {
    let mut m = ReserveMachine::new(profile());
    assert_eq!(m.observe(obs(1, Some(2 * GIB))), None);
    assert_eq!(m.engine_budget(6 * GIB), 6 * GIB);
    assert_eq!(
        m.observe(obs(2, Some(100 * MIB))),
        Some(Action::Release { bytes: 512 * MIB })
    );
    assert_eq!(m.state(), ReserveState::Releasing);
    m.confirm_release(Ok(()), true).unwrap();
    assert_eq!(m.engine_budget(6 * GIB), 6 * GIB - 512 * MIB);
    assert_eq!(m.observe(obs(3, Some(2 * GIB))), None);
    assert_eq!(m.observe(obs(4, None)), None); // UNKNOWN resets the calm count
    assert_eq!(m.observe(obs(5, Some(2 * GIB))), None);
    assert_eq!(m.observe(obs(6, Some(2 * GIB))), None);
    assert_eq!(
        m.observe(obs(7, Some(2 * GIB))),
        Some(Action::Reacquire { bytes: 512 * MIB })
    );
    m.confirm_reacquire(Err("allocator refused".into()))
        .unwrap();
    assert_eq!(m.state(), ReserveState::Released);
    assert!(m.log().iter().any(|e| matches!(e, Event::Refused { .. })));
    assert!(m
        .log()
        .iter()
        .any(|e| matches!(e, Event::Unknown { step: 4 })));
}

#[test]
fn reserve_faults_on_xid_and_flags_unobserved_release() {
    let mut m = ReserveMachine::new(profile());
    m.observe(obs(1, Some(MIB)));
    m.confirm_release(Ok(()), false).unwrap();
    assert!(m
        .log()
        .iter()
        .any(|e| matches!(e, Event::Unobserved { step: 1 })));
    m.observe(Observation {
        step: 2,
        desktop_free: Some(GIB),
        driver_clean: Some(false),
    });
    assert_eq!(m.state(), ReserveState::Faulted);
    assert_eq!(m.observe(obs(3, Some(MIB))), None);
    assert!(m.clear_fault(false).is_err());
    m.clear_fault(true).unwrap();
    assert_eq!(m.state(), ReserveState::Released);
    assert!(m.confirm_reacquire(Ok(())).is_err());
}
