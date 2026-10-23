//! Long-context KV streaming plan (Q-038).
//!
//! A sequence's blocks are split into a device tier (attention-sink blocks plus
//! the most recent window) and a host tier (the cold middle). Nothing is dropped:
//! the two tiers always partition the full block table, so attention still sees
//! every token and output tokens can be exactly equal to the short-context run.
//! Each decode step streams every host block, one layer at a time, through
//! `prefetch_depth` staging buffers.
//!
//! [`BlockMover`] tracks where each block is and refuses any read of a block
//! that is not on the device or staged, and any move of a block that is leased.

use crate::kv::KvLayout;
use crate::MemoryError;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamConfig {
    pub layout: KvLayout,
    /// Device blocks one sequence may keep resident.
    pub device_blocks: u64,
    /// Leading blocks always kept on the device (attention sinks).
    pub sink_blocks: u64,
    /// Layers of host blocks staged ahead of the layer being computed.
    pub prefetch_depth: u64,
    /// Measured host-to-device bandwidth; `None` is `UNKNOWN`.
    pub host_to_device_bytes_per_s: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tiering {
    pub device: Vec<u64>,
    pub host: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepPlan {
    pub tiering: Tiering,
    /// Bytes copied host-to-device per layer per decode step.
    pub bytes_per_layer: u64,
    /// Bytes copied per decode step over all layers.
    pub bytes_per_step: u64,
    /// Device staging bytes needed for `prefetch_depth` layers in flight.
    pub staging_bytes: u64,
    /// Transfer-time lower bound in microseconds; `None` when bandwidth is `UNKNOWN`.
    /// An estimate for planning only, never a stall measurement.
    pub transfer_floor_us: Option<u64>,
}

impl StreamConfig {
    pub fn tiering(&self, total_blocks: u64) -> Result<Tiering, MemoryError> {
        if self.device_blocks <= self.sink_blocks {
            return Err(MemoryError::InvalidRequest);
        }
        if total_blocks <= self.device_blocks {
            return Ok(Tiering {
                device: (0..total_blocks).collect(),
                host: Vec::new(),
            });
        }
        let recent = self.device_blocks - self.sink_blocks;
        let mut device: Vec<u64> = (0..self.sink_blocks).collect();
        device.extend(total_blocks - recent..total_blocks);
        Ok(Tiering {
            device,
            host: (self.sink_blocks..total_blocks - recent).collect(),
        })
    }

    pub fn step_plan(&self, tokens: u64) -> Result<StepPlan, MemoryError> {
        let tiering = self.tiering(self.layout.blocks_for(tokens))?;
        let per_block_layer = self.layout.bytes_per_block_layer()?;
        let host = tiering.host.len() as u64;
        let bytes_per_layer = host
            .checked_mul(per_block_layer)
            .ok_or(MemoryError::Overflow)?;
        let bytes_per_step = bytes_per_layer
            .checked_mul(self.layout.layers)
            .ok_or(MemoryError::Overflow)?;
        let staging_bytes = bytes_per_layer
            .checked_mul(self.prefetch_depth.max(1))
            .ok_or(MemoryError::Overflow)?;
        let transfer_floor_us = self
            .host_to_device_bytes_per_s
            .filter(|&bw| bw > 0)
            .map(|bw| (bytes_per_step as u128 * 1_000_000).div_ceil(bw as u128) as u64);
        Ok(StepPlan {
            tiering,
            bytes_per_layer,
            bytes_per_step,
            staging_bytes,
            transfer_floor_us,
        })
    }

    /// Longest context whose device tier plus staging fits in `device_bytes`.
    pub fn max_tokens_for(&self, device_bytes: u64, host_bytes: u64) -> Result<u64, MemoryError> {
        let block = self.layout.bytes_per_block()?;
        let device_need = self
            .device_blocks
            .checked_mul(block)
            .ok_or(MemoryError::Overflow)?;
        if device_need > device_bytes {
            return Ok(0);
        }
        let staging_left = device_bytes - device_need;
        let per_layer_block = self.layout.bytes_per_block_layer()?.max(1);
        let host_by_staging = staging_left / (per_layer_block * self.prefetch_depth.max(1));
        let host_by_capacity = host_bytes / block.max(1);
        let blocks = self.device_blocks + host_by_staging.min(host_by_capacity);
        Ok(blocks * self.layout.block_tokens)
    }
}

/// Where one KV block is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Location {
    Device,
    Host,
    ToHost(u64),
    ToDevice(u64),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoveTicket {
    block: u64,
    nonce: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadLease {
    block: u64,
    nonce: u64,
}

#[derive(Debug, Default)]
pub struct BlockMover {
    blocks: BTreeMap<u64, (Location, u32)>,
    leases: BTreeMap<u64, u64>,
    nonce: u64,
}

impl BlockMover {
    pub fn new(blocks: impl IntoIterator<Item = u64>) -> Self {
        Self {
            blocks: blocks
                .into_iter()
                .map(|b| (b, (Location::Device, 0)))
                .collect(),
            ..Self::default()
        }
    }

    fn next(&mut self) -> u64 {
        self.nonce += 1;
        self.nonce
    }

    pub fn location(&self, block: u64) -> Result<Location, MemoryError> {
        Ok(self.blocks.get(&block).ok_or(MemoryError::NotFound)?.0)
    }

    fn begin(
        &mut self,
        block: u64,
        from: Location,
        to: fn(u64) -> Location,
    ) -> Result<MoveTicket, MemoryError> {
        let nonce = self.next();
        let entry = self.blocks.get_mut(&block).ok_or(MemoryError::NotFound)?;
        if entry.0 != from || entry.1 != 0 {
            return Err(MemoryError::Busy);
        }
        entry.0 = to(nonce);
        Ok(MoveTicket { block, nonce })
    }

    pub fn begin_evict(&mut self, block: u64) -> Result<MoveTicket, MemoryError> {
        self.begin(block, Location::Device, Location::ToHost)
    }

    pub fn begin_fetch(&mut self, block: u64) -> Result<MoveTicket, MemoryError> {
        self.begin(block, Location::Host, Location::ToDevice)
    }

    /// Complete a move; a failed move leaves the block where it was.
    pub fn complete(&mut self, ticket: MoveTicket, ok: bool) -> Result<(), MemoryError> {
        let entry = self
            .blocks
            .get_mut(&ticket.block)
            .ok_or(MemoryError::NotFound)?;
        entry.0 = match (entry.0, ok) {
            (Location::ToHost(n), true) if n == ticket.nonce => Location::Host,
            (Location::ToHost(n), false) if n == ticket.nonce => Location::Device,
            (Location::ToDevice(n), true) if n == ticket.nonce => Location::Device,
            (Location::ToDevice(n), false) if n == ticket.nonce => Location::Host,
            _ => return Err(MemoryError::NotFound),
        };
        Ok(())
    }

    /// Lease a block for an attention read; only device-resident blocks can be read.
    pub fn lease(&mut self, block: u64) -> Result<ReadLease, MemoryError> {
        let nonce = self.next();
        let entry = self.blocks.get_mut(&block).ok_or(MemoryError::NotFound)?;
        if entry.0 != Location::Device {
            return Err(MemoryError::Busy);
        }
        entry.1 += 1;
        self.leases.insert(nonce, block);
        Ok(ReadLease { block, nonce })
    }

    pub fn release(&mut self, lease: ReadLease) -> Result<(), MemoryError> {
        if self.leases.remove(&lease.nonce) != Some(lease.block) {
            return Err(MemoryError::NotFound);
        }
        self.blocks
            .get_mut(&lease.block)
            .ok_or(MemoryError::NotFound)?
            .1 -= 1;
        Ok(())
    }
}
