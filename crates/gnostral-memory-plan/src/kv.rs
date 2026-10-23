//! Paged KV cache bookkeeping (Q-028).
//!
//! [`KvLayout`] turns a model's KV geometry into bytes per token and per block.
//! [`KvManager`] owns a fixed pool of blocks and one block table per sequence.
//! Blocks are reference-counted so a forked sequence shares its parent's prefix;
//! the first append into a shared tail block returns an explicit copy-on-write
//! instruction instead of mutating shared memory. Free blocks are handed out
//! lowest-id first, so the same operation sequence yields the same tables.

use crate::MemoryError;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvLayout {
    pub layers: u64,
    pub kv_heads: u64,
    pub head_dim: u64,
    /// Bytes per stored element (2 for f16/bf16, 1 for fp8).
    pub elem_bytes: u64,
    pub block_tokens: u64,
}

impl KvLayout {
    /// K and V for every layer and KV head.
    pub fn bytes_per_token(&self) -> Result<u64, MemoryError> {
        2u64.checked_mul(self.layers)
            .and_then(|v| v.checked_mul(self.kv_heads))
            .and_then(|v| v.checked_mul(self.head_dim))
            .and_then(|v| v.checked_mul(self.elem_bytes))
            .ok_or(MemoryError::Overflow)
    }
    pub fn bytes_per_block(&self) -> Result<u64, MemoryError> {
        self.bytes_per_token()?
            .checked_mul(self.block_tokens)
            .ok_or(MemoryError::Overflow)
    }
    /// One layer's slice of one block (the unit streamed per layer in Q-038).
    pub fn bytes_per_block_layer(&self) -> Result<u64, MemoryError> {
        Ok(self.bytes_per_block()? / self.layers.max(1))
    }
    pub fn blocks_for(&self, tokens: u64) -> u64 {
        tokens.div_ceil(self.block_tokens.max(1))
    }
    /// Blocks that fit in `bytes`.
    pub fn blocks_in(&self, bytes: u64) -> Result<u64, MemoryError> {
        Ok(bytes / self.bytes_per_block()?.max(1))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SeqId(pub u64);

#[derive(Debug, Clone, PartialEq, Eq)]
struct Sequence {
    blocks: Vec<u32>,
    tokens: u64,
}

/// Copy `src` into the freshly allocated `dst` before writing new tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CopyOnWrite {
    pub src: u32,
    pub dst: u32,
}

#[derive(Debug)]
pub struct KvManager {
    layout: KvLayout,
    refcount: Vec<u32>,
    free: BTreeSet<u32>,
    seqs: BTreeMap<SeqId, Sequence>,
}

impl KvManager {
    pub fn new(layout: KvLayout, blocks: u32) -> Result<Self, MemoryError> {
        if layout.block_tokens == 0 {
            return Err(MemoryError::InvalidRequest);
        }
        Ok(Self {
            layout,
            refcount: vec![0; blocks as usize],
            free: (0..blocks).collect(),
            seqs: BTreeMap::new(),
        })
    }

    pub fn layout(&self) -> KvLayout {
        self.layout
    }
    pub fn free_blocks(&self) -> usize {
        self.free.len()
    }
    pub fn total_blocks(&self) -> usize {
        self.refcount.len()
    }

    fn take(&mut self, n: usize) -> Result<Vec<u32>, MemoryError> {
        if self.free.len() < n {
            return Err(MemoryError::NoBlocks {
                needed: n,
                free: self.free.len(),
            });
        }
        let blocks: Vec<u32> = self.free.iter().take(n).copied().collect();
        for &b in &blocks {
            self.free.remove(&b);
            self.refcount[b as usize] = 1;
        }
        Ok(blocks)
    }

    fn unref(&mut self, block: u32) {
        let rc = &mut self.refcount[block as usize];
        *rc -= 1;
        if *rc == 0 {
            self.free.insert(block);
        }
    }

    /// Blocks needed to grow `seq` (or a new sequence) by `tokens`.
    pub fn blocks_needed(&self, seq: Option<SeqId>, tokens: u64) -> Result<u64, MemoryError> {
        let current = match seq {
            Some(id) => self.seqs.get(&id).ok_or(MemoryError::NotFound)?.tokens,
            None => 0,
        };
        let shared_tail = match seq.and_then(|id| self.seqs.get(&id)) {
            Some(s) if !current.is_multiple_of(self.layout.block_tokens) => {
                u64::from(self.refcount[*s.blocks.last().expect("non-empty") as usize] > 1)
            }
            _ => 0,
        };
        Ok(
            self.layout.blocks_for(current + tokens) - self.layout.blocks_for(current)
                + shared_tail,
        )
    }

    /// Admit a new sequence with its prompt.
    pub fn allocate(&mut self, id: SeqId, tokens: u64) -> Result<(), MemoryError> {
        if self.seqs.contains_key(&id) || tokens == 0 {
            return Err(MemoryError::InvalidRequest);
        }
        let blocks = self.take(self.layout.blocks_for(tokens) as usize)?;
        self.seqs.insert(id, Sequence { blocks, tokens });
        Ok(())
    }

    /// Grow a sequence by `tokens`; returns a copy-on-write instruction if its tail was shared.
    pub fn append(&mut self, id: SeqId, tokens: u64) -> Result<Option<CopyOnWrite>, MemoryError> {
        let needed = self.blocks_needed(Some(id), tokens)? as usize;
        if self.free.len() < needed {
            return Err(MemoryError::NoBlocks {
                needed,
                free: self.free.len(),
            });
        }
        let bt = self.layout.block_tokens;
        let seq = self
            .seqs
            .get(&id)
            .expect("checked by blocks_needed")
            .clone();
        let mut blocks = seq.blocks;
        let mut cow = None;
        if !seq.tokens.is_multiple_of(bt) {
            let tail = *blocks.last().expect("partial tail exists");
            if self.refcount[tail as usize] > 1 {
                let dst = self.take(1)?[0];
                self.unref(tail);
                *blocks.last_mut().expect("tail") = dst;
                cow = Some(CopyOnWrite { src: tail, dst });
            }
        }
        let grow = (self.layout.blocks_for(seq.tokens + tokens)
            - self.layout.blocks_for(seq.tokens)) as usize;
        blocks.extend(self.take(grow)?);
        self.seqs.insert(
            id,
            Sequence {
                blocks,
                tokens: seq.tokens + tokens,
            },
        );
        Ok(cow)
    }

    /// Share every block of `parent` with a new sequence `child`.
    pub fn fork(&mut self, parent: SeqId, child: SeqId) -> Result<(), MemoryError> {
        if self.seqs.contains_key(&child) {
            return Err(MemoryError::InvalidRequest);
        }
        let seq = self.seqs.get(&parent).ok_or(MemoryError::NotFound)?.clone();
        for &b in &seq.blocks {
            self.refcount[b as usize] += 1;
        }
        self.seqs.insert(child, seq);
        Ok(())
    }

    pub fn free(&mut self, id: SeqId) -> Result<(), MemoryError> {
        let seq = self.seqs.remove(&id).ok_or(MemoryError::NotFound)?;
        for b in seq.blocks {
            self.unref(b);
        }
        Ok(())
    }

    pub fn block_table(&self, id: SeqId) -> Result<&[u32], MemoryError> {
        Ok(&self.seqs.get(&id).ok_or(MemoryError::NotFound)?.blocks)
    }
    pub fn tokens(&self, id: SeqId) -> Result<u64, MemoryError> {
        Ok(self.seqs.get(&id).ok_or(MemoryError::NotFound)?.tokens)
    }
    pub fn sequences(&self) -> impl Iterator<Item = SeqId> + '_ {
        self.seqs.keys().copied()
    }

    /// Invariant check for tests and receipts: refcounts equal table references,
    /// free blocks have refcount zero.
    pub fn check_invariants(&self) -> Result<(), MemoryError> {
        let mut counted = vec![0u32; self.refcount.len()];
        for seq in self.seqs.values() {
            if self.layout.blocks_for(seq.tokens) as usize != seq.blocks.len() {
                return Err(MemoryError::Busy);
            }
            for &b in &seq.blocks {
                counted[b as usize] += 1;
            }
        }
        for (b, (&rc, &c)) in self.refcount.iter().zip(&counted).enumerate() {
            if rc != c || (rc == 0) != self.free.contains(&(b as u32)) {
                return Err(MemoryError::Busy);
            }
        }
        Ok(())
    }
}
