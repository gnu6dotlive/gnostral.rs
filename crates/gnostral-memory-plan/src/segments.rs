//! Segmented allocator: fixed-size segments, best-fit extents, elastic return.
//!
//! The engine maps one segment to one device allocation. Freed extents are
//! coalesced; a segment with nothing live can be returned with
//! [`SegmentedAllocator::release_empty`]. Placement is deterministic: the best fit
//! is the smallest free extent that fits, ties broken by segment index then offset.

use crate::MemoryError;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Allocation {
    handle: u64,
    pub segment: usize,
    pub offset: u64,
    pub bytes: u64,
}

#[derive(Debug, Clone)]
struct Segment {
    free: BTreeMap<u64, u64>,
    used: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stats {
    pub segments: usize,
    pub reserved_bytes: u64,
    pub used_bytes: u64,
    pub free_bytes: u64,
    pub largest_free: u64,
    /// `1 - largest_free / free_bytes`, in parts per million (0 when nothing is free).
    pub fragmentation_ppm: u64,
}

#[derive(Debug)]
pub struct SegmentedAllocator {
    segment_bytes: u64,
    alignment: u64,
    max_segments: usize,
    segments: Vec<Option<Segment>>,
    live: BTreeMap<u64, Allocation>,
    next_handle: u64,
}

impl SegmentedAllocator {
    pub fn new(
        segment_bytes: u64,
        alignment: u64,
        max_segments: usize,
    ) -> Result<Self, MemoryError> {
        if segment_bytes == 0
            || alignment == 0
            || !alignment.is_power_of_two()
            || !segment_bytes.is_multiple_of(alignment)
        {
            return Err(MemoryError::InvalidRequest);
        }
        Ok(Self {
            segment_bytes,
            alignment,
            max_segments,
            segments: Vec::new(),
            live: BTreeMap::new(),
            next_handle: 0,
        })
    }

    fn open_segments(&self) -> usize {
        self.segments.iter().filter(|s| s.is_some()).count()
    }

    pub fn alloc(&mut self, bytes: u64) -> Result<Allocation, MemoryError> {
        if bytes == 0 {
            return Err(MemoryError::InvalidRequest);
        }
        let size = bytes
            .checked_add(self.alignment - 1)
            .ok_or(MemoryError::Overflow)?
            & !(self.alignment - 1);
        if size > self.segment_bytes {
            return Err(MemoryError::TooLarge {
                bytes,
                segment: self.segment_bytes,
            });
        }
        let mut best: Option<(u64, usize, u64)> = None;
        for (index, segment) in self.segments.iter().enumerate() {
            let Some(segment) = segment else { continue };
            for (&offset, &len) in &segment.free {
                if len >= size && best.is_none_or(|b| (len, index, offset) < b) {
                    best = Some((len, index, offset));
                }
            }
        }
        let (index, offset) = match best {
            Some((_, index, offset)) => (index, offset),
            None => {
                if self.open_segments() >= self.max_segments {
                    return Err(MemoryError::OutOfSegments);
                }
                let fresh = Segment {
                    free: BTreeMap::from([(0, self.segment_bytes)]),
                    used: 0,
                };
                let index = match self.segments.iter().position(Option::is_none) {
                    Some(i) => {
                        self.segments[i] = Some(fresh);
                        i
                    }
                    None => {
                        self.segments.push(Some(fresh));
                        self.segments.len() - 1
                    }
                };
                (index, 0)
            }
        };
        let segment = self.segments[index]
            .as_mut()
            .expect("chosen segment is open");
        let len = segment.free.remove(&offset).expect("chosen extent exists");
        if len > size {
            segment.free.insert(offset + size, len - size);
        }
        segment.used += size;
        let allocation = Allocation {
            handle: self.next_handle,
            segment: index,
            offset,
            bytes: size,
        };
        self.next_handle += 1;
        self.live.insert(allocation.handle, allocation);
        Ok(allocation)
    }

    pub fn free(&mut self, allocation: Allocation) -> Result<(), MemoryError> {
        if self.live.get(&allocation.handle) != Some(&allocation) {
            return Err(MemoryError::NotFound);
        }
        self.live.remove(&allocation.handle);
        let segment = self.segments[allocation.segment]
            .as_mut()
            .ok_or(MemoryError::NotFound)?;
        segment.used -= allocation.bytes;
        let mut offset = allocation.offset;
        let mut len = allocation.bytes;
        if let Some((&prev, &prev_len)) = segment.free.range(..offset).next_back() {
            if prev + prev_len == offset {
                segment.free.remove(&prev);
                offset = prev;
                len += prev_len;
            }
        }
        if let Some(next_len) = segment.free.remove(&(offset + len)) {
            len += next_len;
        }
        segment.free.insert(offset, len);
        Ok(())
    }

    /// Close every segment with nothing live; returns the closed indices.
    pub fn release_empty(&mut self) -> Vec<usize> {
        let mut released = Vec::new();
        for (index, slot) in self.segments.iter_mut().enumerate() {
            if matches!(slot, Some(s) if s.used == 0) {
                *slot = None;
                released.push(index);
            }
        }
        released
    }

    /// Lower the segment cap; refused (`Busy`) if more segments are open than allowed.
    pub fn set_max_segments(&mut self, max_segments: usize) -> Result<(), MemoryError> {
        if self.open_segments() > max_segments {
            return Err(MemoryError::Busy);
        }
        self.max_segments = max_segments;
        Ok(())
    }

    pub fn stats(&self) -> Stats {
        let mut s = Stats {
            segments: 0,
            reserved_bytes: 0,
            used_bytes: 0,
            free_bytes: 0,
            largest_free: 0,
            fragmentation_ppm: 0,
        };
        for segment in self.segments.iter().flatten() {
            s.segments += 1;
            s.reserved_bytes += self.segment_bytes;
            s.used_bytes += segment.used;
            for &len in segment.free.values() {
                s.free_bytes += len;
                s.largest_free = s.largest_free.max(len);
            }
        }
        if s.free_bytes > 0 {
            s.fragmentation_ppm =
                1_000_000 - (s.largest_free as u128 * 1_000_000 / s.free_bytes as u128) as u64;
        }
        s
    }
}
