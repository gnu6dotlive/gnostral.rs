# Vendoring policy

Vendoring is allowed because reproducible runtime research sometimes requires an exact upstream snapshot. It must never erase provenance.

## Allowed states

PINNED_UPSTREAM — exact upstream material at a recorded revision with no local modification.

PATCHED_UPSTREAM — pinned upstream material plus explicitly identified local patches.

REFERENCE_ONLY — no vendored source. The lab records upstream identity and analyzes it without copying the source tree.

## Requirements

Every vendored project must record canonical repository, exact revision, upstream license, local state, modification status, patch list, and reason for vendoring.

Preserve upstream license and copyright files verbatim.

## Default

Prefer REFERENCE_ONLY plus patches and manifests. Vendor a full source tree only when byte-identical reproducibility materially benefits the experiment.

## Forked upstreams (PATCHED_UPSTREAM)

A lab fork is recorded as `PATCHED_UPSTREAM` with three extra facts so a reviewer can
reconstruct it without the local tree: the public `base_revision` the fork starts from,
the `local_commits` on top, and a patch file exporting the full base→working-tree delta.
The inventory hashes the applied tree and records the patch digest. An uncommitted
working tree is exported as a patch, never left as the only copy of the change.
