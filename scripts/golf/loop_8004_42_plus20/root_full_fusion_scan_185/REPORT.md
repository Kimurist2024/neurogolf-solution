# Full-archive exact fusion scan 185

The immutable 8009.46 archive was scanned with 21 exact ONNX optimizer pass
sets (8,400 task/pass profiles).  The pass families include MatMul/Add/Gemm,
Conv/Add, BN/Conv, Pad/Conv/Pool, transpose/Gemm, CSE, shape/slice cleanup,
Einsum/MatMul, and the conservative fixed-point fusion set.

Eight strict-lower profiles were emitted across seven unique tasks.  None is
admissible:

- task039 42->41, task111 89->88, task122 101->100, and task183 160->89
  remove dead producers that serve as allocator/shape witnesses.  All four
  are the byte-equivalent lineages already shown to fail the complete known
  set at runtime in `../root_exact_noop26/REPORT.md`.
- task089 1340->1171 is the previously audited allocator/shape failure and is
  known-wrong after cleanup (`../agent_high037_089_279_124/REPORT.md`).
- task165 587->547 (CSE) and 587->546 (combined) merge two identical-looking
  `CastLike` producers.  The rewrite changes allocator/liveness behavior,
  fails all known cases, and has 88 declared/runtime shape mismatches as
  independently documented in `../agent_8008_exact_white102/REPORT.md`.
- task264 344->343 deduplicates the `axes1`/`s1` initializer.  It retains the
  authority's default-ORT runtime failure and 44 truthful-shape mismatches,
  as recorded in the current-loop 8009 exact-B audit.

The additional 11 pass families generated no new strict-lower profile.
Safe adoptees: **0**.  Projected gain: **+0.0**.  Root submission and score
ledgers were not modified.  Complete machine evidence is in `scan.json`.
