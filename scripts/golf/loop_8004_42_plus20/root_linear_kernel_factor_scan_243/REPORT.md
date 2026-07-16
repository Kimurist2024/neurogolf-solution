# Exact linear kernel/factor scan 243

## Result

No globally strict-lower candidate exists in the requested family.

- Authority: `submission_base_8009.46.zip`
- Authority SHA-256:
  `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
- Models scanned: 400
- Target nodes: 167 in 37 tasks
  - Conv: 156
  - MatMul: 4
  - Gemm: 7
- Unique constant weight initializers: 23
- Constant weight occurrences: 143; constant bias occurrences: 10
- Dynamic/generated weight occurrences retained in the census: 24
- Exact global factor choices costed: 34
- Globally strict-lower factor choices: **0**
- Direct one-hot/diagonal/permutation strict-lower choices: **0**
- Consecutive linear edges, before even applying the single-consumer gate: **0**
- Candidate models: none
- Projected score gain: 0
- Decision: `NO_CANDIDATE`

The authority ZIP, root submission, staging area, score ledger, and
`others/71407` were not modified. Kimi and `try_candidate.py` were not used.

## Exhaustive boundary and exactness

Every Conv/MatMul/Gemm node in the archive was inventoried. For every constant
weight, the scan tested the following families:

1. every legal sequential-Conv split of the spatial axes, including the two
   channel-only endpoints, by exact rational matrix rank;
2. explicit depthwise-then-pointwise and pointwise-then-depthwise forms, with
   a separate exact rank test for every input/output channel;
3. shared spatial-bank / outer-product forms for existing grouped and
   depthwise kernels;
4. stricter contiguous block/group structure;
5. MatMul/Gemm exact rank factors;
6. one-hot selection, diagonal, identity, and permutation matrices;
7. direct consecutive Conv/MatMul/Gemm composition.

No SVD, floating tolerance, approximate rank, or fitted factor is used.
Serialized integers/floats are converted to exact rational values. Proposed
stored factors must themselves represent every rational coefficient exactly in
the source dtype. An independent post-scan reconstruction rechecked all 46
stored-dtype Conv factorizations coefficient-by-coefficient over exact
rationals.

The Conv census contains 60 axis partitions, 46 stored-dtype exact
factorizations, and 11 parameter-reducing partitions. Sixteen kernels were
also checked explicitly in each depthwise ordering: 14 satisfy an exact
depthwise-then-pointwise form and 15 an exact pointwise-then-depthwise form;
only five pointwise-first forms reduce parameters. Three authority kernels are
already depthwise. No stricter block/group plan exists.

All seven constant MatMul/Gemm matrices were exact-ranked. None has a
parameter-reducing low-rank factorization.

## Global reuse and official cost accounting

The baseline of every target task was profiled through the official-compatible
ORT-disable-all scorer. All 37 profiles match `all_scores.csv` exactly.

A weight is credited as removed only if one action covers every occurrence of
that initializer. New factor tensors are deduplicated globally by dtype,
shape, and raw bytes against all surviving initializers and against one
another. The cost of each global choice is then:

`official baseline + new intermediate bytes - removed params + unique new params`.

All 34 choices have concrete intermediate shapes; none has an unresolved
memory term. The nearest choice of any kind is still **+5 cost** (task275's
trivial non-reducing split). The nearest genuinely parameter-reducing choice
is task297 at **+112 cost**.

The complete parameter-saving cases are:

| task / weight | exact form | params | added activation | official delta / barrier |
|---|---|---:|---:|---:|
| 018 `q_w` | rank-1 pointwise then full spatial | 160 -> 26 | 3,600 | +3,466 |
| 018 `q_w` | rank-1 height then width | 160 -> 28 | 2,760 | +2,628 |
| 018 `q_w` | rank-1 width then height | 160 -> 82 | 480 | +402 |
| 178 `WP` | rank-1 pointwise then depthwise | 40 -> 14 | 46,800 across 13 uses | +46,774 |
| 207 `Wagg` | three rank-1 splits | 40 -> 14/22 | n/a | initializer also feeds Einsum; cannot remove |
| 297 `conv_w` | rank-1 pointwise then depthwise | 20 -> 12 | 120 | +112 |
| 345 `Wpack` | rank-1 pointwise then width | 100 -> 20 | 720 across 6 uses | +640 |
| 012 `w` | rank-2 shared bank across existing depthwise groups | 700 -> 160 | at least 72,000 | at least +71,460 |
| 193 `W` | rank-2 shared bank across existing depthwise groups | 160 -> 52 | at least 72,000 | at least +71,892 |

For the two shared-bank rows, the lower bound deliberately omits the added
Reshape outputs. Even if every factor parameter were free through reuse, their
activation-only deltas remain positive (+71,300 and +71,840 respectively).

## One-hot, diagonal, permutation, and composition disposition

There are two direct matrix-structure hits:

- task177 `psph` is a scalar diagonal/scaled-permutation weight, but it has 15
  total uses including Einsum operands;
- task340 `oneMM` is a scalar identity/one-hot/permutation weight, but it has
  six total uses including four Einsum operands.

Rewriting only the Gemm occurrences cannot delete either initializer, so
neither can lower official cost. There is no eligible 1x1 Conv one-hot,
diagonal, or permutation kernel, and no stricter block-sparse Conv group.

The graph-wide producer/consumer pass found zero direct edges from a
Conv/MatMul/Gemm output to another Conv/MatMul/Gemm. Therefore the exhaustive
consecutive-linear composition family is empty in this authority; there is no
rounding-order candidate to audit.

## Runtime and safety audit disposition

The requested known-266 plus two fresh seeds x 5,000 x four ORT configurations
raw-bit/error/nonfinite/shape/UB audit is conditional on a strict-lower
structural survivor. The survivor count is zero, so no runtime corpus was
launched and no claim of runtime equivalence is made for a non-lower plan.

Had a choice survived cost, its changed multiplication/accumulation order
would have required raw output bytes to match in all configurations; exact
factor identity alone would not have been accepted as sufficient.

## Artifacts

- `scan.py`: reproducible full-authority exact scanner and global cost model
- `scan.json`: node/initializer census, all algebraic family rows, official
  profiles, all 34 global choices, and the two blocked direct simplifications
- `candidates/`: intentionally empty

Final invariant checks passed: 400 models, 167 target nodes, 37/37 official
profiles equal the ledger, 46/46 independent exact reconstructions, zero
unresolved activation shapes, zero strict-lower choices, and zero candidate
files.
