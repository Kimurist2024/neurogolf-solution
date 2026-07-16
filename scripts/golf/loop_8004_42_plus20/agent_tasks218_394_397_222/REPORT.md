# task218 / task394 / task397 SOUND exact-regolf

## Outcome

`winner: null`

No admissible strictly-lower model was found.  Projected gain is `+0.0`.
Nothing was copied to `others/71407`, and this lane did not edit the root
submission or score ledger.

The comparison authority is `submission_base_8009.46.zip`.  Its three members
were copied byte-for-byte into `base/` and have these profiles:

| task | SHA-256 | memory | params | cost | disposition |
|---:|---|---:|---:|---:|---|
| 218 | `6740bffa1e0434430c998a6a7b1b05251258071f2b741b362d00c53d86934113` | 260 | 69 | 329 | reject: not fresh-exact; internal shape cloak |
| 394 | `cb47909c49db2fab103bdbaa0be19c49d2eacc2336393086260c7754bd0ffb89` | 285 | 65 | 350 | reject: runtime-shape cloak |
| 397 | `2361956a9b6d1391aff9d8bc4af26d5112877e0e2946b365d04e461899f4d7e1` | 249 | 89 | 338 | reject: lookup/shape cloak/default-session failure |

## Generator rules decoded

- **task218 / `90c28cc7`**: the 21x21 input contains a rectangular quilt with
  2 or 3 row blocks and 2 or 3 column blocks.  Each block is constant.  Return
  the compact 2x2, 2x3, 3x2, or 3x3 table of block colors.  Row depths and
  column widths are independently 3..12, their sums are below 21, and the
  quilt offset is arbitrary.
- **task394 / `f9012d9b`**: a size 4..7 periodic two-color grid has one black
  square bite.  The period is 2 for sizes 4..6 and 3 for size 7.  Recover the
  missing 1x1, 2x2, or 3x3 patch from the equal-residue row one period away and
  place it at the output origin.
- **task397 / `fcc82909`**: find each disjoint 2x2 colored box.  Its shadow
  height is the number of distinct colors in that box (2..4).  Paint a
  two-column green shadow immediately below it while preserving the input.

These are Type D/A/A-B respectively.  The transforms are input-dependent; no
fixed example lookup is valid.

## Authority-member audit

The complete machine evidence is `evidence/audit.json`.  Tests used all 266
known cases plus two independent generator seeds (`21839401`, `21839402`) with
1500 fresh cases per seed.  Every available run used all four configurations:
ORT_DISABLE_ALL/default x threads 1/4.  Raw tensors were compared bitwise to
ORT_DISABLE_ALL/threads1, and runtime errors/non-finite values were counted.

### task218

- Known: 266/266 in every configuration; raw-identical across all four.
- Fresh: **1497/1500 on each seed**, identically in all four configurations.
  Therefore it is not a generator-exact parent.  The first seed has three
  concrete counterexamples at indices 577, 943, and 1016; they are preserved
  in `evidence/task218_counterexamples.json`.
- Runtime-shape trace exposes 23 tensors and finds two contradictions:
  `gn` and `qi` are declared `[1,1,1,1]` but execute as
  `[1,10,30,30]`.  This is an internal cost cloak.
- Example failure 577 drops the entire third output column from the expected
  `[[8,2,9],[7,6,8],[1,5,3]]`; this is a real support failure, not noise.

### task394

- Known: 266/266 in every configuration.
- Fresh: 1500/1500 for both seeds in every configuration.  All raw tensors are
  bitwise identical across configurations; errors/non-finite values are zero.
- It is still inadmissible under this lane's truthful-shape rule.  The trace
  finds four contradictions: `hid` and `input16` are declared one scalar
  channel but execute as full `[1,10,30,30]`; `O` executes with ten channels;
  and the declared `[1,1,30,30]` output is actually `[1,10,30,30]`.

### task397

- ORT_DISABLE_ALL threads 1/4: known 266/266 and fresh 1500/1500 for both
  seeds, raw-identical with no runtime error/non-finite value.
- Default ORT threads 1/4 cannot create a session: `Concat` has contradictory
  inferred/declared dimensions (`1` versus `3`).
- Strict shape inference with data propagation fails at a `Gather`; the graph
  declares output `[1,1,1,1]` but ORT_DISABLE_ALL produces
  `[1,10,30,30]`.  It also contains ten `CenterCropPad` nodes and one
  `TfIdfVectorizer`.  This violates the no-lookup, truthful-shape, and
  four-configuration gates.

## Sound exact upper bounds

To distinguish “no cheap winner” from “rule not understood”, the existing
truthful `artifacts/optimized` implementations were independently audited.
Each one passes checker/full strict+data-prop, has truthful runtime shapes,
standard domains, no banned/lookup/giant node, and zero Conv-family bias UB.
For each task it is 266/266 known plus 1500/1500 on both fresh seeds in every
configuration (13,064 raw executions per task), with bitwise cross-config
equality and zero errors/non-finite values.

| task | sound exact SHA-256 | memory | params | cost | authority cost |
|---:|---|---:|---:|---:|---:|
| 218 | `efdaa7c8154ea6b7622da22d7e48f880c94899139d1ba327e1f847f6e36eeb03` | 16438 | 46 | 16484 | 329 |
| 394 | `fcc9b7620204c217e3921ea437676f8dd50c145f580ac707313464e41d3156c2` | 4462 | 36 | 4498 | 350 |
| 397 | `f775f0c6ed2d5be77fd3f02340b3b4da67c60c0d7ce7edd47595ae1d4f89561f` | 20146 | 87 | 20233 | 338 |

They prove the decoded rules but are all far above the authority costs, so
none can be a score improvement.

## Exact-cleanup search

Eleven fixed-point profiles were applied to each authority member and each
truthful exact reference: dead-end elimination, CSE, duplicate/unused
initializer elimination, no-op removal, consecutive-idempotent removal,
Conv/Pad fusions, shape folds, Einsum-to-MatMul, Where rewrite, and Add
adjustment.  This is 66 profiles total.

- strictly lower authority descendants: 0
- strictly lower truthful-reference descendants: 0
- dead outputs / initializer aliases / no-op/factor-absorption winner: none

The `candidates/` directory is intentionally empty.  A behavior-changing
fresh-90 candidate was not considered admissible, and the task218 counterexample
prevents treating its authority member as an exact equivalence oracle.

## Evidence index

- `audit_exact_lane.py` — reproducible structural, optimizer, known, fresh,
  four-configuration, raw-equality, error, and non-finite audit
- `diagnose_task218.py` — deterministic fresh counterexample extractor
- `evidence/audit.json` — full machine-readable results
- `evidence/task218_counterexamples.json` — three concrete generator-legal
  failures from seed 21839401

