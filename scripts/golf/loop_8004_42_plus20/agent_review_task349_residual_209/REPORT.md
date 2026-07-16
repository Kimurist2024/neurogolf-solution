# task349 residual independent review 209

## Decision

**PASS / ACCEPT_GENERATOR_SUPPORT_EXACT**

- authority: `others/71407/task349.onnx`
  - SHA-256: `f7531b66a5399973ed57835584023c5bf1f61966c218b283cb721ba7ca45c8e2`
  - official profile: `memory 3233 + params 315 = cost 3548`
- candidate: `scripts/golf/loop_8004_42_plus20/agent_task349_residual_205/candidates/task349_residual_patch_final.onnx`
  - SHA-256: `8ab46bc1217c80c1d15c6064ea12a502c15274e12f79d9546f3d4620b76b72a3`
  - official profile: `memory 3239 + params 293 = cost 3532`
- strict reduction: **-16 cost**
- projected gain: **+0.004519781705619557**

The candidate is safe to stage as an exact pass-through of the currently staged
authority on the complete default generator support. It does not claim to repair
the authority's pre-existing fresh misses; it introduces no new miss.

## Independent proof of the five transformations

### 1. Valid-column affine reuse

The generator AST fixes `factor = common.randint(2, 6)` with inclusive Python
`random.randint`, then `size = 5 * factor`. The complete side support is therefore
`{10,15,20,25,30}`. `convert_to_numpy` one-hot encodes every in-grid cell, so the
area reduction is one of `{100,225,400,625,900}` and its square root is the exact side.

For every supported side, the authority lookup and candidate expression agree:

| side | authority table | `affine_width_factor[-side]` | `BitwiseNot` result |
|---:|---:|---:|---:|
| 10 | 1023 | -1024 | 1023 |
| 15 | 32767 | -32768 | 32767 |
| 20 | 1048575 | -1048576 | 1048575 |
| 25 | 33554431 | -33554432 | 33554431 |
| 30 | 1073741823 | -1073741824 | 1073741823 |

The candidate graph is exactly `Neg(side_i8) -> Cast(int32) -> Gather(affine_width_factor)
-> BitwiseNot`. The negative indices remain in `[-30,-10]`, within the length-30 tensor.

### 2. Shift lookup to uint8 BitShift

`radius_code` is reduced modulo 11, so all possible codes are exactly 0 through 10.
All eleven authority table rows were enumerated. The corresponding radii are
`[5,0,4,1,2,0,0,0,3,0,0]`; `uint8(1) << radius` produces
`[32,1,16,2,4,1,1,1,8,1,1]`, byte-for-byte the old int32 table after Cast.
The largest intermediate is 32, so uint8 overflow is impossible. The ONNX node's
direction attribute is independently verified as `LEFT`.

### 3. Side and coordinate narrowing

For every possible area, float32 Sqrt returns exactly 10, 15, 20, 25, or 30.
Consequently `Cast(float32 -> int8)` equals the removed
`Cast(float32 -> int32 -> int8)`. Negated side remains in `[-30,-10]`.

`coords4` contains exactly 0 through 29. Converting it from int32 to int8 preserves
all values, and exhaustive `5 sides x 30 coordinates` comparison proves that
`coords4 < side` is unchanged.

### 4. Beam rank construction

All supported sides are positive. Therefore
`Unsqueeze(Clip(side, 0, 29))` equals broadcasted
`Min(side, int8[1,1,1,1](29))`. Exhaustive outputs are
`[10,15,20,25,29]` in both graphs, with the same rank-4 shape consumed by Concat.

### 5. Special h-patch separation

The authority's six-entry update triplets reconstruct exactly from the candidate's
four main plus two special entries:

- indices: `[9,12,5,8] + [19,27]`
- signatures: `[495564,495564,133111344,133111344] + [214431744,214431744]`
- values: `[-24576,24576,-16384,16384] + [63,-63]`

The authority beam condition gathers signature index 4, whose value is 214431744.
The candidate computes that same scalar equality once and uses it both for the two
special halo updates and the existing beam update. The Concat order of indices and
updates is unchanged, so this transformation is all-input algebraically exact, not
sample-dependent.

## Protobuf scope audit

The diff closes exactly over the five declared groups:

- 99 common primary-output nodes are byte-identical.
- The nine changed common outputs are exactly `side_i8`, `valid_cols`,
  `shift_factor`, `valid_rows4`, `beam_end_scalar_i8`, `beam_indices_i8`,
  `halo_indices_i8`, `halo_updates`, and `sp_bupdate`.
- Source-only and candidate-only node outputs exactly match the required
  replacement plumbing; there are no unexplained nodes.
- 22 common initializers are byte-identical.
- The only modified common initializers are `coords4` and the three shortened
  h-patch arrays.
- The only removed initializers are `five_i32`, `shift_by_mod`, `unsq4`, and
  `valid_cols_table`; the only additions are the five declared scalar/special arrays.
- With nodes and initializers removed, every other ModelProto field is byte-identical.

## Static and runtime verification

- full ONNX checker: pass
- strict shape inference: pass
- strict shape inference with data propagation: pass
- standard domain opset 18 only
- functions / sparse initializers / nested graphs: 0 / 0 / 0
- banned ops / unused initializers / nonfinite initializers: 0 / 0 / 0
- Conv-family nodes and short-bias UB: 0
- typed runtime trace: 123 tensors, shape mismatches 0, nonfinite values 0

Known 267 cases were executed under disable-all/default optimization and threads
1/4. Both models were correct on 267/267 in every configuration, with raw-bitwise
equality 267/267 in every configuration.

Independent fresh seeds, not used by lane 205 or the earlier review:

| seed | cases | authority/candidate correct | raw equality across four configs |
|---:|---:|---:|---:|
| 20934973 | 2500 | 2486 (99.44%) | 10,000 / 10,000 |
| 20934991 | 2500 | 2482 (99.28%) | 10,000 / 10,000 |

Across known plus fresh, all **21,068 case-config comparisons** were raw-bitwise
equal. Runtime errors and nonfinite outputs were both zero. The fresh misses are
identical, pre-existing authority misses; candidate-specific regressions are zero.

Protected root `submission.zip`, `submission_base_8009.46.zip`, and
`all_scores.csv` hashes were unchanged before and after the review.

Machine-readable evidence: `audit.json`. Reproducible independent audit: `audit.py`.

