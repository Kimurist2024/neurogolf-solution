# task158 complete-support residual regolf

## Outcome

One SOUND, strictly cheaper task158 replacement is accepted for independent
review against the staged cost-7525 authority.  Official-like actual cost falls
from **7525 to 7498** (memory **6662 -> 6638**, parameters **863 -> 860**), for
an incremental projected score gain of **+0.003594492321219123**.

- candidate: `candidates/task158_exact_anchor_role_bits.onnx`
- candidate SHA-256: `e7101699bfc022fa794e15d7f374a8febe3e2680b8388c67b9a81cdc9962ced0`
- authority: `others/71407/task158.onnx`
- authority SHA-256: `127984c6807d84559bbf74fd58e3b09a66459d142cef65a8635647e64f5e59fd`
- node count: 165 -> 163
- no root ZIP, ledger, staged model, or prior lane was changed

## Complete-support proof

The dynamic anchor convolution gives weight code 1 to the numerically lower
endpoint colour, code 2 to the higher endpoint colour, and zero to every other
channel.  Therefore background and the variable sprite-fill pattern cannot
affect the anchor score.

`prove_anchor_support.py` exhaustively enumerates all 48 local generator
geometries:

- magnification 1, 2, or 3;
- either opposite diagonal after flips;
- either endpoint carrying low code 1;
- both row and column translations modulo Conv stride 2.

Different generator boxes use `common.overlaps(..., spacing=2)`.  Their nearest
cells are separated by at least three coordinates on one axis, so a 3-cell
Conv window cannot see endpoints from two boxes.  Boundaries only delete local
windows and cannot create a new score.  Each endpoint block intersects at most
3x3 sampled windows; two endpoints across at most four objects affect at most
72 of the 169 Conv windows.  Thus at least 97 windows are exact zero, and TopK
cannot admit a negative score.

The resulting complete TopK support is exactly:

`{0, 2, 4, 8, 10, 16, 20, 24, 26, 48, 52, 72, 106, 144, 212}`.

On that full support, the incumbent threshold chain classifies low anchors as
`{2,8,10,24,26,72,106}` and high anchors as
`{4,16,20,48,52,144,212}`.  Every low value has a nonzero intersection with
uint8 mask `0b1010`; every high value and zero has intersection zero.  The
candidate therefore computes:

1. `low_mask = (uint8(top_values) & 0b1010) > 0`;
2. `anchor_high = anchor_valid XOR low_mask`.

The same exhaustive table also proves `top_values >= 6` iff
`uint8(top_values) > 4`.  The candidate reuses existing scalar
`lutnp_shift4=4` and removes `phase_cut_0`.

This removes three 8-byte role-selection tensors and one 16-byte float16 role
threshold tensor while replacing them with two 8-byte uint8 tensors.  It also
removes the three dedicated role-threshold parameters, adds one uint8 bitmask,
and removes the dedicated phase-0 cutoff: memory -24, parameters -3, cost -27.

## Verification

The candidate passes:

- full ONNX checker;
- strict shape inference with data propagation;
- shared structure gate;
- standard domains only, banned ops zero, functions/nested graphs zero;
- sparse/external initializers zero, lookup red flags zero;
- Conv-family bias UB findings zero;
- static/nonstatic violations zero;
- runtime declared/actual intermediate shape mismatches zero;
- recomputed static floor 7498 and runtime intermediate memory 6638.

Known-complete testing covers all 266 cases in each of four configurations:
ORT disable-all/default x threads 1/4.  Candidate and authority are raw-bitwise
equal and truth-correct for all **1,064** comparisons.

Fresh testing uses seeds 1582151 and 1582152, 3,000 generated cases per seed,
in the same four configurations.  All **24,000** candidate/authority raw
comparisons are bitwise equal and truth-correct.  Runtime errors, output
nonfinite values, and raw mismatches are all zero.

## Rejected residuals

- Replacing the 6x3x3 permutation matrices by 6x3 integer permutation indices
  cannot remove the matrix: it is also required to evaluate all six assignment
  costs.  The simple rewrite fails dependency validation and was discarded.
- Sampling observed only permutation rows 0, 1, and 2, but no complete generator
  proof excluded rows 3--5.  The candidate retains the full six-permutation
  table; no sample-only support pruning is admitted.
- The two unused TopK Values outputs remain schema-required Single outputs.

Evidence:

- `evidence/build.json`
- `evidence/audit.json`
- `evidence/anchor_support_proof.json`
- `evidence/four_config_raw_equivalence.json`
- `evidence/fresh_dual_2x3000.json` (earlier two-mode confirmation; superseded
  by the four-configuration matrix)

