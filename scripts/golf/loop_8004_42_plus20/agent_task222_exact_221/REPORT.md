# task222 exact/regolf lane 221

## Result

**No adoptable strict-lower candidate exists in this lane.** The result is
`winner: null`, projected gain `+0.0`. Nothing under `submission.zip`,
`all_scores.csv`, or `others/71407` was changed.

The immutable authority is `submission_base_8009.46.zip::task222.onnx`, SHA-256
`49cc513b28ccc5edca65c2cf886b79337a098400785a928599719d4bcfade7d7`,
official cost **380 = memory 0 + params 380**.

## True rule and why a complete-support replacement cannot be proved

The task index maps to generator hash `91714a58` and source
`inputs/arc-gen-repo/tasks/task_91714a58.py`. The decoded rule is:

1. Work on a fixed 16x16 grid.
2. A latent filled monochrome rectangle is planted with width/height in 2..8,
   area in 9..16, and a one-cell outer border.
3. Emit that planted rectangle unchanged and zero every other output cell.

The decoded rule was checked structurally against all 266 stored pairs
(train 3, test 1, arc-gen 262): all 266 contain exactly one output rectangle
with the required dimensions/area/interior placement and copy the same input
cells. Evidence: `evidence/true_rule_known_certificate.json`.

More importantly, the full generator parameter support is **not identifiable
from the input**. `evidence/generator_support_ambiguity.json` constructs one
16x16 input containing two disjoint 3x3 rectangles:

- color 2 at row/col `(2,2)`;
- color 3 at row/col `(10,10)`.

Calling the generator with the first rectangle as latent target and with the
second rectangle as latent target produces the **same input** but outputs that
differ at 18 cells. Both also lie in the no-argument generator support with
positive probability: `random_pixels` can select exactly the other nine cells,
all nine random colors can equal its color, the planted color has no
pre-overlay adjacent pair, and all four boundary checks pass.

Therefore no deterministic ONNX graph can be correct on every element of the
full generator support. For this private-risk task, the only zero-added-risk
admission route is an **all-input pass-through rewrite of the admitted
authority**. None was found.

## Authority algebra

The authority is a single 21-input `Einsum` with dense parameters:

| factor | shape | params | nonzero | exact rank |
|---|---:|---:|---:|---:|
| `V` | 30x8 | 240 | 128 | 8 |
| `U` | 2x8 | 16 | 16 | 2 |
| `A` | 2x2 | 4 | 4 | 2 |
| `S` | 10x2 | 20 | 19 | 2 |
| `P` | 10x10 | 100 | 19 | 10 |

Ranks were computed by Gaussian elimination over the exact binary rationals
represented by the serialized float32 values. `V[:16]` is already exact rank
8, every `U` entry is nonzero, and `P` is full rank 10. Evidence:
`evidence/algebra_certificate.json`.

### `P` removal

Immediately before `P`, let the ten channel values be `s_k`. The authority
computes:

```text
output_0 = 10 * (s_0 - sum(s_1..s_9))
output_o = 10 * s_o, o > 0
```

Removing `P` and changing the output to `->bkrc` saves 100 params, yielding
cost 280, but is not threshold-equivalent. On all 266 known cases the exact
formula above reproduces the authority raw tensor, while the required
background condition
`(s0 - sum_nonbackground > 0) iff (s0 > 0)` fails in every case, over 31,971
background cells in total. The first counterexample is train[0], row 0, col 4:
`s0=0`, `sum_nonbackground=-0.561805...`; the authority background is positive,
the no-`P` background is not. The candidate is **0/266 in every one of four
ORT configurations**.

### Rank deletion

Deleting each of the eight shared `V/U` components gives cost 348, but changes
the real polynomial. Full/strict/data-prop pass, yet known correctness in each
of all four ORT configurations is respectively:

```text
component: 0   1    2   3    4    5   6   7
right/266: 0  128  28  120  112  96  76  117
```

Every candidate is raw-different from the authority and has no runtime or
nonfinite errors. These reproduce and extend the earlier retained-history
rank-drop failures to all eight components and all four configurations.

### `V` zero tail

Rows 16..29 of `V` are all zero (112 zero elements), but the `D/E/F/H` indices
are tied directly to the static 30-element input axes. A dense `[16,8]` tensor
is therefore not a legal replacement in the same `Einsum`. Standard dense
reconstruction by padding a `[16,8]` initializer would cost 128 params plus a
counted float32 `[30,8]` activation of 960 bytes: **1088 before the other
factors**, versus the current 240-param `V`. Cropping input is still larger.

An exact sparse `V/S/P` probe would count only 186 parameters, but it is not a
valid scorer model: ONNX full checking and strict data-prop expose each sparse
`Einsum` operand as rank 0, so the equation rejects it before runtime. The
official sanitizer also does not rename sparse initializer names. It is
quarantined as an invalid probe, not a candidate.

## Verification summary

The authority is 266/266 on the complete known set in all four configurations
(`ORT_DISABLE_ALL`/default x threads 1/4), with zero errors, zero nonfinite
values, and no positive value in `(0, 0.25)`.

An independent fresh seed of 1,000 generator cases gives exactly **943/1000**
in all four configurations (57 wrong, zero runtime errors). Raw tensors are
identical across the four configurations. This confirms that known/fresh
admission alone cannot turn the current analog matcher into a SOUND true-rule
net, and fresh testing was not used to excuse any rewrite.

Evidence:

- `evidence/candidate_audit.json`
- `evidence/authority_fresh_four_config.json`
- `evidence/algebra_certificate.json`
- `evidence/generator_support_ambiguity.json`
- `evidence/true_rule_known_certificate.json`

Builders and audit scripts are isolated in this directory. `try_candidate.py`
was never called. No lookup, custom-domain op, undefined behavior, malformed
dense tensor, or new giant initializer was introduced.

## Root guards

Final read-only guards:

```text
submission.zip              4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927
submission_base_8009.46.zip 4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927
all_scores.csv              8c99379ce1c77e6894d53b593ccb5ae10f01c3c70e5666f832b1b2454c709b78
```

