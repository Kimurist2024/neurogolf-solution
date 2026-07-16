# task192 exact relation-factor golf

## Result

Accepted SOUND candidate:
`task192_selected_masks.onnx`, SHA-256
`40244ab462644481407ebb7200984dfdff1475c0d8e6ff731ba2d588ec92ea09`.

- Immutable 8008.14 authority: cost **1609** (memory 88, parameters 1521).
- Candidate: cost **1197** (memory 248, parameters 949).
- Reduction: **412**.
- Projected gain: **+0.29579444143441**.

## Exact rewrite

The prior exact polynomial model stored a relation tensor with slices

- `relation[0,d,a] = 1`, and
- `relation[1,d,a] = (d == a)`.

The selected-color vector is one-hot. Therefore, for every input and selected
color,

`sum_a relation[r,d,a] * selected[a]`

is exactly `1` for `r=0` and `selected[d]` for `r=1`. The candidate constructs
those two rows directly as `Concat(all_colors, selected)`. This removes the
200-element relation initializer. The extra 2x10 float intermediate raises
profiled memory by 80 bytes, while parameters fall by 190, for a net cost
reduction of 110 from the prior cost-1307 exact candidate.

The decoded rule and sign proof are unchanged: with selected nonzero color A,
`P = nonzero(center) * horizontal_count(A) * vertical_count(A)`. The selected
channel is positive exactly when `P>0`; the background is `B-9P`, where the
in-grid product `B` is in `[1,9]`. The exhaustive finite proof covers all 163
local count tuples with zero failures.

## Verification

- Full checker and strict shape inference/data propagation: pass.
- Standard domains, no banned ops, maximum Einsum arity 10, Conv-family UB0.
- Runtime shape trace: all six node outputs truthful; nonfinite values 0.
- Known corpus: 265/265 in disabled/default ORT with threads 1 and 4.
- Fresh seeds `192800661` and `192930007`: 5000/5000 each in disabled ORT.
- Independent dual-mode rerun of both seeds: 5000/5000 in disabled and
  5000/5000 in default ORT for each seed; runtime errors 0, nonfinite values 0,
  and threshold outputs equal across modes on all 10,000 examples.
- Readable decoded-rule reference: 10,000/10,000.
- Positive margin minimum 1.0; maximum nonpositive value 0.0.

Evidence:

- `audit/task192_exact_poly.json`
- `fresh_dual.json`
- `build_selected_masks.py`
- `audit_candidate.py`
- `audit_fresh_dual.py`

Independent review in `agent_review_task192_112/REPORT.md` reproduced the
factorization proof, cost1197, known four-config results, strict/truthful/UB0
gates, and raw equality to the prior exact model on two different2000-case
fresh seeds in both ORT modes.

The candidate is staged as `others/71407/task192.onnx`. Root submission and
score-ledger files remain unchanged; the candidate is not counted as an
LB-fixed gain until its exact SHA is externally confirmed. In the later
8009.46 checkpoint, authority task192 remained byte-identical at cost1609, so
the candidate and proof rebased without change.
