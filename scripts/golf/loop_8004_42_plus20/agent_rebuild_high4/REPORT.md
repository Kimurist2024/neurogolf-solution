# Wave 5 true-rule rebuild: tasks 118 / 173 / 077 / 054

## Result

No candidate is adoptable. All four healthy rebuilds pass the correctness and
runtime gates, but each is substantially more expensive than the immutable
`submission_base_8004.50.zip` member. The baseline members were not changed and
no ZIP was produced.

| task | fixed baseline cost | healthy candidate cost | fresh seed A | fresh seed B | decision |
|---:|---:|---:|---:|---:|---|
| 118 | 3,665 | 51,350 | 1948/2000 (97.40%) | 1950/2000 (97.50%) | reject: +47,685 cost |
| 173 | 3,525 | 18,494 | 2000/2000 | 2000/2000 | reject: +14,969 cost |
| 077 | 3,364 | 17,653 | 2000/2000 | 2000/2000 | reject: +14,289 cost |
| 054 | 2,291 | 49,618 | 2000/2000 | 2000/2000 | reject: +47,327 cost |

The user's 90% fresh threshold does not change these decisions: candidate cost
must also be strictly lower than the fixed member.

## Candidate provenance and gates

- task118: generator-derived plus reconstruction (`scripts/golf/scratch/task118/cand_paint.onnx`).
  It has no lookup, private-zero branch, shape cloak, or Einsum. Known 267/267;
  the residual 2.5% error is expected because the generator is not an injective
  input-to-output mapping when a cross is completely hidden in gray static.
- task173: safe coordinate engine built as `task173_coord_safe.onnx`. It uses the
  generator maxima `K_PIX=51`, `K_PROTO=3`, `K_CENTER=15`, `K_ANCHOR=12`, not
  sample-fitted truncation. Its sole `CenterCropPad` honestly pads a 25x25 grid
  to 29x29; it does not under-declare runtime shapes.
- task077: exact rectangle detector built as `task077_exact_convpack.onnx`. The
  original three-input floating Einsum packer was replaced with a normal Conv;
  the remainder is an exact uint32 per-width band scan.
- task054: exact vector reference (`scripts/golf/scratch_codex/task054/task054_vector.onnx`).
  It uses ordinary Conv/CumSum/ScatterND and no shape cloak or Einsum.

Every healthy candidate passed:

- known examples: 100%;
- two independent fresh seeds (2,000 each), with the task118 non-injective limit
  still above the authorized 90% threshold;
- ORT CPU with `ORT_DISABLE_ALL` and `ORT_ENABLE_ALL`, zero disagreements/errors;
- ONNX full checker and strict shape inference, all dimensions static;
- banned/nested graph scan, margin scan, and Conv-family bias check (0 UB).

Detailed evidence is in `candidate_task*_audit.json`. SHA-256:

- `task173_coord_safe.onnx`: `4d2a8e5bec512afb7064602dfac3a14808ad3dd2bcf91d52eff8e353f3972985`
- `task077_exact_convpack.onnx`: `e78b782afb2428019d59949562692ba164bbe5b93e70853581c56ba594da8992`

## Stop-floor evidence

The fixed costs are below the honest structural floors, so further local
shaving cannot cross the adoption boundary:

- task118: the honest two relevant-channel slices alone cost 2,800 bytes each;
  a Conv-decode alternative costs 3,600 bytes before the required paint mask.
- task173: honest decode (2,500), label cast/flat grids, padded probe grid, and
  output scatter already exceed 3,525 before sprite routing.
- task077: the measured exact height-2/height-3 band scans are the dominant
  ~13 KB core; the complete healthy candidate costs 17,653.
- task054: one decoded f32 label grid already costs 3,600, exceeding 2,291
  before box/marker/line/star detection. The exact rebuild costs 49,618.

Alternatives rejected at the architecture level were: task118 full-grid
QLinearConv after honest 10-channel Cast (Cast alone 9,000 bytes), task173 unsafe
TopK/K truncation, task077 uniform-reach propagation (known width-7 miss), and
task054 one-axis separable banding (fails the mandatory three-box L fixture).
Each either exceeds the baseline before completing the rule or violates the
correctness/safety constraints.

## Baseline observations (not adoption candidates)

Independent 1,000-case seeds on the fixed members measured:

- task118: 85.7% / 87.8%; below the new 90% threshold, but it is an already-fixed
  LB member rather than a new candidate.
- task173: 98.5% / 98.9%.
- task077: 98.8% / 98.3%.
- task054: 99.3% / 99.5%; its optimized-ORT session rejects a declared/runtime
  shape merge, while the official `ORT_DISABLE_ALL` session runs. This is another
  reason not to derive a new candidate from that member.

These measurements are recorded in `baseline_task*_audit.json`.
