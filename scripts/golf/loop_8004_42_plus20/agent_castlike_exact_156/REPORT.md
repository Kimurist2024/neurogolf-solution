# CastLike exact lane 156 — independent strict-lower audit

## Outcome

No candidate is admissible. The winner set is empty and projected gain is
`+0.0`. Fresh was not run because no candidate passed the mandatory pre-fresh
gates. Root `submission.zip`, `all_scores.csv`, `others/`, and docs were not
modified.

The immutable authority is LB **8009.46**:

- `submission.zip` and `submission_base_8009.46.zip` SHA-256:
  `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
- `all_scores.csv` SHA-256 before/after:
  `8c99379ce1c77e6894d53b593ccb5ae10f01c3c70e5666f832b1b2454c709b78`

## Decisive audit

The declared profiler, the competition-style all-known profiler, and the
independent team validator were all run from the pinned files. The two actual
profilers agree exactly for every authority and candidate.

| task | declared cost | competition actual cost | candidate node-shape mismatches | known default t1/t4 | known disabled t1/t4 | decision |
|---:|---:|---:|---:|:---:|:---:|---|
| 071 | 188 -> 187 | 188 -> 187 | 3 | 265/265 both | 0/265, 265 errors both | reject runtime-error task |
| 133 | 4393 -> 4347 | 4393 -> 4347 | 30 | 267/267 both | 0/267, 267 errors both | reject structure/runtime/private |
| 216 | **1025** -> 903 | **1499** -> 903 | 53 | 266/266 both | 0/266, 266 errors both | reject value-info mirage/runtime/private |
| 285 | 8623 -> 7273 | 8623 -> 7273 | 57 | session failure both | 0/265, 265 errors both | reject runtime/private |
| 388 | 85 -> 84 | **305 -> 1599** | 14 | 266/266 both | 266/266 both | reject actual-cost regression |

For task216 the warning is confirmed: authority `cost_of` reports
`979 + 46 = 1025`, while both all-known profilers observe
`1453 + 46 = 1499`. The candidate really profiles at `857 + 46 = 903`, but
that apparent saving cannot be used: removing the CastLike type witness changes
the ORT buffer plan and every known `ORT_DISABLE_ALL` execution fails.

Task285's proposed `8623 -> 7273` reduction is also real as an actual profile,
but the candidate fails every disabled known execution and neither authority
nor candidate can create a default ORT session because of the incumbent
`Concat` inferred/declared shape conflict.

Task388 is the only candidate that is raw-equal to the immutable authority and
correct on every known case in all four runtime configurations. Its declared
`85 -> 84` improvement is a profiler illusion in the opposite direction:
all-known actual memory changes from 283 to 1578, so total cost regresses from
305 to 1599.

## Formal Cast semantics

Every candidate is byte-for-byte the expected one-node rewrite after dead
initializer pruning:

| task | target type | saturate |
|---:|---|---|
| 071 | INT32 | not applicable |
| 133 | FLOAT16 | CastLike=1, Cast=1; not a float8 target |
| 216 | UINT32 | not applicable |
| 285 | INT32 | not applicable |
| 388 | INT8 | CastLike=1, Cast=1; not a float8 target |

At ONNX value semantics, `CastLike(x, fixed_witness)` and
`Cast(x, to=witness_dtype)` perform the same conversion here. Saturation only
changes conversions to float8-like targets, and none of the five targets is
float8-like. This proves the numeric operator identity, but it does **not**
prove operational pass-through for these malformed compact graphs: the second
input also affects shape inference and ORT buffer allocation. Four candidates
therefore diverge by runtime failure; task388 preserves outputs but loses its
cost advantage.

## SOUND gates

- Full checker and strict data propagation pass for all five candidates.
- Conv-bias undefined-behavior findings are zero for all five.
- All models are lookup-free. Task071 is giant-Einsum (`max inputs = 39`);
  the others are not.
- Candidate task133 contains a nonfinite initializer and fails the finite
  structure gate.
- Direct traces expose every node output and compare runtime shapes with strict
  inferred declarations. Authority and candidate mismatch counts are identical:
  3, 30, 53, 57, and 14 for tasks 071/133/216/285/388 respectively. Incumbent
  defects were treated as evidence, not as an exemption for descendants.
- Tasks133/216/285 are in the private-zero or unsound-incumbent catalog. Their
  authority all-input pass-through condition is not closed because their
  candidates fail the required runtime/shape gates.
- No candidate passed structure + truthful node shapes + four-config known +
  actual strict-lower + private policy. Consequently, the requested two-seed
  fresh `>=90%` stage was correctly skipped.

## Evidence

- `audit_candidates.py`: reproducible fixed-SHA audit.
- `result.json`: full independent profiles, all four known runs, raw/sign
  differential results, structure, formal semantics, and runtime traces.
- `winner_manifest.json`: empty promotion manifest.

