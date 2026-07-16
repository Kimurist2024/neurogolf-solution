# Lane C15 — exact-base sound audit, tasks 112/148/212/301/316/325/341

## Result

No candidate is admissible. Lane C15 contributes **+0.0**, and the exact
`submission_base_7999.13.zip` remains unchanged.

- baseline SHA-256:
  `a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`;
- exact bases remeasured: **7**;
- local/spec/history probes fully audited: **20**;
- additional harvest rows statically screened: **10**;
- archive candidates retained for these tasks: **0**;
- accepted candidates: **0**;
- root ZIP, CSV, score ledger, and shared model artifacts: **not modified by C15**.

Every loadable probe was remeasured through the scorer's actual cost path and
tested on the complete known corpus under `ORT_DISABLE_ALL` and default ORT.
The audit also ran full ONNX checking, strict shape/data propagation,
standard-domain/banned/function/nested/sparse checks, Conv-family bias checks,
and runtime tracing of every intermediate. The complete per-model evidence is
in `candidate_audit.json`.

## Summary

| task | exact cost | audited probe costs | decisive result |
|---:|---:|---|---|
| 112 | 422 | 354, 498, 15325, 422, 420, unscored | cost-420 is known-complete only in disabled mode; default ORT fails, strict inference fails, and 11 shape declarations are false |
| 148 | 265 | 265, 264, 15402, 7708 | sole cheaper model is wrong on 5/266 known cases in both modes |
| 212 | 240 | 4398 | incumbent is already memory-zero/direct-output; all prior sub-240 structural searches failed |
| 301 | 240 | 1991 | no checker-valid sub-240 representation; harvest alternatives are giant-Einsum or cost floor 1141 |
| 316 | 246 | none | historical floor 406; sparse projection shaves fail full shape inference |
| 325 | 235 | 249, 249, 249, 2955, 253 | every correct probe is dominated; both cost-249 shift probes are 0/266 |
| 341 | 260 | 261, 260, 36256 | equal-cost probe is 0/266; correct alternatives are more expensive |

## Task findings

### task112

The true rule completes four red reflections around a data-dependent green
2x2 pivot. The exact cost is 422. The only known-correct cheaper file is the
cost-420 affine-grid/ScatterND graph. It passes 266/266 with optimizations
disabled, but default ORT rejects its malformed two-axis `CenterCropPad` shape
input. Strict shape/data inference also fails, and runtime tracing records 11
false declarations, including output `[1,1,1,1]` while execution returns
`[1,10,30,30]`. It is therefore not eligible for fresh promotion.

The apparent sign-table factorization was independently confirmed invalid:
cost-354 and cost-498 variants raise ScatterND update/index shape errors on
all 266 known inputs. A fully truthful spec-derived control has zero shape
contradictions and passes both runtimes, but costs 15325. No lookup approach
was considered or accepted.

### task148

The exact cost-265 graph is complete under both runtimes. Replacing the
two-element `z5hi` table with a scalar reaches cost 264, but returns the wrong
answer on 5/266 known cases in each mode; prior fresh was 99/100. The
known-complete spec-derived controls cost 7708 and 15402. The one-parameter
shave is numerical, not algebraically neutral, so task148 remains frozen.

### task212

The exact model is one direct-output tensor contraction: memory 0, parameters
240. The truthful spec control costs 4398. Earlier exhaustive work tested all
123 single-axis/single-slice collapses and all six coupled rank-3 reductions;
none survived fresh screening. Dense-to-sparse changes are not valid in the
official-compatible graph path. There is no credible sub-240 candidate.

### task301

The exact cost-240 graph is a direct-output tensor network with truthful
runtime shapes. The separate spec engine costs 1991. Harvest history adds one
static-floor-1141 graph and one 51-input giant-Einsum graph, neither eligible.
The only theoretical cost-239 sparse coefficient representation fails full
shape inference because ONNX presents the sparse input to `Einsum` as rank
zero. No candidate reaches the known gate.

### task316

The exact cost is 246 and is complete under both runtimes; prior independent
fresh is 5000/5000. No retained archive candidate exists, and the sole harvest
lineage starts at static cost 406. The obvious sparse projection opportunity
is checker-invalid at the `Einsum` boundary. Direct-output, zero-point, and
scale-sharing searches have already exhausted the smaller standard carriers.

### task325

The exact 7999.13 archive already contains a cost-235 model, below all local
spec-derived candidates. The previously strong prefix-outer model is exact and
has substantial fresh evidence, but remeasures at 249. The local7 model costs
253 and the fully truthful control costs 2955. Two reverse-shift probes also
cost 249 but are wrong on every known case. Historical candidates start at
static floors 253/295 or use a 20-input giant Einsum, so none can improve 235.

### task341

The exact archive member costs 260. The known-complete center-index rebuild
costs 261 and retains 12 runtime/declaration contradictions. The equal-cost
zero-carrier attempt is wrong on all 266 known inputs, while the fully expanded
Cast-attribute control costs 36256. Harvest floors are 280 and 1427. No
strictly cheaper known-complete candidate exists.

## Fresh gate

Adoption requires a strictly cheaper model that is complete on all known
inputs in both ORT modes and passes the structural truthfulness gate before an
independent 5000 cases per mode. No C15 candidate passed those prerequisites.
Accordingly, no rejected graph consumed a new dual-5000 run. Prior decisive
fresh evidence is indexed in `fresh_evidence.json`.

## Artifacts

- `candidate_audit.json`: actual cost, known dual-runtime, structural, bias,
  and runtime-shape evidence for 7 bases and 20 probes;
- `historical_scan_summary.json`: ten harvest rows and why none was retained;
- `fresh_evidence.json`: gate requirements and prior decisive fresh evidence;
- `rejected_manifest.json`: task-level rejection decisions;
- `winner_manifest.json`: empty acceptance manifest;
- `validation/root_integrity.json`: immutable baseline hash check;
- `audit_candidates.py`: reproducible C15 auditor.
