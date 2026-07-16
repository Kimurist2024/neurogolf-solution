# Lane C13 — archive/headroom audit

## Result

No model is admissible. The exact `submission_base_7999.13.zip` remains
unchanged and lane C13 contributes **+0.000000000000000**.

- exact archive SHA-256:
  `a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`;
- tasks audited: 285, 365, 370, 378, 382, 394, 398;
- retained candidates audited: **46**;
- accepted candidates: **0**;
- root ZIP, CSV, and score ledger: **not modified**.

Every loadable model was evaluated with the same actual/static cost path, full
ONNX checker, strict shape/data propagation, standard-domain/operator checks,
function/nested/sparse checks, Conv/QLinearConv bias inspection, runtime shape
tracing, and all known examples under `ORT_DISABLE_ALL` and default ORT where a
session could be created. `candidate_audit.json` contains the per-model record.

## Cost and correctness summary

| task | exact cost | retained costs | known/runtime outcome | decision |
|---:|---:|---|---|---|
| 285 | 8717 | 8767, 8732, unsupported, 8845, 8862, 8887, 8896, 9036 | all runnable models are more expensive; several default-ORT failures/shape cloaks | reject all |
| 365 | 1381 | 1199, 1337, 1377 | all known-complete, but all failed prior fresh probes; cost-1199 was 2559/3000 | reject all |
| 370 | 1011 | 1316, 1322, 1352, 1385, 1376, 1965, 1634, 2018 | all more expensive; prior sound rebuild costs 2687 | reject all |
| 378 | 525 | 543, 554, 546, 548, 529, 531, 594, 614 | all complete on both runtimes, all more expensive | reject all |
| 382 | 820 | 808, 811, 814, 817, 818, 818, 818, 818 | only cost-814 is known-complete with optimizations disabled; every candidate fails default ORT | reject all |
| 394 | 350 | 177, 342, 20141, 348, 348, 351, 352, 355 | all cheaper models are known-wrong; complete models cost at least 351 | reject all |
| 398 | 350 | 332, 332, 332 | every candidate is wrong on 268/268 known cases | reject all |

The exact task382 model itself is not a sound adoption source: it is 254/266
under disabled optimization and produces 266/266 runtime errors under default
ORT. Lane C13 therefore does not treat “smaller than exact task382” as enough;
the candidate still has to pass the independent soundness gates.

## Decisive rejects

### task365

The apparent cost-1199 winner is hash
`56c70262d89d54954d4cf4c5f9ba651078a391d9b83cf15bc0c94fd6e5681b9d`.
It passes all 266 retained examples under both runtimes, but prior independent
generator validation was only **2559/3000** (441 mismatches). Runtime tracing
also finds four false static declarations, including a declared length 3 that
is actually 64 and declared 1x1x1x1 tensors that are actually 1x1x64x10 and
1x1x64x64. The cost-1337 and cost-1377 alternatives likewise failed the prior
quick fresh screen (3/5 each); the latter has two large false shapes. None is
eligible for the final fresh gate.

The recovered rule is: choose the separated rectangle with the unique maximum
red-cell count and crop its bounding box. The exact base passes the previously
recorded 3000-case audit; no cheaper retained graph implements that rule
exactly.

### task382

The cost-814 candidate is hash
`ac0d47cfa37effc8453f77a8498a0a8516ef63ab610ed6e4fb49afef485dee29`.
It passes 266/266 known and the supplied fresh **5000/5000** only under
`ORT_DISABLE_ALL`. Under default ORT it creates no usable outputs: supplied
fresh evidence is **0/5000 with 5000 runtime errors**, and this audit
independently reproduces **0/266 with 266 errors** on the known suite. It also
contains 23 declared/runtime shape mismatches; most importantly, output is
declared 1x10x1x1 while the actual tensor is 1x10x30x30. This is the prohibited
shape/value cloak, not an admissible optimization. The other seven candidates
are wrong on 12 known cases and fail every default-ORT example; r08 additionally
has a dynamic QLinearConv bias whose channel safety cannot be proven.

The true rule tracks red boundary markers and shifts cyan sources along the
perpendicular direction by the cumulative number of passed red markers, with
orientation flips/gravity. No retained graph is both smaller and sound.

## Fresh gate

Admission requires complete known correctness, a strictly lower actual cost,
clean structure/static/runtime-shape and bias checks, then independent
**5000/5000** validation with zero errors under both default ORT and
`ORT_DISABLE_ALL`. No candidate passed the prerequisite gates, so C13 did not
spend a new 10,000-case dual-runtime run on already disproven models.
`fresh_evidence.json` records the decisive prior evidence.

## Artifacts

- `candidate_audit.json`: complete base/candidate structural, cost, trace, and
  known-suite audit;
- `winner_manifest.json`: empty acceptance manifest;
- `rejected_manifest.json`: task-level rejection record;
- `fresh_evidence.json`: prior independent fresh evidence and gate decision;
- `audit_candidates.py`: reproducible auditor.
