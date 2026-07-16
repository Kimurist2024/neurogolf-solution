# 7999.13 headroom lane report

## Outcome

Sixteen non-tail, non-catalog high-cost tasks were screened against
`submission_base_7999.13.zip`. The scan resolved 14,610 archive/file references
to 1,280 unique ONNX models. Only two models were both visible-correct and
strictly cheaper. Fresh verification rejected task131 and left one isolated
task382 candidate.

No root submission, `best_score.json`, `all_scores.csv`, handcrafted artifact,
or root artifact was changed.

## Approved isolated candidate

### task382

- Candidate: `candidates/task382.onnx`
- SHA-256: `ac0d47cfa37effc8453f77a8498a0a8516ef63ab610ed6e4fb49afef485dee29`
- Actual cost: `820 -> 814`
- Predicted score gain: `+0.007343974255758505`
- Visible: `266/266`, both library and official decode paths pass
- Fresh generator: `3000/3000`, zero mismatches
- Baseline on the same fresh gate: `2854/3000` (`146` mismatches)
- Minimum non-zero raw margin: `0.99462890625`
- Full ONNX checker and strict shape inference: pass
- Foreign domains, banned ops, nested graphs, functions, sparse initializers:
  none
- Conv-family bias UB: zero, both for the model and the isolated ZIP

The candidate fixes the incumbent's out-of-range fp16-to-uint8 narrowing by
using the generator-bounded `float16 -> int8 -> uint8` path. Its rule was
derived from `task_f15e1fac.py`, and the existing reproducible implementation is
`scripts/golf/scratch_codex/task382/build_factor_reordered.py`.

The isolated archive is
`submission_7999.13_task382_isolated.zip`, SHA-256
`6400d01af82d3a6b5d124eed2935580ae8972c5039800991b09050f98d0d3ae0`.
It preserves all 400 baseline member names, order, and ZipInfo metadata; only
`task382.onnx` differs and CRC validation passes.

## Residual task382 risk

The checker-visible output declaration is `[1,10,1,1]`, while ORT produces
`[1,10,30,30]`. Full checker, strict inference, visible, official decode,
fresh-3000, actual-cost, and bias gates pass, but this shape-spoof warning makes
the candidate inappropriate for an unisolated batch. Generic canvases without
the red/cyan markers also crash; those inputs are outside the authoritative
generator domain. Submit only as the isolated A/B ZIP if accepting that
platform-risk tradeoff.

## Rejected candidate

### task131

- Candidate SHA-256:
  `a13e6337acc30ddc9bc7f3276f3e464cc8144c12d40577bdd625d721ab1db182`
- Visible: `266/266`
- Fresh generator: `895/3000`; `2105` failures
- Runtime errors include `ScatterND invalid indice=255`
- Historical evidence: it was placed in `submission_1200merge.zip`, then
  reverted before both `submission_1200merge2.zip` and
  `submission_1200final.zip`.

This model is a visible-set false positive and must not be merged despite the
apparent `746 -> 596` cost reduction.

## Scan coverage

Tasks scanned:
`002, 005, 012, 013, 080, 089, 107, 125, 131, 280, 340, 349, 364, 367, 370, 382`.

The per-model evidence is in `scan_results.json`; the accepted/rejected decision
is machine-readable in `winner_manifest.json`.
