# Reduction-predicate exact fusion scan (lane 266)

## Result

No policy-safe winner survived the runtime-shape gate.  The pinned authority
archive remained unchanged at
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`.

## Exhaustive scope and proof rules

- All 400 authority models scanned.
- 57 models contain a relevant reduction.
- 129 ReduceSum/ReduceMax/ReduceMin nodes examined.
- 26 reduction inputs have a formally proved binary domain.
- Five Any/All predicate-fusion units and one Reduce/Squeeze unit survived the
  finite proof.
- Two strict-lower task candidates were constructed; neither was accepted.

Binary status is not inferred from a convenient value range.  It must trace to
an ONNX bool comparison (`Equal`, `Greater`, etc.), proved boolean logic, a
bool-preserving shape/index operator, a Cast/CastLike of that bool tensor, or a
OneHot whose values initializer is a subset of `{0,1}`.  General integer and
floating reductions are excluded.

For every candidate comparison, the scanner evaluates the complete aggregate
truth table.  ReduceSum is checked on every integer from zero through the exact
static reduced-element count; ReduceMax/Min are checked on both binary values.
Only exact Any or All tables survive, including operand-reversed comparisons
and an optional Not.  Eight synthetic tests cover `sum>0`, `sum>=1`,
`sum==full_count`, max/min predicates, operand reversal, Not(`sum==0`), and
rejection of the unsafe `sum==1` generalization.

Bool ReduceMax/Min is only introduced for opset 20 or newer.  The task366 graph
is opset 18, so its uint8 0/1 carrier is retained; the transformation only
removes comparisons whose numeric result is already exactly 0 or 1.

## task319 — rejected truthful-policy candidate

The proved chain is:

`Equal eq1[1,1,2] -> ReduceMin(axis=2, keepdims=1) -> Squeeze -> scalar`

Both unreduced axes are statically singleton, so a single
`ReduceMin(all axes, keepdims=0)` is exactly equivalent.  This removes one bool
scalar intermediate.

- Official cost: 1003 -> 1002 (memory 863 -> 862, params 140 unchanged).
- Full checker, strict data-propagating inference, structural audit, UB0: pass.
- Official known correctness: pass.
- Known4 disable-all/default x threads 1/4: 267/267 raw and threshold equal,
  zero authority/candidate errors in every configuration.
- Runtime-shape gate: fail; authority and candidate each have 26 declared vs
  actual mismatches.
- Candidate SHA-256:
  `85ce0a8f2440adeac0e309fcf0209b7950a50d8a772084f621abd17efac5d060`.

## task366 — rejected truthful-policy candidate

Five independent chains at reduction nodes 161, 180, 256, 275, and 340 are:

`ReduceMax(uint8 binary) -> Greater(0) -> Cast(float16)`

Because ReduceMax of a 0/1 carrier is already exactly 0 or 1, each final Cast
can read the ReduceMax result directly.  The five bool comparison scalars are
deleted.  The shared zero initializer receives no deletion credit.

- Official cost: 7987 -> 7982 (memory 7622 -> 7617, params 365 unchanged).
- Full checker, strict data-propagating inference, structural audit, UB0: pass.
- Official known correctness: pass.
- Known4 disable-all/default x threads 1/4: 255/255 raw and threshold equal,
  zero authority/candidate errors in every configuration.
- Runtime-shape gate: fail; authority and candidate each have 98 declared vs
  actual mismatches.
- Candidate SHA-256:
  `877e30f89da200b56ab1d79cc215b8c1a2db29369d5b3c27b2c54d5f724b0bb9`.

Fresh 2 x 1000 testing was skipped because both strict-lower candidates failed
the required runtime-shape truthfulness gate.  Machine-readable proofs and all
gate results are in `scan_result.json`; rejected candidates are retained under
`candidates/` for reproducibility.

