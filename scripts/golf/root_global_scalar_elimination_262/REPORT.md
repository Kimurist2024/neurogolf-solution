# Global scalar-carrier elimination 262

## Outcome

The all-400 scan found **no new mixed-use scalar removal**. Of 1,403 scalar
initializers, 86 individual consumer sites matched one of the requested local
identities, but every neutral/unsigned/Clip initializer also had at least one
unsupported use. Consequently zero mixed-use initializer could be removed as
a whole.

Twelve initializers had every use covered. All twelve are previously proved
positive-domain Mul/Div-to-Selu cases, not new neutral-carrier discoveries.
After full, strict, official, truthful-runtime-shape, known-four-config,
fresh-two-seed, error, nonfinite, and UB gates, only task205 cost 1041 survives.
It is dominated by the separately proved task205 cost-1038 rewrite, so this
lane's final disposition is **NO NEW PROMOTION**.

No root submission, stage, score ledger, `others/71407`, or other lane was
modified.

## Complete scan

Authority: `submission_base_8009.46.zip`, SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`.

| measure | result |
|---|---:|
| authority models scanned | 400 |
| scalar initializers | 1,403 |
| locally rewriteable consumer sites | 86 |
| initializers whose every use is covered | 12 |
| fully covered mixed-use initializers | **0** |
| strict-lower candidate records | 13 |
| strict-lower tasks before runtime gates | 9 |
| final survivors | 1 |

The 86 local sites comprise 54 unsigned `Greater(x,0)->Cast`, 14 proved
positive `Mul->Selu`, eight proved positive `Div->Selu`, five unsigned
`Clip(min=0)` omissions, three `Add(x,0)` identities, one `Mul(x,1)` identity,
and one signed `Sub(0,x)->Neg` site.

None of the unsigned Cast, Clip, zero, or one sites freed its initializer:
another consumer of the same scalar was unsupported. The scanner does not
partially rewrite such a scalar and then falsely claim its parameter was
removed.

`Sub(0,x)->Neg` was rejected for unsigned carriers. ONNX `Neg` does not admit
`uint8`/`uint16`; this blocks the tempting mixed-use zero initializers in
tasks048, 148, 157, and 366. Broadcast-changing identities are represented
with `Expand` and re-profiled including any new shape parameter. None produced
a lower fully covered neutral candidate.

## Lower-candidate gates

For each task, the cheapest non-conflicting combined candidate was taken to
the final gates.

| task | scan cost | official cost | truthful result | final disposition |
|---:|---:|---:|---|---|
| 013 | 356 | 356 | 5 shape mismatches; 2,270 nonfinite intermediate elements | reject |
| 066 | 561 | 561 | shapes match; one nonfinite intermediate | reject |
| 090 | 1049 | 1049 | 27 shape mismatches | reject |
| 134 | 422 | 422 | 6 shape mismatches | reject |
| 158 | 7577 | 7577 | truthful; known4 pass; fresh error asymmetry | reject |
| 205 | 1041 | 1041 | truthful; all gates pass | survivor, dominated |
| 209 | 1719 | 2085 | 16 shape mismatches; cloaked cost disagreement | reject |
| 233 | 7307 | 7307 | 28 shape mismatches | reject |
| 366 | 7985 | 7985 | 98 shape mismatches | reject |

Every candidate in the table passes full ONNX checking, strict shape
inference with data propagation, standard-domain/payload checks, and the
Conv-bias UB scan. The truthful gate exposes every statically typed node output
under disabled optimization and compares its declared and actual shape; it is
not inferred from checker success.

Task158 is a semantic/runtime rejection rather than a shape rejection. It is
raw-identical on all 266 known cases in disabled/default × threads 1/4. In
fresh seeds `262000159` and `262000160`, the authority throws an out-of-bounds
`ScatterElements` error on one and two distinct cases respectively, in all
four configurations, while the candidate returns normally. Thus 12
configuration-level authority errors are recorded and error behavior is not
preserved. No candidate error or nonfinite output occurred, but error symmetry
is mandatory.

## Surviving task205 candidate

Candidate:
`scripts/golf/root_global_scalar_elimination_262/candidates/task205_init_rowpow_thr.onnx`

- SHA-256:
  `2b6125fd7b39be8cf810986da8a010ba488c5e8fe5c44511b012d8759ad730f6`
- rewrite: both uses of float32 `rowpow_thr=1.9019999504089355` move to
  `Selu(alpha=1,gamma=rowpow_thr)` attributes; the initializer is removed;
- proof sources: `tall_f` is a cast-Boolean sum and `roww_max` is a maximum of
  nonnegative contractions, so both are finite nonnegative on valid one-hot
  inputs;
- official profile: memory 1031, params 10, cost **1041**, correct on all 266
  known cases;
- full/strict/standard-domain/payload/UB0 all pass;
- truthful trace: 38 outputs, zero shape mismatches, zero nonfinite elements;
- known four configurations: 266/266 raw-byte equal to authority in each,
  zero errors and nonfinite values;
- fresh seeds `262000206` and `262000207`: 1000/1000 raw-byte equal in each of
  disabled/default × threads 1/4, zero errors and nonfinite values.

Authority and candidate share the same generator-gold misses on those fresh
streams: 984/1000 and 986/1000 respectively. This lane's gate is exact
pass-through to the immutable authority; it does not misreport those finite
accuracy numbers as 100% generator correctness.

The survivor is not promoted because task205's independently proved candidate
`43c963c4...9f9a8` has official cost 1038 and stronger all-binary-mask
equivalence evidence. Replacing it with cost1041 would be a regression.

## Artifacts

- `scan.py`, `scan.json`: reproducible all-400, all-use scalar census and
  candidate construction;
- `audit.py`, `audit.json`: full/strict/official/truthful/known4/fresh2×1000×4
  audit;
- `result.json`: concise machine-readable disposition;
- `candidates/`: isolated lower probes only; no staging occurred.

Final decision: **NO NEW PROMOTION; retain the separately proved task205
cost-1038 candidate.**
