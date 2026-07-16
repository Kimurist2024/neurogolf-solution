# tasks109/184/368/374 — high-memory / POLICY90 residual audit

## Outcome

**Promotion candidates: 0. Projected gain: `+0.000000`. Final verdict:
`NO_PROMOTION`.**

The immutable 8009.46 authority is `submission.zip`, SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`,
byte-identical to `submission_base_8009.46.zip`. The authority ZIPs and
`all_scores.csv` had identical hashes before and after the run. This lane did
not write to `others/71407`, any root submission, or any score ledger.

No candidate passed the pre-fresh gates, so the two large fresh streams were
intentionally not run. POLICY90 cannot rescue a cost, schema, complete-known,
shape-truth, nonfinite/margin, UB, or lookup failure.

## Authority profile

| task | member SHA-256 | official cost | known in four configs | declared/runtime mismatches | observed intermediate bytes | disposition |
|---:|---|---:|---|---:|---:|---|
| 109 | `2e7be8671e2e8abe9d3f2f77f0b068f54a70a584ce477affb28fee6372bd25ef` | **405 = 362 + 43** | 266/266 in disable/default × threads 1/4 | 10 | 63,587 | authority only; new graphs may not inherit cloak/lookup |
| 184 | `156fe12922d290876f63c210f9cec8252e308e1af0512e309cb2fc6fad8928fc` | **421 = 389 + 32** | disable: 169/169 ×2; default: 169 session errors ×2 | 6 | 64,961 | authority only; default-ORT and cloak failures |
| 368 | `0d950f5053aa62e7a3208be01514ad061b85580875e0e93aa7ee941cbacaa811` | **521 = 481 + 40** | 265/265 in all four | 2 | 45,476 | authority only; new graphs must declare `gn/qi` truthfully |
| 374 | `93fb94260388ab83bc35043c0ee11ae08b1bf3e8fa962a3b47b08ba73794d24a` | **481 = 451 + 30** | 267/267 in all four | 9 | 46,634 | authority only; exact local shaves expose the cloak |

For task184, 169 of the 266 stored cases are convertible to the canonical
30×30 carrier; the other 97 generator cases exceed the competition carrier.
All 169 compatible cases are included above. The default session fails at a
two-axis `CenterCropPad` receiving a one-element shape.

The independent generator rules are:

- task109 (`47c1f68c`): remove the pivot cross and reflect the colored quadrant
  into all four quadrants using the line color.
- task184 (`780d0b14`): split a 2×2 or 3×3 block grid at zero separators and
  emit each block's nonzero patch color.
- task368 (`e76a88a6`): recover the unique two-color 3×3/3×4/4×3 prototype and
  copy it into every equal-sized gray footprint.
- task374 (`ea32f347`): rank three separated gray lines by distinct length and
  recolor shortest/middle/longest as 2/4/1.

These are global geometry, segmentation/template, or ranking rules. Merely
correcting the current graphs' declarations would charge roughly 45–65 KB of
observed intermediate memory, far above the 405–521 authority frontier.

## Candidate verdicts

All models below were copied or regenerated under this lane directory and
checked with full ONNX checker, strict shape inference with data propagation,
runtime shape tracing, standard-domain/banned-op/Conv-bias gates, and the
official scorer. “Unscorable” means no valid official cost or gain exists.

| task | candidate / SHA-256 | cost vs authority | score gain | decisive evidence | verdict |
|---:|---|---:|---:|---|---|
| 109 | constant `Shape([12])` fold / `de3943bb2cf025343807e3f09f74bd64369dff2420dfc6885e30502474cffbf2` | **unscorable** vs 405 | N/A | strict inference exposes two `CenterCropPad` 12-vs-1 contradictions; 10 runtime shape mismatches; forbidden ArgMax/Gather/Scatter lookup chain | reject structure/cloak/lookup |
| 109 | retained GlobalLpPool shave / `f35d159228663ed68f4cb70249221a244d17799dc4c1caefc66136e20ad7d70c` | **422** vs 405 | **-0.041118** | not lower; 10 shape mismatches and ArgMax/Gather/Scatter lookup chain | reject cost/cloak/lookup |
| 184 | exact batch-axis identity bypass / `cd261e74325432c0b4b752649557a21f54882516c71db8a70d794b46fa3d4196` | **unscorable** vs 421 | N/A | bypass exposes `CastLike` channel 10-vs-1 contradiction under strict inference; 5 remaining shape mismatches | reject structure/cloak |
| 184 | historical ground-u8 / `d6010bf00f1fd62966a0d78c9c3798ac4a37b9ceb99e28b334d37cd3ac427f79` | **420** vs 421 | **+0.002378** | only actual lower candidate; 6 shape mismatches; disable 7/169 (4.14%) in both thread modes, first failure train[0], 4 cells; default 0/169 with 169 session errors; 24 values in `(0,0.25)`, minimum positive 0.038452 | reject known/runtime/cloak/margin |
| 368 | exact `CastLike→Cast(UINT8)` / `af7d8318545c6aa3023ebf28e9867127f163e21ae107d64ba6ff92457a05fad1` | **9,520** vs 521 | **-2.905400** | exact type rewrite exposes the full 1×10×30×30 `gn/qi`; still 2 declared/runtime mismatches | reject cost/cloak |
| 374 | three exact batch-axis identity bypasses / `9854cc294b89920af7c44ae7717e8252e474f0d67bd48ece27798f8c249e9098` | **unscorable** vs 481 | N/A | strict inference exposes three rank-4-vs-rank-1 `CastLike` contradictions; 6 remaining mismatches | reject structure/cloak |
| 374 | exact `CastLike→Cast(INT32)` / `b82ae4b99fbac6853d995f64f5f30aefd12b14fa2a2f29404a3f7f66d6b5faa1` | **876** vs 481 | **-0.599499** | deleting one parameter exposes 847 bytes of charged memory; 9 mismatches remain | reject cost/cloak |

The only numeric lower model is task184 ground-u8. Its four-configuration
failure is complete and deterministic: both disable-all modes have the same
7/169 accuracy, both default modes fail session construction, and no runtime
or output-shape error is hidden inside the disable-all results. It is far below
the normal 90% gate and also violates independent fail-closed conditions.

## Fresh disposition

Fresh eligibility requires, in this order:

1. official cost strictly below the authority;
2. full checker, strict data propagation, static positive shapes, standard
   domains, banned-op clear, and Conv-family UB0;
3. zero declared/runtime intermediate or output shape mismatches and no lookup;
4. every compatible known case correct in disable/default ORT × threads 1/4,
   with zero runtime errors, nonfinite values, output-shape failures, or unstable
   positive margins;
5. only then, two disjoint 1,500-case fresh streams with minimum accuracy 90%.

Zero candidates reached step 5. Large fresh generation was therefore skipped,
as required by the cost/structure gate. None of these four tasks entered a
private-zero percentage exception; such an exception would require a complete
pass-through guarantee rather than 90%.

## Evidence

- `report.json`: complete machine-readable authority, four-config known,
  runtime-shape, cost/gain, candidate failure, and protected-hash evidence.
- `audit_lane.py`: reproducible non-promoting audit and exact local transforms;
  it does not call `try_candidate.py`.
- `authority/`: byte-exact 8009.46 member snapshots.
- `candidates/`: all seven rejected probes, keyed by the SHA values above.

Final decision: **do not merge any model from this lane**.
