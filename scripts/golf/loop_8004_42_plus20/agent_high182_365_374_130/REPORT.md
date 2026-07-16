# Lane 130 — task182/task365/task374 exact mem/param audit

## Outcome

No safe strictly cheaper candidate was found. Winner count is **0**, admitted
cost delta is **0**, and projected gain is **+0.0**.

The 8009.46 authority archive remained SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
before and after the run. This lane did not edit `submission.zip`,
`all_scores.csv`, `others/`, or shared `artifacts/`.

## Independent true rules

`run_lane.py` implements each generator rule independently of the authority
ONNX and every transformed candidate:

- **task182 (`776ffc46`)**: identify the sprite enclosed by the unique
  complete gray 7x7 frame and recolor every external component with the same
  translation-normalized shape to the framed sprite's color.
- **task365 (`e50d258f`)**: separate the fully occupied rectangles and extract
  the rectangle with the greatest number of red cells.
- **task374 (`ea32f347`)**: rank the three separated gray lines by distinct
  length and recolor shortest/middle/longest as 2/4/1.

| task | known | fresh seed A | fresh seed B | errors |
|---:|---:|---:|---:|---:|
| 182 | 267/267 | 1500/1500 | 1500/1500 | 0 |
| 365 | 266/266 | 1500/1500 | 1500/1500 | 0 |
| 374 | 267/267 | 1500/1500 | 1500/1500 | 0 |

The references therefore pass all **800 known** and **9000 fresh** cases. Seeds
and counters are recorded in `audit/reference_audit.json`.

## Authority members and dual-ORT control

| task | authority member SHA-256 | scorer cost | known disabled/default | runtime-shape evidence |
|---:|---|---:|---|---|
| 182 | `625b31492d9135295229c67ca0322000a2ff351e81e627bb882b89dde6bfda97` | **949 = 893 + 56** | disabled 267/267; default load failure | 47 contradictions; 154,246 measured intermediate bytes |
| 365 | `85d63fa65d51d5aa065c5966725d092813c21b0e7a92453bf745d096371e3214` | **1355 = 1274 + 81** | 266/266 in both modes | 12 contradictions; 15,909 measured bytes |
| 374 | `93fb94260388ab83bc35043c0ee11ae08b1bf3e8fa962a3b47b08ba73794d24a` | **481 = 451 + 30** | 267/267 in both modes | 9 contradictions; 46,634 measured bytes |

The current task182 default session fails because a one-element
`CenterCropPad` shape is used with two axes. It remains the exact LB-white
authority member, but no new candidate may inherit that local failure.

For task365 and task374, both authority controls additionally passed raw
execution in both ORT modes on every known case and two fresh streams:

- task365: known 266/266 per mode; seeds `130200365` and `130300365`,
  each 1500/1500 per mode, zero wrong/errors.
- task374: known 267/267 per mode; seeds `130200374` and `130300374`,
  each 1500/1500 per mode, zero wrong/errors; minimum positive margin
  0.496576.

These controls fix the behavior that an exact candidate would have to preserve
bitwise.

## Type and initializer audit

All three authority graphs have no byte-identical same-shape initializer
groups, so direct initializer aliasing offers no reduction.

For task182, `cnt6` is `Mul(col_count_u8, 6)` and its input type is
`tensor(uint8)`. At opset 18, `Selu` admits only float16/float/double.
Therefore the proposed uint8 `Mul -> Selu` rewrite is schema-invalid.
Adding float casts around it would add charged intermediates and would no
longer be a local bitwise reduction. The machine-readable schema evidence is
in `audit/type_constraints.json`.

## Regenerated candidate scan

Eleven byte-distinct candidates were generated directly from current authority
members. No past payload was promoted or relied on as an equivalence proof.

### task182

Five variants were generated.

- Identity bypass SHA
  `d65c9a6f75477d219c9257693bcbcea86119c331cb889352262381bc9ede813e`
  is algebraically exact, but removing `sh19 = Identity(s19)` exposes
  `CenterCropPad` shape conflicts at `graypad19`, `spcrop19`, and
  `cc0`. Full checker and strict inference fail.
- Constant-fold and combined variants expose a larger set of malformed
  one-element/two-axis shape contracts and fail full/strict.
- Metadata normalization is unscorable; normalized-combined also fails
  full/strict.

No task182 candidate reaches a measurable strict-lower cost.

### task365

The metadata-normalized graph is unscorable.

The new fixed-shape fold SHA
`bc183bf653363480b723d2417209ef6a09e05404401cbf9b4cb5bcbb778debf1`
is algebraically exact: `tl_flat_u8` has model-contract shape `[64]`, so its
`Shape` output is always `[64]`. However, materializing that constant makes
the existing witness explicit and reveals seven shape contradictions, including
`__hc_tlflat_hid` dimension 64 versus declared 3 and several 64-versus-1
carrier dimensions. Full checker and strict inference reject it.

No task365 candidate obtains a valid new cost below 1355.

### task374

Four variants were generated.

- Fixed-shape fold SHA
  `4810ce84d5084075804e491d24b5baa3b60a547cdcfbf42eced81ac6410da150`
  replaces `Shape(input,start=0,end=1)` with the provably constant `[1]`.
  It exposes rank-4-versus-rank-1 contradictions in the three
  `CenterCropPad` carriers and fails full/strict.
- `CastLike(color_fake,i32dummy) -> Cast(color_fake,to=INT32)` SHA
  `b82ae4b99fbac6853d995f64f5f30aefd12b14fa2a2f29404a3f7f66d6b5faa1`
  is type-semantics exact and removes one parameter. Nevertheless the official
  profiler changes from **451 memory + 30 params = 481** to
  **847 memory + 29 params = 876**. It is not strictly lower.
- The combined shape-fold/Cast variant fails full/strict.
- Metadata normalization fails strict type inference.

## Final disposition

Stage totals:

- `REJECT_CHECKER_OR_STRICT_SHAPE=8`
- `REJECT_UNSCORABLE=2`
- `REJECT_NOT_STRICTLY_LOWER=1`
- `ADMIT=0`

Because no candidate survived the strict-lower and structural prerequisites,
no transformed ONNX was allowed to claim candidate fresh validation.
Candidate known/fresh/bitwise counts are therefore **not applicable**; the
authority controls and independent true-rule counts above are reported
explicitly. New cost/SHA/gain are **none / none / +0.0**.

## Evidence

- `audit/results.json`: authority profiles, known/default results, dual-ORT
  fresh controls, runtime-shape evidence, exact proofs, all candidate hashes,
  measured costs, and terminal stages.
- `audit/reference_audit.json`: all independent known/fresh rule checks.
- `audit/type_constraints.json`: task182 Selu schema proof.
- `manifest.json`: authority hashes and empty winner list.
- `baseline/`: exact authority-member snapshots.
- `candidates/`: rejected regenerated intermediates only.
- `candidate/README.md`: explicit no-winner marker.
- `run_lane.py`: reproducible audit and transform builder.

