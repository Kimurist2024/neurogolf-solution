# Lane 121 — task009/task076 sound memshave audit

## Outcome

No safe strictly cheaper candidate was found. Winner count is **0** and the
projected score gain is **+0.0**. This lane did not edit `submission.zip`,
`all_scores.csv`, `artifacts/`, or `others/71407/`.

The authority archive observed throughout the run was SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`.
Both assigned members remained byte-identical to the captured authority.

## task009

- Authority member SHA-256:
  `372fef762ffbc873f8c6ef0f3e2f59478773e17702f4129d5e7e9ce8c783bfaa`
- Official-like cost: **2619 = memory 2567 + params 52**
- Full checker and strict shape inference: pass
- Declared/runtime tensor-shape mismatches: **0**
- Known: **265/265** in `ORT_DISABLE_ALL` and **265/265** in default ORT
- Independent fresh seeds `121000009` and `121100009`: each **2000/2000** in
  both ORT modes, with zero wrong and zero runtime errors

The current graph is already a scalarized compiler of the full generator rule.
The truthful 2567-byte intermediate memory is dominated by the terminal label
map (900), scalar float cell decode (400), row-cell assembly (300), row-line
assembly (150), uint8 cell labels (100), and blank predicates (96). The
all-input-exact cleanup/dedup/no-op/CSE/constant-fold/optional-output/factor
scan found no semantic or metadata rewrite at all. Therefore it emitted no
task009 candidate; there is no strict decrease to validate.

## task076

- Authority member SHA-256:
  `9d31114f8af80bf54b6c908ad61eadd6dbe4fb63f52b5b97ecb70f1f0fcce791`
- Official-like cost: **2550 = memory 2498 + params 52**
- `ORT_DISABLE_ALL` known: **266/266**
- Default ORT: session creation fails on the inherited `CenterCropPad` shape
  contract
- Declared/runtime tensor-shape mismatches: **30**
- Truthful one-example intermediate bytes: **122843**, versus the scored 2498

More importantly, the authoritative generator is non-injective. Two valid
parameterizations with rotations `[0,3,3]` and `[0,1,1]` produce the exact same
input SHA-256
`be1adb70ce87d233cb35a52b0c1f440ce11683aae693fafb2791a820acfc6bdf`,
but their outputs differ at **12 cells**. No deterministic input-only ONNX can
be exact over the full generator relation, so a SOUND_REBUILD winner is
impossible without relying on private-distribution behavior.

Four mechanically generated variants were fail-closed: two retained the same
cost or became unscorable, and two constant-fold variants failed checker/strict
shape. None advanced to admission.

## Evidence

- `audit/results.json`: full costs, tensor breakdown, structural checks,
  dual-ORT checks, fresh runs, non-injectivity witness, and every rejected row
- `manifest.json`: authority members and empty winner list
- `baseline/`: byte-exact authority snapshots for the two assigned tasks
- `candidates/`: generated rejected mechanical variants only
- `candidate/README.md`: explicit no-winner marker

