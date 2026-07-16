# Lane C22 — exact 7999.13 audit for tasks 009 and 076

No candidate is promotable. Projected gain is **+0.0**. The exact archive,
score CSV, best-score ledger, and handcrafted artifacts remain unchanged.

## Exact authority and actual cost

- Archive: `submission_base_7999.13.zip`, SHA-256
  `a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`
- task009: cost **2619**, SHA-256
  `372fef762ffbc873f8c6ef0f3e2f59478773e17702f4129d5e7e9ce8c783bfaa`
- task076: cost **2550**, SHA-256
  `9d31114f8af80bf54b6c908ad61eadd6dbe4fb63f52b5b97ecb70f1f0fcce791`

The archive still has 400 unique entries and passes `unzip -t`.

## task009: sound incumbent is already the raw-rule rebuild

The authoritative generator is `task_06df4c85.py`. It creates a spacing-2
coarse line-grid. The output connects every equal-colored nonzero pair sharing
a coarse row or column, then expands the connected bitmap back to the physical
line-grid. The readable rule passes known **265/265** and fresh **5000/5000**.

The exact 2619 graph is a standard-domain, shape-consistent compiler for this
rule: full checker and strict inference pass, declared/runtime shape mismatches
are zero, and there are no banned ops, functions, sparse initializers, giant
Einsum, or lookup nodes. In this lane it passes:

- known: 265/265 in `ORT_DISABLE_ALL` and default ORT;
- fresh seed 92201: 5000/5000 in each ORT mode, wrong 0, errors 0.

The cost-2072 archive line was excluded as instructed. It contains 101
`TfIdfVectorizer` lookup nodes and already has retained independent catastrophic
fresh failures.

The other strict-cheaper archive graph costs 2457 and is structurally clean on
the stored set, but it prunes necessary scalar row/column propagation. On the
same 5000 fresh cases it scores **4327/5000 (86.54%)** in both ORT modes, with
673 wrong and zero runtime errors. The first mismatch is valid case 5 on a
26x26 line-grid. It is rejected.

Earlier generator-first rebuild work tested procedural concat, row-gather,
output-index, dtype, deduplication, and local scalar-rewire forms. Correct
rebuilds cost at least the exact incumbent; every apparent cheaper propagation
prune has valid border/extra-line counterexamples. No new sound cheaper form is
supported by the rule or measurements.

## task076: deterministic sound rebuild is impossible

The authoritative generator is `task_36d67576.py`. It creates one fully visible
colored sprite and 2–3 partial copies whose blue/green pixels must be restored.
The required target orientation is not always encoded by the visible yellow/red
pixels.

The constructive witness in `spec_noninjective_proof.py` was rerun. Two valid
parameterizations, `megarotates=[0,3,3]` and `[0,1,1]`, produce byte-identical
input grids but output grids differing at 12 cells. The common input SHA-256 is
`be1adb70ce87d233cb35a52b0c1f440ce11683aae693fafb2791a820acfc6bdf`;
the two output hashes differ. Therefore no deterministic input-only ONNX can be
fresh-exact over the full generator distribution. A claimed sound rebuild
would be false, so none was emitted.

The exact 2550 member itself confirms the risk: it has 30 declared/runtime
shape contradictions and 19 CenterCropPad nodes. It passes 266/266 known under
`ORT_DISABLE_ALL`, but default ORT rejects its CenterCropPad shape contract at
session creation. It is an authority baseline, not an eligible C22 candidate.

All eight unique archive leads fail even earlier. Every one uses unsupported
int8/uint8 TopK under `ORT_DISABLE_ALL`; default ORT also rejects each inherited
CenterCropPad contract. Thus known-all/both-ORT and runtime-error-zero fail
before fresh testing. Their static filename costs are not executable costs.

## Gate and evidence

`candidate_audit.json` records actual scoring, both-ORT known checks, full
checker, strict inference with data propagation, runtime shape traces, domains,
banned/nested ops, functions, sparse initializers, lookup/Einsum indicators,
and convolution-bias checks. Only the executable strict-cheaper task009 lead
advanced to fresh5000 in both ORT modes, where it failed decisively.

Evidence files:

- `candidate_audit.json`
- `reference_task009.json`
- `fresh_task009_{base,r02}_{disabled,default}_5000.json`
- `fresh_task076_base_default_5000.json` (session rejection evidence)
- `generator_analysis.json`
- `rejected_manifest.json`, `winner_manifest.json`
- `validation/root_integrity.json`
