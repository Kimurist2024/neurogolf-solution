# Lane C37 — task162 truthful-shape repair audit at 8000.46

## Decision

No task162 candidate is eligible. The winner manifest is empty and this lane
contributes **+0.0**.

The authoritative `submission_base_8000.46.zip` member is SHA-256
`afcc4eaa...c3f0032`, cost **451 = memory 438 + parameters 13**. The copied
baseline bytes match the ZIP entry exactly.

## Cost-373 CSE diagnosis

The two supplied starting points were both audited:

- `lane_b20/task162_cse.onnx`: cost 451 because it retains unused stale
  value-info entries;
- `lane_exact_cse/task162.onnx`: cost 373 after removing those unused entries.

Their executable topology is the same. The cost-373 graph is not a valid
improvement:

- under ORT_DISABLE_ALL, **201 of 207** declared intermediate shapes disagree
  with runtime;
- under ORT_DEFAULT, session construction fails because a length-one target
  tensor is supplied to `CenterCropPad axes=[1,2,3]`;
- representative `csp30f` is declared `[1,1,1,1]` but runs as
  float32 `[1,30,30,30]`.

This is a direct hidden-shape contract, not score-bearing CSE.

## Truthful repair

`build_truthful_repair.py` performs a reproducible semantic repair:

1. replace every broadcast length-one CenterCropPad target with a constant
   whose length exactly matches its axes count;
2. remove six dead `ConstantOfShape` nodes;
3. discard all stale value-info;
4. run strict inference with data propagation and serialize every true static
   intermediate shape.

Both source variants converge byte-for-byte to SHA-256
`81852562...44da5e2`.

The repaired model passes:

- full ONNX checker and strict static inference;
- standard-domain, finite-initializer, no-function/subgraph/sparse checks;
- runtime shape trace in both ORT modes: **201 outputs, 0 mismatches** each;
- known fixtures in both ORT modes: **266/266**, wrong 0, errors 0 each;
- lookup audit: empty;
- Conv UB audit: empty.

The repair exposes the real cost:

```text
memory 699,739 + parameters 37 = cost 699,776
```

This cannot be reduced below 451 while retaining this topology. Its first
required intermediate alone is `csp30f`, float32 `[1,30,30,30]`, or 108,000
bytes. CSE removes duplicate computation, but cannot remove the tensor chain
that implements the row-major 3x3-hole rule.

Native rule rebuilds do not provide a hidden route under the budget: a truthful
float detector over the 18x18 candidate-anchor grid already requires 324
float32 scores (1,296 bytes) before suppression and 3x3 painting. The compact
historical 860–928 rule rebuilds use the same CenterCropPad opacity and are not
truthful-shape alternatives.

## Gate stopping

Fresh-5000 and external-validator-500 were intentionally not run. The skill
gate requires strict cost reduction before expensive admission tests, and
699,776 is not below 451. Running later gates cannot rescue this failure.

Evidence:

- `audit.json`
- `cse_truthful_build.json` / `exact_cse_truthful_build.json`
- `candidates/task162_exact_cse_truthful.onnx`
- `build_truthful_repair.py` / `audit_truthful.py`
- `winner_manifest.json`

No shared submission ZIP, aggregate CSV, baseline archive, or root model was
modified.
