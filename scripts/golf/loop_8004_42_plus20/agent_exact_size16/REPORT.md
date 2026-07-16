# Exact Size-fold lane report

## Outcome

- Immutable baseline: `submission_base_8004.50.zip`
- Baseline SHA-256: `63cb4c2abf794bb3cc0ceb531db907625c82638656e7d1ab29865d39b42a6cac`
- Primary targets: task177, task387
- Secondary checks: task069, task367
- Accepted: **0**
- Safe projected gain: **+0.0**
- Fresh audit: **not run**, because every candidate failed a mandatory pre-fresh gate
- Final verdict: **NO_SAFE_EXACT_CANDIDATE**

No root ZIP/CSV, protected artifact, or handcrafted model was modified.  The
ONNX files in `candidates/` are rejected experiment evidence only and must not
be integrated.

## Results

| task | exact rewrite | incumbent cost | unscorable arithmetic floor | theoretical gain | full checker / strict data-prop | dual ORT known | truthful runtime shapes | decision |
|---:|---|---:|---:|---:|---|---|---|---|
| 177 | `Size(spw)` -> scalar int64 `27` | 81 | 74 | +0.090384 | FAIL / FAIL | both modes fail to load | candidate cannot load | **REJECT** |
| 387 | `Size(input)` -> scalar int64 `9000` | 337 | 330 | +0.020990 | FAIL / FAIL | both modes fail to load | baseline output declares `1x1x1x1`, runtime is `1x10x30x30`; candidate cannot load | **REJECT** |
| 069 | `Size(codes_i8)` -> scalar int64 `10` | 259 | 252 | +0.027399 | FAIL / FAIL | DISABLE_ALL 264/264 exact; default mode fails to load | 6 baseline/candidate mismatches | **REJECT** |
| 367 | `Size(CBi)` -> scalar int64 `10` | 2197 | 2190 | +0.003191 | FAIL / FAIL | DISABLE_ALL 266/266 exact; default mode fails to load | 21 baseline/candidate mismatches | **REJECT** |

The “arithmetic floor” is only `old cost - 8-byte Size node output + 1 scalar
parameter = old cost - 7`.  It is not an official candidate cost: the official
full checker rejects every folded model before scoring.

## Why task177 and task387 cannot use the apparent -7 fold

The source element counts are indeed fixed.  That fact alone is insufficient
because these graphs use a scalar `Size` result as a `CenterCropPad` shape input
whose required vector length does not match its `axes` attribute.  While the
value is produced dynamically, ONNX shape inference cannot propagate it and the
invalid relationship stays hidden.  Replacing the value by a scalar initializer
makes the relationship statically visible:

- task177: `CenterCropPad` receives 1 shape element for 3 axes.  The same fold
  also reveals that `HannWindow` infers length 27 while its recorded
  `value_info` says length 1.  Both ORT modes reject the session at load time.
- task387: seven `CenterCropPad` nodes receive 1 shape element while recording
  `axes=[]`.  Both ORT modes reject the session at load time.  Independently,
  the incumbent graph's declared output shape is false (`1x1x1x1` declared vs
  `1x10x30x30` profiled), so this lineage also fails the no-cloak rule.

Splitting the scalar into spec-valid crop/window vectors would change the graph
semantics and parameter/memory accounting, and would require repairing the
known shape-cloak chain.  It is therefore not an exact micro-fold and was
stopped as requested.

## Secondary confirmation

task069 and task367 initially look less dangerous because the selected
`CenterCropPad` consumer has one axis.  The initializer nevertheless propagates
the real crop size and exposes false downstream annotations:

- task069: channel dimension 10 inferred where three recorded values claim 1;
- task367: channel dimension 10 inferred where `sel` claims 1.

They happen to remain bit-identical to their incumbents on all known cases under
`ORT_DISABLE_ALL`, but their default-optimization sessions fail because other
unfolded scalar `Size` values still feed multi-axis crop paths.  They fail the
strict, truthful-shape, and dual-runtime gates and are not eligible for fresh
testing.

## Evidence

- `result.json`: per-task SHA-256, checker errors, strict data-prop errors,
  profiled shape mismatches, complete known dual-mode counters, and verdicts.
- `build_manifest.json`: exact source tensors, fixed shapes, constant values,
  and candidate hashes.
- `build_exact_size.py`: reproducible narrow constant-fold builder.
- `audit_exact_size.py`: reproducible structural, profiler-shape, and dual ORT
  known audit.

