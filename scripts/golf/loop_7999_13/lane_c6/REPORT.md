# C6 SOUND wave — exact 7999.13 baseline

One strict winner was found. **task185 improves cost 284 -> 279**, giving a
projected score gain of **+0.0177624563**. No root submission ZIP, score CSV,
merge artifact, or aggregate baseline was modified.

## Accepted: task185

The raw generator places one encoded 3x3 object on a regular line lattice with
spacing 2, 3, or 4. The output is the recovered 3x3 colour pattern. The exact
baseline's moment/TopK/TfIdf extraction is retained byte-for-byte where its
float32 token identity matters.

The candidate maps a selected colour channel `c` to the signed label `2*c-9`;
padding therefore represents background as `-9`. Ten two-feature affine
classifiers over `(label, label^2)` recover channels 0..9. This removes the
ten-element bias and four-element Reshape shape tensors while adding one safe
Unsqueeze, changing **216 memory + 68 params = 284** to
**219 memory + 60 params = 279**.

Evidence:

- dual known-gold paths: pass; positive-margin stability: pass, minimum 1.0;
- fixed fresh gate (seed 777185): **5000/5000**, wrong 0;
- independent fresh (seed 185799913), ORT_DISABLE_ALL: **5000/5000**, errors 0;
- the same independent 5000 cases with default ORT optimizations:
  **5000/5000**, errors 0;
- ONNX full checker and strict data-propagating shape inference: pass;
- canonical static output `[1,10,30,30]`; standard domains only; no functions,
  sparse initializers, nested graphs, banned/sequence ops, or Conv bias issues.

Candidate SHA256:
`b9cdf53f270874779b816741c0562222876bf8d33eb0e7a268702ff8a9b1c910`.
Detailed machine-readable evidence is in `task185_audit.json` and
`winner_manifest.json`.

## Rejected / unchanged

| task | exact cost | raw-generator result | decision |
|---:|---:|---:|---|
| 051 | 283 | 2992/3000 | unsound incumbent; no complete cheaper laser detector/renderer |
| 064 | 271 | 2991/3000 | unsound incumbent; full rectangle/dot alignment rebuild exceeds floor |
| 123 | 266 | 3000/3000 | all six shared CP components required; six drop probes scored 0/100 |
| 124 | 265 | default-ORT runtime error | reject hidden-shape contract; no error-prone shave accepted |
| 148 | 265 | 2982/3000 | unsound incumbent; full portal/flip renderer exceeds floor |
| 178 | 269 | 3000/3000 | already sound floor; sparse Conv weight fails ONNX full inference |

For task124, a direct default-ORT run reproduces the unsafe allocator mismatch
at MaxPool: `{1,1,1,1} != {1,10,1,1}`. Its apparent success under
ORT_DISABLE_ALL was not treated as sufficient evidence.

The unsuccessful task178 sparse probe is retained only as checker-failure
evidence. The task123 component probes are retained as rejected search output;
none appears in the accepted manifest.
