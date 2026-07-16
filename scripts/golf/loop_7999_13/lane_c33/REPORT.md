# Lane C33 — task143 / task301 exact-factor audit

Result: **no promotable winner**; projected gain `0.0`.

| Task | Cost | SHA-256 | Known disabled/default | External | Shape cloak | Conv UB |
|---:|---:|---|---|---|---:|---:|
| 143 | 212 | `c1f3481c23d501fdd1d6046fec6a6c14c4a4e777896cedb32fd522d64c37079c` | 266/266 each, errors 0 | 266/266 | false | 0 |
| 301 | 240 | `f613c7078ca9e622061826376e4c628cb59e19bbba1d8ac208fc26ea0f2f4a0d` | 266/266 each, errors 0 | 266/266 | false | 0 |

Both pass full checker, strict shape inference/data propagation, truthful static/runtime shapes, finite embedded initializers, and canonical I/O. No new or enlarged giant Einsum was emitted.

## Findings

- task143: 607 files / 40 unique models, with 14 below cost 212. Thirteen fail `train[0]` in both ORT modes. The sole known-complete cost-148 model uses `TfIdfVectorizer` and lookup-style memorization, explicitly prohibited by this lane, so it is ineligible without fresh admission.
- The four task143 learned `2x2x2` coefficient tensors have no exact, signed, scalar, power-of-two, permuted, or diagonal relation. Historical substitutions that reuse one tensor for another all fail immediately.
- task301: 576 files / 22 unique models and no below-baseline candidate. No same-dtype full initializer alias or diagonal tie exists. The apparent `-0.5 = 0.25 × -2` scalar relation crosses float16 and int64 (`axis_neg2`) and cannot be shared by ONNX consumers.
- Exact constant pair contraction enumeration yields no stored-parameter reduction. The smallest task301 contraction adds four parameters; task143 contractions add at least 32.

No semantically eligible lower-cost candidate reached the known gate, so fresh-5000 was not applicable.

Evidence: `audit.json`, `history_screen.json`, `exact_relations.json`, `exact_contractions.json`, external-validator JSON files, and `winner_manifest.json`.
