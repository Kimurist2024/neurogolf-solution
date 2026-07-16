# Lane C31 — task199 / task212

Result: **no promotable winner**; projected gain `0.0`.

| Task | Cost | SHA-256 | Known disabled/default | External | Truthful shape | Conv UB |
|---:|---:|---|---|---|---|---:|
| 199 | 261 | `d236c732d0df80270154b8ee593e17768dd54fc8dcec4aac93e752474651383e` | 266/266 each, errors 0 | 266/266 | yes | 0 |
| 212 | 240 | `e3f20fe069499de6c8ab36eadb10e69802ab61c4c87eb98d9843c7a87869ad42` | 265/265 each, errors 0 | 265/265 | yes | 0 |

Both incumbents pass full ONNX checking, strict shape inference/data propagation, static truthful runtime shapes, finite embedded initializers, and canonical I/O. Their existing large Einsums were not enlarged.

## Exploration

- Scanned task199 across 602 files / 27 unique models. All 12 below-baseline candidates (cost 222, 241, 243, or 254) fail `train[0]` in both ORT modes. Past independent checks also recorded 0% known/fresh for the latent-state, m-state, and rank-3-coordinate reductions.
- Scanned task212 across 570 files / 17 unique models. No model below cost 240 exists.
- Enumerated all directly contractible pairs of constant Einsum factors. No pair in either model reduces stored parameter count; the smallest exact contractions increase parameters.
- task199 incumbent independently passed fresh 5000/5000 in both ORT modes, errors 0, across all generated sizes 3–15.
- No candidate reached the known gate, so candidate fresh-5000 and candidate external validation were not applicable. The task212 baseline-only fresh control was stopped when immediate handoff was requested.

Evidence is recorded in `audit.json`, `candidate_screen.json`, `exact_contraction_inventory.json`, `fresh_baselines.json`, both external-validator JSON files, and `winner_manifest.json`.
