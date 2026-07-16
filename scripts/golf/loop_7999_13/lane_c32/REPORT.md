# Lane C32 — task224 / task240 exact-factor audit

Result: **no promotable winner**; projected gain `0.0`.

| Task | Cost | SHA-256 | Known disabled/default | External | Shape cloak | Conv UB |
|---:|---:|---|---|---|---:|---:|
| 224 | 162 | `02d6386ace32270c71ee2072328187a4c3a2a8355babd6b69fdc4a0e5b6bac79` | 266/266 each, errors 0 | 266/266 | false | 0 |
| 240 | 172 | `1ac586676b5ef226ead36bdaf92333f30b18a5ed53d9fdea1eb1036e3b692465` | 266/266 each, errors 0 | 266/266 | false | 0 |

Both incumbents pass full ONNX checking, strict shape inference/data propagation, truthful static/runtime shapes, finite embedded initializers, and canonical I/O. No Einsum was enlarged.

## Exact-factor exploration

- No byte-exact, signed, scalar, power-of-two, or axis-permuted full-initializer alias exists in either model.
- task224 `Csum` and `Cdiag` are diagonally equivalent. A global log-scale/sign gauge system was solved across all three Einsums, including allowable diagonal scales on `row_codes` and `col_codes`; the system is inconsistent. Therefore tying them cannot preserve all three contractions without adding parameters/operands.
- task224 `H0B`/`H1B` share one row and differ by a component permutation, but both remain independently required by the final parity branches. Precontracting the first two Einsums does not free either global initializer.
- task240 has one row shared across `U1..U4`, but no full tied factor or diagonal equivalence. Splitting that row would require reconstruction operands and would enlarge the existing giant Einsum.
- A task240 cost-170 `A3` absorption probe removed two operands. External validation returned 0/266, errors 0: two additional unpaired `B` uses prevent the proposed row gauge from being global. It is rejected.

History scan covered task224 578 files / 18 unique models and task240 591 files / 26 unique models. All nine below-baseline candidates fail `train[0]` in both ORT modes. No candidate reached the known gate, so fresh-5000 was not applicable.

Evidence: `audit.json`, `candidate_screen.json`, `exact_relations.json`, `history_inventory.json`, both external-validator JSON files, and `winner_manifest.json`.
