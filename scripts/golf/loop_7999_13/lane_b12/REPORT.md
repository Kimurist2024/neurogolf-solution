# Lane B12 — archive micro-winner safety audit

## Result

All five candidates are **REJECTED**. There are no winners, no cost adoption, and no score change.

The immutable base is `submission_base_7999.13.zip`, SHA-256
`a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`.
No root ZIP, CSV, score ledger, or canonical optimized model was modified.

| task | actual/static base | actual/static candidate | known, both ORT modes | early rejection |
|---:|---:|---:|---:|---|
| 254 | 76 | 42 | 265/265, errors 0 | one 33-input giant `Einsum` |
| 267 | 60 | 30 | 264/264, errors 0 | one 37-input giant `Einsum` |
| 322 | 20 | 19 | 266/266, errors 0 | `ConvTranspose` produces 10 channels but bias has 9 elements; nonfinite initializer/output |
| 323 | 106 | 104 | 172/172, errors 0 | final 56-input giant `Einsum`; extreme float32 magnitude |
| 372 | 13 | 12 | 266/266, errors 0 | `ConvTranspose` produces 10 channels but bias has 9 elements; nonfinite initializer/output |

The requested immediate-reject rule applies before fresh verification. Since every candidate failed that gate, dual fresh-5000 was intentionally not run. Passing a finite local sample cannot make a short Conv bias or a prohibited giant contraction safe.

## Source lineage and semantic rule

- task254 / `a61f2674`: source `others/2/7805/task254_improved_cost42.onnx`. The generator makes fixed 9x9 gray bars and keeps only the shortest bar in red and tallest in blue. The candidate replaces the base's three Einsums and two `[2]` intermediates with one 33-operand numerical contraction.
- task267 / `aabf363d`: source `others/2/7907/task267_improved.onnx`. The rule recolors a fixed 7x7 creature using the marker color at `(6,0)`. The candidate merges two 30-element coefficient vectors into one, but expands the single output contraction from 5 to 37 inputs.
- task322 / `d037b0a7`: sources `others/2/1203/71202/task322_rebuilt_cost19.onnx` and `scripts/golf/scratch_codex_plus10/pool_actual/task322.onnx`. The rule fills downward from one colored seed in each column of a fixed 3x3 grid. The only cost edit is truncating bias `B` from 10 elements to 9 while the operator still has 10 output channels.
- task323 / `d06dbe63`: source `submission_candidate_cost100_300.zip::task323.onnx`; reproduced by `scripts/golf/scratch_agents/task323/candidate_cost104.onnx`. The rule draws the bounded two-down/two-side gray staircase from a cyan seed on fixed 13x13. The candidate removes the two-element `V0`, reparameterizes `C`, and shrinks the final contraction from 62 inputs to 56; it remains a giant numerical Einsum.
- task372 / `e98196ab`: sources `others/2/1203/71202/task372_improved.onnx` and `scripts/golf/scratch_codex_plus10/pool_actual/task372.onnx`. The rule merges two colored 5-row halves around a gray separator. As in task322, the sole saving is a 10-to-9 bias truncation for a 10-output-channel `ConvTranspose`.

## Structural and numerical evidence

All five files pass `onnx.checker(..., full_check=True)`, strict shape inference with data propagation, standard-domain, static-dimension, nested-graph, sparse-initializer, and ordinary banned-op checks. Every declared/inferred runtime node-output shape exactly matches the observed shape, so none is a shape/value cloak.

That does not clear the decisive gates:

- task254, task267, and task323 use 33-, 37-, and 56-input `Einsum` contractions respectively. These are explicitly disallowed giant-Einsum structures. Task267 reaches absolute raw values of about `1.66e19`; task323 reaches `6.24e34`, making its sign result especially numerical and platform-sensitive.
- task322 and task372 have the exact known ConvTranspose bias defect: `bias=9`, `output_channels=10`. Their initializer `X` contains NaNs. Every one of the 266 known cases produces nonfinite raw output: task322 has 2,154,600 nonfinite cells across the corpus; task372 has 1,915,200. Thresholding happens to decode them correctly locally, but this is not a portable defined computation.

Runtime tensor evidence is complete:

- task254, task267, task322, and task372 candidates have no counted intermediates; their only node output is the correctly shaped `[1,10,30,30]` graph output.
- task323 has one counted `D` tensor, declared/inferred/actual `[1,2]`, float32, 8 bytes, followed by the correctly shaped graph output.
- Exact baseline task254 has `GH` and `GL`, each `[2]` float32 / 8 bytes; all other baseline/candidate intermediate inventories are recorded in `structural_audit.json`.

## Cost and decision

Official-like profiled costs equal the independently computed static costs for every base and candidate. The apparent combined gain would be `ln(76/42)+ln(60/30)+ln(20/19)+ln(106/104)+ln(13/12)`, but every term is ineligible. The adopted delta therefore remains exactly zero.

Evidence:

- `structural_audit.json`: frozen payload hashes, source lineage, actual/static costs, full graph/initializer diffs, all runtime node-output shapes, checker/shape/domain/bias/giant-Einsum findings, dual-ORT known results, and raw numeric audits.
- `manifest.json`: machine-readable per-task rejection decisions.
- `winner_manifest.json`: empty winner list.
