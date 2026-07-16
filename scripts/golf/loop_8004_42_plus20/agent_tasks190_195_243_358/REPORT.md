# Exact regolf audit: tasks 190, 195, 243, 358

## Outcome

No strict-lower authority-equivalent candidate is admissible. `winner` is
`null`; no submission, stage, score CSV, or shared artifact was changed.

The immutable authority is `submission_base_8009.46.zip`, SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`.
`submission.zip` had the same SHA when frozen and profiled.

| task | authority cost (memory + params) | decisive result |
|---:|---:|---|
| 190 | 153 (56 + 97) | The graph is a whole-input token plus two `TfIdfVectorizer` lookup tables and a 25-input `Einsum`; policy-banned. The five retained cost-141 variants score at best 56/266 known. |
| 195 | 150 (129 + 21) | No numeric lower history. The incumbent has five verified full-grid declared/runtime shape mismatches caused by three `CenterCropPad` nodes, so it cannot be a source for a truthful rewrite. |
| 243 | 147 (116 + 31) | No numeric lower history. Strict shape inference with data propagation fails (`Reshape` 30 vs declared 1); output is declared 1x1x1x1 but runs as 1x10x30x30. |
| 358 | 155 (12 + 143) | Truthful and exact, but already contains the proven R2/R3 polynomial fusion (161 to 155). No further dead, duplicate, identity, rank-1, or permuted initializer remains, and no lower history exists. |

## Generator rules and output shapes

- task190: a same-color 2x2 core and diagonal direction markers select which
  rays extend to the 10x10 boundary. Generator and logical output are 10x10;
  runtime output is truthfully embedded as 1x10x30x30 only in a clean build.
- task195: recover the translated 3x-enlarged 3x3 gray motif and emit its 9x9
  Kronecker self-product. Runtime output is 1x10x30x30.
- task243: four-neighbor flood all zero cells reachable from blue on square
  grids of size 12 through 18. Runtime output is 1x10x30x30.
- task358: infer the length-3/4 color cycle and cross center from the fragment,
  then extend the periodic center row and column over width 10 through 20 and
  height `width` or `width+1`. Runtime output is 1x10x30x30.

## History and cleanup search

The only retained models below these four authority costs are five task190
cost-141 payloads. Their known results are 51, 0, 0, 0, and 56 correct out of
266, with no runtime errors; all are rejected before fresh validation.

Dead/unused cleanup, identical or permuted initializer aliasing, no-op/CSE
cleanup, exact rank-1/dictionary factoring, and manual polynomial-factor review
produce no strict-lower safe rewrite. Task358 is itself the prior exact
`(x-2)(x+2)=x^2-4` fusion and has independent 5000/5000 dual-ORT evidence plus
100/100 raw random equality. Task190/195 prior true-rule audits are documented
in `scripts/golf/loop_7999_13/lane_c28/REPORT.md`; task243 retained-history and
task358 algebraic evidence are cited in the JSON files.

No large fresh run was started because no candidate passed the cheaper +
known-complete + strict-data-propagation + truthful-runtime-shape pre-gates.

## Evidence

- `audit/authority_profiles.json`: authority costs, SHA, node/initializer
  counts, checker/shape findings, runtime mismatches, and prior exact-SHA gates.
- `audit/mechanical_scan.json`: cleanup and exact-algebraic search disposition.
- `audit/history_scan.json`: all retained numeric-lower payloads and their
  known mismatches/errors.
- `result.json` and `winner_manifest.json`: machine-readable `winner: null`.
