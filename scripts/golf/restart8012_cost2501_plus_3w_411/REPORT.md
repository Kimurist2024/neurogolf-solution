# cost >2500 / 3-worker POLICY90 scan (8012.15 authority)

## Outcome

No candidate is admissible. Projected gain is **+0.000000** and the
conditional LB remains **8012.150000**. No candidate ONNX or replacement ZIP
was emitted.

The immutable authority is `submission_base_8012.15.zip`, SHA-256
`1dce2bc9bd65497daf59575512972309a3d71b87ee406a97c21fe2ee16010231`.
The post-run root `submission.zip` has the same SHA. `all_scores.csv` remained
at SHA-256 `3f9914a0db88302f9e0424d604f9c0e300dc75115570625d296e21b7fcfaf731`.
This lane did not edit either protected file or `others/`.

## Scope and parallel scan

There are 17 non-score25 tasks above cost2500. Thirteen were excluded because
they occur in `docs/golf/private_zero_tasks.md`, have explicit black history,
or are in the latest hard blacklist `{070,134,202,343}`. The four eligible
members were:

| task | authority cost | result |
|---:|---:|---|
| 076 | 2550 | no strict-lower truthful candidate |
| 080 | 3050 | no strict-lower truthful candidate |
| 118 | 3665 | all executable histories wrong or more expensive |
| 349 | 3532 | no strict-lower truthful candidate |

Three worker processes completed with PIDs `81412`, `81413`, and `81414`.
The discovery pass read 182 ONNX-file references, 1,486 ZIP member references,
and 20 exact-optimizer variants. After task/SHA deduplication and the initial
parameter/declared lower bound, 92 unique strict-lower leads remained.

## Fail-closed gates

Before execution, every lead had to pass canonical I/O, full ONNX checker,
strict shape inference with data propagation, fully static positive shapes for
every node output, and a truthful inferred scorer-cost floor below its
authority. This rejected 83 leads:

| task | pre-execution rejects | main evidence |
|---:|---:|---|
| 076 | 30 | 20 noncanonical output cloaks; 26 nonstatic; healthy historical graph floor 42654 |
| 080 | 16 | 15 nonstatic descendants; no strict-lower healthy graph |
| 118 | 4 | healthy observable-rule floors 9142/9144/9316, all above 3665 |
| 349 | 33 | 32 nonstatic descendants; healthy historical graph floor 21722 |

The remaining nine models were all task118 histories. Six failed the first
known-case POLICY90 screen. Three were known-exact but their actual official
costs were 3911, 3914, and 3915, above the authority cost3665. Consequently no
model reached fresh testing; `fresh_audited=0` is a deliberate fail-closed
result, not missing evidence.

The mandatory admission rule was accuracy >=90% in every known/fresh
configuration and seed, with zero runtime errors, nonfinite values, output
shape mismatches, `(0,0.25)` positives, Conv-family bias UB, lookup/cloak, or
other structural violations. No candidate reached all gates.

## Evidence

- `inventory.json`: authority pin, scope/exclusions, all source counts, 83
  pre-execution rejection records, and worker assignment.
- `worker_0_evidence.json`, `worker_1_evidence.json`,
  `worker_2_evidence.json`: per-worker screens and pinned authority details.
- `evidence.json`: merged task results and final zero-admission summary.
- `MANIFEST.json`: empty promotion manifest.
- `scan.py`: reproducible three-worker scanner and empty-candidate short circuit.

Final decision: **NO_ADMISSION / +0.000000**.
