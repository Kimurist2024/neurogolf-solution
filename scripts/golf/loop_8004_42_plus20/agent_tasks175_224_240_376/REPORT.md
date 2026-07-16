# Exact regolf audit — tasks 175 / 224 / 240 / 376

## Outcome

No authority-equivalent strict-lower model was found. `winner` is **null**.
Nothing was staged, merged, or written outside this lane. Root
`submission.zip`, `submission_base*`, `all_scores.csv`, and `others/71407`
were treated as immutable authority inputs.

The authority is `submission_base_8009.46.zip`, SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`.
It is byte-identical to the current `submission.zip` observed at audit start.

## Generator semantics and independent truth

- **task175 / 73251a56**: the input is a 21x21 deterministic quotient pattern
  with rectangular zero cutouts. For coordinates `A,B=2..22`, the true color
  is `1 + (A//B + B//A + input[0][0] - 3) % max(input)`. The transform restores
  every cutout from this formula.
- **task224 / 928ad970**: preserve the gray marker pixels and the inner hollow
  box, then complete the missing outer hollow rectangle in the inner box's
  non-gray color. The four gray markers determine the outer top/bottom/left/
  right bounds; the raw rule rotates recursively so the same construction is
  orientation-independent.
- **task240 / 9d9215db**: reflect the sparse 19x19 odd-lattice bitmap across
  both axes. When a diagonal bitmap cell has its paired right/below neighbor,
  fill the corresponding odd-spaced dotted square before/while applying the
  fourfold symmetry.
- **task376 / eb281b96**: if the input row list is `j`, the output row list is
  exactly `(j + j[1:-1]) * 2 + j[:1]`. Width is fixed at 17 and stretch is
  1..4, giving input heights 3..6 and output heights 9..21.

The independent `inputs/sakana-gcg-2025/raw/taskNNN.py` rules matched all
known examples: 266/266 for tasks175/224/240 and 39/39 for task376.

## Authority profiles and runtime audit

| task | authority SHA-256 | official memory | params | cost | known per ORT config | shape mismatches |
|---:|---|---:|---:|---:|---:|---:|
| 175 | `0979ba8969cdfd796f0c4e0c40c1ebf062d28093ab8866801bad9f504d537945` | 0 | 166 | 166 | 266/266 | 0 |
| 224 | `02d6386ace32270c71ee2072328187a4c3a2a8355babd6b69fdc4a0e5b6bac79` | 24 | 138 | 162 | 266/266 | 0 |
| 240 | `2aca909a256ffcafc876ff045b879c2c4b48e46724f14c7f59efe90c93032bda` | 0 | 160 | 160 | 266/266 | 0 |
| 376 | `d4983dcf71b72916ac9ac812f6d214702e3bff4b2fb6f1abcef26e779d02b4de` | 128 | 30 | 158 | 39/39 | 0 |

Each authority passed full ONNX checker, strict shape inference with
`data_prop=True`, canonical `[1,10,30,30]` output, positive/static inferred
node-output shapes, standard-domain, banned-op, nested-graph/function,
Conv-bias UB, finite-value, margin, and runtime-shape gates. Every inferred
node output was exposed and traced; unresolved outputs, hidden shape/type
omissions, declared/runtime contradictions, nonfinite values, runtime errors,
and `(0,0.25)` outputs were all zero.

Known data was run under four independent configurations:
`ORT_DISABLE_ALL` with 1/4 threads and default optimization with 1/4 threads.
All four configurations produced bit-identical raw authority outputs within
each task and only the truthful `[1,10,30,30]` runtime output shape.

## Exact-reduction search

Eleven fixed-point optimizer profiles per task (44 total) covered dead-end
removal, CSE, duplicate/unused initializers, idempotent and no-op removal,
Conv/Pad fusion, shape folding, Einsum-to-MatMul, Where rewrite, and Add
adjustment. None changed an authority graph, so none produced a strict-lower
candidate. Manual anatomy also found no dead node output, unused initializer,
byte-identical initializer alias, duplicate deterministic producer, removable
no-op, or hidden optional output.

The cost floors are tight in the current representation:

- task175 and task240 are single-node/free-output Einsums with zero counted
  intermediate memory; any win must delete live parameters.
- task224 counts only two truthful `[1,3]` row/column codes (24 bytes) plus 138
  parameters. Shrinking those codes or their three-row basis has already
  destroyed the rule.
- task376's 30-element packed Gather index and its scalar/scalar/30-element
  intermediates account exactly for 30 params + 128 bytes; the cleanup passes
  expose no redundant value.

Historical evidence was SHA-deduplicated only as evidence. The earlier full
repository scan covered 32 / 25 / 29 / 12 non-authority SHAs respectively:

- task175 latent prunes cost 142--146; the best is only 262/266 known.
- task224 cost156/158 leads are all 0/266 known.
- task240 cost154 leads reach at most 69/266 known.
- task376 has no retained numeric strict-lower lead.

No exact finalist survived, so expensive exact-fresh testing was stopped at
the mandatory known/cost gate.

## Isolated task175 POLICY90 evidence

Per the follow-up policy request, historical SHA
`40a9405880836a60f100e0072b476e4383c12c7ee053eb12ada1f049ee2e8d7c`
was audited separately. It costs 145 (gain `+0.135254045936` if a non-exact
policy admitted it), is checker/strict/truthful, and has no lookup,
CenterCropPad, banned op, runtime error, nonfinite value, margin violation, or
shape mismatch. It still retains an 18-input Einsum.

The candidate shrinks the shared latent `L` axis from 2 to 1 in `C0` and `G1`.
That removes `L=1`, which is not algebraically zero. Consequently it scores
262/266 known in every ORT configuration. The four failures are all original
fixed examples (`train[0..2]`, `test[0]`) with 20 / 12 / 8 / 20 threshold-cell
differences. Its threshold mask equals authority on 262 known cases, but its
raw output is bit-identical to authority on **0/266**.

Two complete, disjoint generator seeds were then tested:

| seed | cases | independent truth | candidate per config | executions | errors / nonfinite / margin / shape |
|---:|---:|---:|---:|---:|---:|
| 175224240376 | 3000 | 3000/3000 | 3000/3000 | 12000 | 0 / 0 / 0 / 0 |
| 376240224175 | 3000 | 3000/3000 | 3000/3000 | 12000 | 0 / 0 / 0 / 0 |

This is a strong POLICY90-only result, but it is **not** an exact winner: it
fails complete known correctness and has zero raw authority equality on both
known and fresh inputs. It remains quarantined evidence and was not copied,
staged, or merged.

## Evidence

- `result.json`: compact verdict, costs, SHAs, counts, errors, and shape totals
- `evidence/audit.json`: full checker, strict, profile, 4-config, optimizer,
  history, and POLICY90 fresh records
- `evidence/policy90_failure_details.json`: all four known failures and the
  nonzero latent-component diagnosis
- `audit_lane.py`, `policy90_failure_details.py`: reproducible audit scripts

