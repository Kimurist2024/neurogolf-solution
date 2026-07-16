# Additional normal-POLICY90 backlog census (lane 281)

## Outcome

**Clean candidates: 0.  Preliminary fresh screens: 0.**

The historical exact-only backlog contains no unadopted candidate that is
simultaneously strict-lower than immutable 8009.46, at least 90% correct on
known data, non-private, structurally clean, runtime-safe, margin-clean, and
truthful about its runtime shapes.

The only ten candidates reaching 90% known accuracy are all shape cloaks:
nine task091 models are 100% known-correct, and one task384 model is 265/266,
but their declared intermediate shapes materially understate the actual
runtime tensors.  They are not eligible for fresh testing.

No root submission, `all_scores.csv`, stage tree, or `others/71407` file was
modified.  Kimi was not used.

## Inventory coverage

The census reuses seven completed, disjoint historical rescreens.  Those
rescreens already combine the all400 archive, loose ONNX, and historical ZIP
sources, and record which candidates reached the exact-known rejection stage.

| measure | count |
|:---|---:|
| historical target tasks inventoried | **140** |
| historical candidate rows represented | 6,758 |
| tasks after active/explicit exclusions | 130 |
| tasks after private-risk exclusion too | **104** |
| 8009.46-strict-lower `known_reject` SHA before policy exclusions | 140 |
| eligible `known_reject` SHA after exclusions | **104** |
| tasks containing those 104 SHA | 21 |

The seven target lists do not overlap.  Every task row in `inventory.json`
contains its immutable 8009.46 authority SHA/cost, historical source rescreen,
old stage counts, exclusion reason, and strict-lower known-reject SHA list.

Authority is `submission_base_8009.46.zip`, SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`.
Costs come from the 400-task canonical 8009.46 census, not from the currently
mutable root submission.

Excluded tasks include the requested task007/task012/task071/task161 set, all
21 active `others/71407` tasks, and the conservative private-zero/unsound
monitor catalog.  The complete lists are serialized in `inventory.json`.

## Candidate funnel

Each of the 104 candidate binaries was resolved from its recorded loose path or
ZIP member and checked against the recorded SHA before use.  The current audit
then applied full/strict static structure, known execution in
`ORT_DISABLE_ALL/default × threads 1/4`, finite/margin/output-shape checks,
cross-configuration sign/raw stability, and intermediate runtime-shape tracing.

| disposition | SHA | tasks |
|:---|---:|:---|
| current structure policy rejection | 28 | 088, 355, 357, 382, 394 |
| known accuracy below 90% | 65 | 088, 091, 092, 123, 153, 160, 184, 193, 200, 218, 225, 239, 250, 297, 330, 345 |
| runtime-shape cloak after known ≥90% | 10 | 091, 384 |
| source binary no longer resolvable | 1 | 382 |
| **qualified for fresh** | **0** | - |

The 28 structural failures have nonstatic inferred node-output shapes.  No
lookup, giant-initializer/Einsum, Conv-family bias UB, private-risk, or runtime
error candidate was allowed through the clean funnel.

Among candidates that were structurally executable but below POLICY90, the
best per-task rates remained decisively short: task092 reached 83.3962%,
task330 78.9474%, task345 57.9545%, and task193 57.8947%.  The other task-best
rates were at most 15.7895%.

## The ten misleading near/exact-known leads

### task091

Nine historical SHAs are 100% known-correct and appear to cost 122--162 versus
the 8009.46 authority cost 184.  Runtime tracing exposes 8--14 declared/actual
shape contradictions per model.  Common examples include:

- declared `output=[1,1,1,1]`, actual `[1,10,30,30]`;
- declared full-grid intermediates `[1,1,1,1]`, actual
  `[1,10,30,30]` or `[1,1,30,30]`;
- declared row/column vectors `[1,1]`, actual `[1,30]`.

The traced intermediate footprint is 65,877--102,144 bytes, so the apparent
sub-184 costs are shape-cloak artifacts rather than truthful improvements.

### task384

SHA `d4f13184877f748a40c9a486ff471309c2dc6d086b872fa74ae2f4fa64b36234`
is 265/266 = 99.6241% in all four configurations and appears to cost 179 versus
authority 180.  Its `ych` and `hid` tensors are both declared `[1,1,1,1]` but
execute as `[1,1,30,30]`.  A single traced example exposes 4,657 intermediate
bytes, eliminating the claimed one-unit strict-lower result.  It is rejected
for shape cloak, independently of its one known semantic failure.

## Fresh-screen disposition

The requested fresh `1000 × 2` dual screen is conditional on a candidate
surviving known, structure, runtime, margin, and truthful-cost gates.  Survivor
count is zero, so no generator execution was started and no empty-result rate
is misrepresented as a fresh pass.  `candidates.json` records
`fresh_selected_count=0` and an empty `fresh_results` list.

## Artifacts and reproduction

- `inventory.json` — 140-task source/exclusion/authority inventory;
- `candidates.json` — all 104 SHA audits, known-four evidence, shape traces,
  classifications, and the empty fresh ledger;
- `audit_backlog.py` — deterministic reproduction script;
- `REPORT.md` — this summary.

From repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  scripts/golf/agent_policy90_backlog_281/audit_backlog.py
```
