# high135 — task125 / task165 / task268 current-only SOUND audit

## Outcome

No candidate is admissible. The winner set is empty and projected gain is
`+0.0`. Root `submission.zip`, `all_scores.csv`, `others/`, and `artifacts/`
were not modified.

The immutable authority is LB **8009.46**:

- `submission.zip` SHA-256:
  `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
- `submission_base_8009.46.zip` has the same SHA-256.
- Current members are task125 `c30ac7a079a4` (cost 1045), task165
  `d6d40c11204c` (cost 587), and task268 `4c8ec91a517e` (cost 420).

These three exact current SHAs are LB-white fixed evidence. That exemption is
member-specific and was not transferred to any derived payload. All new
payloads were required to pass full/strict structure, truthful runtime shape,
known execution in DISABLE_ALL/default ORT at 1/4 threads, and then two-seed
fresh in both ORT modes.

## Authority audit

| task | official memory + params = cost | full/strict structural | runtime-shape mismatches | known four configs |
|---:|---:|:---:|---:|:---:|
| 125 | 916 + 129 = 1045 | pass | 32 | DISABLE_ALL pass; default session fail |
| 165 | 517 + 70 = 587 | pass | 89 | DISABLE_ALL pass; default session fail |
| 268 | 373 + 47 = 420 | pass | 39 | DISABLE_ALL pass; default session fail |

The baseline defects are retained as evidence, not treated as permission for a
descendant. Each default failure is a malformed `CenterCropPad` shape/axes
relationship. This is why a local algebraic rewrite can be mathematically exact
yet still change ORT allocation and competition cost.

## Exact and targeted exploration

The current graphs were scanned for dead nodes/initializers/value-info,
initializer aliases, no-ops, identical pure-node CSE, optional outputs,
constant folding, and normalization. Initializer element counts were audited
separately; dtype width alone does not change parameter cost.

### task125

- No dead node, initializer alias, unused initializer, duplicate pure node, or
  type-only `CastLike` initializer exists.
- Exact folding of the five fixed integer expressions
  `5+5=10`, `10+3=13`, `13+13=26`, `26+1=27`, and `26-1=25`
  exposes the actual shape tensors to strict inference.
- The result fails full/strict checking because one-element shape tensors feed
  `CenterCropPad` nodes with two or three axes; later crop sizes also expose
  `(26)/(13)/(10)` versus declared `(1)` conflicts.
- Prior SOUND generator evidence has a truthful cost-1167 rule model with
  3000/3000 fresh, already above the former cost-1050 member and therefore
  further above current cost 1045. It cannot be a strict-lower replacement.

Verdict: no strict-lower candidate reaches runtime validation.

### task165

The only exact reduction is CSE of two byte-identical
`CastLike(__sp_hid, seventeen_u8)` nodes.

| metric | current | CSE probe |
|---|---:|---:|
| memory | 517 | 477 |
| params | 70 | 70 |
| cost | 587 | 547 |
| competition correctness | true | **false** |

Although the pure-node substitution is algebraically exact, deleting the
second producer changes allocator reuse in the malformed graph:

- runtime trace: 88 declared/actual mismatches;
- DISABLE_ALL threads 1 and 4: **265/265 runtime errors**, first at `Slice`
  with `{1,9,30,30} != {1,10,30,30}`;
- default threads 1 and 4: session construction fails at `CenterCropPad`.

Verdict: rejected for runtime-shape witness and known4 runtime failure. Fresh
was not run after these terminal gates.

### task268

`_bool_like` is the only type-only initializer. Replacing
`CastLike(x, _bool_like)` by `Cast(x, to=BOOL)` and deleting the initializer is
all-input algebraically exact. It does not save competition cost: the explicit
`Cast` exposes the true 30x30 tensor to the profiler.

| metric | current | attribute probe |
|---|---:|---:|
| memory | 373 | 1272 |
| params | 47 | 46 |
| cost | 420 | **1318** |
| competition correctness | true | true |

The probe remains untruthful (39 shape mismatches), passes all 266 known cases
only under DISABLE_ALL, and fails default session construction in both thread
settings. The prior compliant true-rule control costs 18665, while the old
cost-327 private-lineage lead was independently only 2187/5000 and 2141/5000
fresh. Neither is a safe strict decrease from 420.

Verdict: rejected for cost, runtime-shape witness, and default ORT failure.
Fresh was not run after these terminal gates.

## Gate result

No probe cleared the mandatory pre-fresh gates, so running two-seed fresh would
not change any decision and was intentionally skipped. There is no candidate
to merge or promote.

Evidence:

- `authority_audit.json`: current SHA, official profile, full/strict,
  truthful-shape trace, and known4.
- `exact_scan.json`: current-only mechanical exact scan.
- `initializer_analysis.json`: initializer sharing/type-use audit.
- `targeted_probe_audit.json`: official and known4 rejection of the task165
  CSE and task268 attribute probe.
- `rejected_probes/`: the two derived payloads, retained only for reproducible
  rejection.
- `manifest.json`: machine-readable final disposition.
