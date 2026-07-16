# Historical margin-repair scan 282

## Decision

`NO_ELIGIBLE_MARGIN_REPAIR_CANDIDATES`

Fifty tasks with an explicit historical `margin_stable=false` record were
inventoried. Forty-two came from the primary archive `quick_k5` screen and the
first eight unique sorted tasks from `lower_quick_k20` filled the requested
upper bound. The cohort contains 103 margin-false candidate records.

Only two records had already passed accuracy, runtime, and cost and were rejected
solely by the `(0,0.25)` margin gate. Neither is eligible under the requested
normal-policy constraints:

| Task | Candidate | Accuracy/cost evidence | Terminal scale | Exclusion |
|---:|---|---|---|---|
| 277 | `task277_r02_static366.onnx` | known exact, fresh 5/5, cost 366; current authority cost 631 | one-use terminal `Einsum` inputs `CF`, `Out`, `RF`, or `Sel` could uniformly scale output | private-zero/unsound monitor and ten `TfIdfVectorizer` lookup nodes |
| 328 | `task328_r02_static427.onnx` | known exact, fresh 5/5, cost 427; authority cost 558 | one-use terminal `Einsum` inputs `CoefAB`, `Rflip`, or `frow` could uniformly scale output | active in `others/71407` and a 75-input giant `Einsum` |

Both graphs pass full checking, strict data-propagating inference, canonical
static I/O, finite-initializer checks, and truthful runtime-shape tracing. Those
passes do not override the lookup/private/active/giant exclusions. The other 48
tasks did not have an accuracy-and-cost-pass margin-only record; their historical
screen failed accuracy or cost before margin could be the sole blocker.

## Cross-checks

- `submission_base_8009.46.zip` was pinned to SHA-256
  `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`.
- The live `others/71407/MANIFEST.json` contained exactly the requested 22 active
  tasks; task328 is among them.
- The separate POLICY90 backlog census contained 104 strict-lower historical
  candidates over 21 tasks. Its current classifications contain zero
  `REJECT_MARGIN` candidates, so it adds no scale-repair lead.
- The task161 precedent was checked: its safe repair scales the one-use terminal
  initializer `poly` by positive eight without changing cost. task161 is already
  repaired and active, so it was not reconsidered.

## Candidate and preliminary-audit status

No ONNX candidate was emitted. Consequently the requested known plus fresh
`1000 × 2 × 4` preliminary audit was not run: it is conditional on an eligible
candidate, and both possible scale repairs fail the policy gate before runtime
auditing. The empty status is recorded in `candidates/README.md` and
`scan.json` as `SKIPPED_NO_ELIGIBLE_CANDIDATE`.

## Reproduction

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/golf/agent_margin_repair_scan_282/scan.py
```

The script pins the authority, snapshots all evidence-source hashes, confirms
the active-22 manifest, recomputes current profiles for both margin-only leads,
and fails closed if an eligible repair source appears without a corresponding
audit. It writes only this lane. Root, other lanes, and `others/71407` were not
modified; Kimi was not used.
