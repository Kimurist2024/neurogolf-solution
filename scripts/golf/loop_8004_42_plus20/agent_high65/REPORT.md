# high65 exhaustive low-cost audit

## Decision

**SAFE WINNERS: 0** for `task276,305,309,312,337,373,053,087`.

The immutable `8005.16` models are already at actual official-like costs 5--10.
No SHA-distinct local model is strictly cheaper, so none can enter the required
known100-dual, structure, fresh, or private-guarantee gates. No candidate was
integrated and no submission ZIP or protected root file was intentionally changed.

## Immutable baseline

- Archive: `submission_base_8005.16.zip`
- SHA256: `73073208ec8b5b296f1fc13300a7e4df871135f0a9edd4a682ac7b48077aba00`

| task | memory | params | actual cost |
|---:|---:|---:|---:|
| 276 | 0 | 10 | 10 |
| 305 | 0 | 10 | 10 |
| 309 | 0 | 10 | 10 |
| 312 | 0 | 10 | 10 |
| 337 | 0 | 10 | 10 |
| 373 | 0 | 8 | 8 |
| 053 | 0 | 6 | 6 |
| 087 | 0 | 5 | 5 |

## Search coverage

The lane first audited the complete retained inventory. It then scanned every
loose file matching `taskNNN*.onnx` in the repository. Deduplication by
`(task, SHA256)` reduced 4,520 path aliases to 41 unique task/model pairs.

| task | path aliases | unique SHA | distinct nonbaseline | min nonbaseline cost | strict lower |
|---:|---:|---:|---:|---:|---:|
| 276 | 565 | 1 | 0 | -- | 0 |
| 305 | 566 | 7 | 6 | 10 (tie) | 0 |
| 309 | 566 | 3 | 2 | 10 (tie) | 0 |
| 312 | 565 | 8 | 7 | 10 (tie) | 0 |
| 337 | 565 | 3 | 2 | 10 (tie) | 0 |
| 373 | 564 | 10 | 9 | 30 | 0 |
| 053 | 565 | 4 | 3 | 6 (tie) | 0 |
| 087 | 564 | 5 | 4 | 368 | 0 |

The retained-inventory audit contained no leads for these tasks. The loose-file
audit found no candidate below its corresponding baseline. Equal-cost variants
do not improve score and are excluded by the strict-lower gate.

## Gate disposition

- Actual official-like trace cost: applied to every unique local SHA.
- Strict lower: 0 candidates.
- Known100 dual runtime: correctly not run because there is no strict-lower lead.
- Structure/data propagation/runtime-shape/domain/UB/lookup/cloak/giant audit:
  correctly not run after the strict-lower failure.
- Two-seed fresh testing and decoded true-rule private-zero proof: correctly not
  run because no candidate reached the pre-fresh gate.
- Bias-shortening/nonfinite checks: no candidate reached a shortening proposal.

## Evidence

- `history_lead_audit.json`: retained-inventory audit.
- `all_candidate_audit.json`: exhaustive loose-file SHA-deduplicated audit,
  including aliases, actual cost, and per-task coverage counts.

