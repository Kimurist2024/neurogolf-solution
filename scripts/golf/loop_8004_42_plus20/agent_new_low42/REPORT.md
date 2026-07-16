# low42 target expansion — final report

Immutable baseline: `submission_base_8005.16.zip`  
SHA-256: `73073208ec8b5b296f1fc13300a7e4df871135f0a9edd4a682ac7b48077aba00`

**Completed 8/8 targets. Accepted models: 0. Projected gain: +0.0.**

| task | exact baseline cost | true rule / all known JSON pairs | baseline dual ORT | truthful runtime shapes | best lower search result | decision |
|---:|---:|---:|---:|---:|---|---|
| 339 | 53 | 266/266 | 266/266, 266/266 | yes | no below-53 graph; other floors 53/59/140 | reject: no strict decrease |
| 126 | 52 | 266/266 | 266/266, 266/266 | no, 19 mismatches | other floors 61/64/66/73/79/90/605 | reject: no strict decrease; shape cloak |
| 021 | 51 | 266/266 | 77/77, 77/77 | yes | only different harvested graph costs 177 | reject: no strict decrease |
| 171 | 50 | 54/54 | 54/54, **0/54 with 54 errors** | no, 8 mismatches | other floors 78/414 | reject: no strict decrease; default ORT failure |
| 346 | 50 | 267/267 | 267/267, 267/267 | no, 6 mismatches | other floors 92/109/111/1254 | reject: private-zero plus shape cloak |
| 227 | 49 | 267/267 | 267/267, 267/267 | no, 1 mismatch | other floors 72/100 | reject: no strict decrease; shape cloak |
| 318 | 49 | 267/267 | 267/267, 267/267 | no, 1 mismatch | other floors 72/100 | reject: no strict decrease; shape cloak |
| 332 | 49 | 267/267 | 267/267, 267/267 | no, 3 mismatches | cost-49 tie; other floor 80 | reject: tie only; shape cloak |

For task021, 189 generator cases exceed the competition's fixed 30x30 input
envelope and are skipped by `scoring.convert_to_numpy`; the generator rule itself
still reproduces all 266 stored pairs. The baseline passes both ORT modes on all
77 runnable cases.

## Search coverage

- The complete archive inventory covered 1,196 ZIPs, 448,568 ZIP members,
  233,751 loose files, and 13,591 unique non-baseline graphs. It retained no
  numeric below-baseline lead for any of these eight targets.
- The focused 1,134-graph harvest found only ties or more expensive graphs for
  these targets. All task-level rows and source paths are preserved in
  `history_audit.json`.
- A fresh narrow exact-rewrite pass tested initializer aliases, unreachable
  nodes/initializers, internal Identity/no-op Cast/Reshape, duplicate producers,
  and unused optional outputs. It found zero opportunities across 8/8 models.
- Every compact Sakana rule was decoded and reproduced every stored known pair;
  see `true_rule_audit.json`.
- Every incumbent was rerun over all runnable known pairs with both
  `ORT_DISABLE_ALL` and default ORT; see `known_baseline_dual.json`.

Because no strictly cheaper numeric candidate exists, candidate known/fresh
testing is correctly fail-closed and not run. In particular, task346's explicit
private-zero status cannot use the guarantee exception without a new truthful
strict-lower model and fresh100 dual evidence.

## Authoritative artifacts

- `baseline_audit.json` — exact member hashes, measured cost, ops, checker,
  strict data propagation, Conv-bias audit, and runtime-shape trace;
- `true_rule_audit.json` — decoded rules and complete stored-pair reproduction;
- `known_baseline_dual.json` — dual-ORT runnable-known evidence;
- `history_audit.json` — complete-archive, focused-harvest, and fresh exact-scan
  evidence;
- `scan_report.json` — zero-opportunity exact rewrite result;
- `result.json` — 8/8 fail-closed decisions;
- `winner_manifest.json` — empty authoritative promotion list.

No ZIP was built and no protected root file was modified.
