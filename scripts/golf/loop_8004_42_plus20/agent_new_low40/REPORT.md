# Latest-8005.16 target expansion audit (8 tasks)

## Outcome

The eight requested members were independently audited against the immutable
`submission_base_8005.16.zip` (SHA-256
`73073208ec8b5b296f1fc13300a7e4df871135f0a9edd4a682ac7b48077aba00`).
**No safe strictly-cheaper candidate exists in the searched archive, exact,
Einsum-factor, or true-rule families.** Completed: **8/8**; accepted: **0**;
projected gain: **+0.0**. No ZIP or protected file was changed.

All eight latest members are byte-identical to both the 7999.13 and 8004.50
baselines. Therefore the earlier all-task archive inventory applies without a
rebase ambiguity. That inventory covered 1,196 ZIPs, 448,568 ZIP members,
233,751 loose observations, and 13,591 unique non-baseline graphs. It retained
**zero below-baseline model for every one of these eight tasks**.

## Per-task decision

| task | actual cost | DISABLE_ALL known | default known | decisive result |
|---:|---:|---:|---:|---|
| 022 | 63 | 266/266 | session fails | output declaration false and 10 runtime-shape mismatches; no sub-63 history |
| 181 | 63 | 266/266 | 266/266 | output declaration false and 5 runtime-shape mismatches; no sub-63 history |
| 104 | 61 | 7/7 | 7/7 | output truthful, but 2 hidden shape mismatches; four focused alternatives do not beat 61 safely |
| 294 | 61 | 265/265 | session fails | 16 hidden shape mismatches; 44 focused alternatives and exact rescan produce no sub-61 winner |
| 128 | 60 | 266/266 | 266/266 | truthful one-node 10-input Einsum already at 60 params / zero intermediate memory; no exact ≤59 rewrite |
| 152 | 60 | 267/267 | 267/267 | truthful but prohibited 19-input giant Einsum; only alternate is a 21-input giant Einsum |
| 203 | 60 | 267/267 | 267/267 | truthful one-node 12-input Einsum already at 60 params / zero intermediate memory; no exact ≤59 rewrite |
| 236 | 60 | 267/267 | 267/267 | output truthful, but 2 hidden shape mismatches in the quantized-convolution lineage; no sub-60 history |

The base models' known correctness is evidence about the already LB-white
incumbents only. It does not authorize inheriting their shape cloak or giant
contraction in a new file.

## Exact lead disposition

The local scanner found no Identity, unused initializer, duplicate initializer,
or duplicate deterministic producer in any target. The apparent task294 lead
was checked explicitly: its three `ConstantOfShape(l)` nodes emit different
shape constants, **31, 29, and 30**. They are not duplicate computations and
cannot be merged. No task294 candidate was emitted.

Two independent all-400 exact scans agree:

- the checker/strict/dead-code/initializer scan produced no opportunity or
  candidate for these eight tasks;
- the initializer alias, outer-product fusion, sign-absorption, and exact
  Einsum scan produced no target hit.

Thus no model reaches the fresh-test gate. Fresh dual-seed testing cannot make
a non-existent, non-cheaper, shape-cloaked, or giant candidate admissible, so
it was intentionally not run.

## True-rule audit

The authoritative generators were read directly. The tasks require,
respectively: assembling four partial 3x3 copies (022); mirroring a cyan sprite
around a yellow marker (181); quadrant-keyed 9x9 block drawing (104); painting
rectangle interiors under four rotations (294); lifting three rectangles by
their own heights (128); two-axis mirroring (152); reversing concentric ring
colors (203); and XOR of two 4x4 bitmaps (236).

These rules confirm that none of the suspicious low-cost graph fragments is a
trivial no-op. The exact generator paths, SHA-256 hashes, and compiled rule
summaries are recorded in `true_rule_audit.json`.

## Evidence

- `baseline_audit.json`: actual costs, full checker/strict results, dual known
  runs, Conv-bias audit, output truthfulness, and all-intermediate runtime trace;
- `history_audit.json`: latest/old byte identity, focused harvest, 1,196-ZIP
  all-task inventory, and both all-400 exact scans;
- `exact_candidate_scan.json`: reproducible local exact rewrite search;
- `true_rule_audit.json`: direct generator audit;
- `result.json`: final decisions and empty adoption contract;
- `winner_manifest.json`: authoritative empty promotion list.
