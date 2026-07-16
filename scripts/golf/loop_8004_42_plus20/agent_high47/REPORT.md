# High47 expansion — 8-task strict audit

## Outcome

All eight assigned files were independently audited against the immutable
`submission_base_8005.16.zip` (SHA-256
`73073208ec8b5b296f1fc13300a7e4df871135f0a9edd4a682ac7b48077aba00`).
**No safe strictly-cheaper candidate exists in the complete loose/current
history and SOUND-control families searched.** This lane contributes `+0.0`,
emits no winner, and does not build or modify a submission ZIP.

The current loose-tree rescan covered **4,872 observations and 312 unique
non-baseline models**. It retained 21 numeric lower leads. Every lead was
audited in its own process so invalid allocator behavior could not contaminate
another task.

| task | base cost | unique alternatives | numeric lower | terminal result |
|---:|---:|---:|---:|---|
| 044 | 1086 | 51 | 0 | no lower model; generator non-injective |
| 012 | 710 | 16 | 1 | cost 500 is 235/265 known; reject |
| 198 | 661 | 43 | 16 | all use 22–57-input giant Einsum; reject |
| 277 | 631 | 37 | 2 | cost 299/366 use TfIdf×4/10; reject |
| 117 | 606 | 71 | 1 | cost 278 is unscorable with 95 shape mismatches |
| 270 | 594 | 38 | 0 | no lower model; gated cost 595 is dominated |
| 019 | 536 | 28 | 1 | cost 535 terminates ORT with exit `-11` |
| 062 | 465 | 28 | 0 | no lower model |

The earlier all-400 archive independently covered 1,196 ZIPs, 448,568 ZIP
members, and 233,751 loose observations. The current rescan includes models
created after that archive and reproduces every prior lower frontier member.

## Generator references and fixed-baseline characterization

The generator for each task was read and its rule was represented by a
readable input-only reference where deterministic. References were checked on
the complete stored corpus and two independent 2,000-case fresh seeds.

| task | reference stored | reference fresh seeds | fixed base fresh, disable/default |
|---:|---:|---:|---:|
| 012 | 265/265 | 2000/2000, 2000/2000 | 2000/2000 in both modes on both seeds |
| 019 | 267/267 | 2000/2000, 2000/2000 | 2000/2000 in both modes on both seeds |
| 044 | 266/266 | 1998/2000, 2000/2000 | 1962/2000, 1963/2000 in both modes |
| 062 | 267/267 | 2000/2000, 2000/2000 | 1990/2000, 1989/2000 in both modes |
| 117 | 265/265 | 2000/2000, 2000/2000 | disable 2000/2000; default session reject |
| 198 | 266/266 | 2000/2000, 2000/2000 | 1878/2000, 1865/2000 in both modes |
| 270 | 266/266 | 2000/2000, 2000/2000 | 2000/2000 in both modes on both seeds |
| 277 | 266/266 | 2000/2000, 2000/2000 | 1921/2000, 1914/2000 in both modes |

Task044's two reference disagreements are consistent with the prior exhaustive
ambiguity proof: legal latent parameterizations can produce identical inputs
and different outputs. The fixed members are characterization only; their
historical LB status is preserved and their shape/approximation mechanisms are
not permission for a new candidate.

## Lower-frontier dispositions

- **task012:** the only cost-500 model passes full/strict/truthful-shape gates
  but is just 235/265 known in both ORT modes. The incumbent is one depthwise
  Conv with zero intermediate memory. The prior complete 1,712-alignment search
  proves no smaller exact one-node kernel geometry exists.
- **task198:** all 16 numeric lower models are giant contractions (maximum
  arity 22–57); 15 also use Hardmax. They are terminal policy rejects even
  though they reproduce the known corpus. The independent truthful direct
  control is known/fresh exact but costs 43,040.
- **task277:** both lower models reproduce known data, but contain four or ten
  `TfIdfVectorizer` nodes. Truthful component-mass/width controls cost
  3,831/5,341, and correcting the old exact graph's false declarations costs
  45,726; none beats 631.
- **task117:** the sole lower model produces correct known thresholds but is
  unscorable, declares the output as 1×1×1×1, and has 95 runtime shape
  mismatches. The truthful reflected-leg control costs 6,762.
- **task019:** omitting the unused variadic Split output appears to save one
  unit, passes ONNX checking, then causes an isolated ORT `SIGSEGV` (`-11`).
  It is an error candidate, not an optimization.
- **task044/task270/task062:** the current scan finds no numeric lower model.
  For task270, the previously exhaustively/fresh-gated cost-595 model is one
  unit more expensive than the fixed cost-594 member. For task062, equal-valued
  constants have incompatible schema-required ranks, so reuse is not a free
  shave.

## Safety and artifacts

No proposal reaches the prerequisite lower-cost plus safe-structure gate, so
candidate fresh promotion testing is not applicable. No lookup, shape-cloak,
giant-contraction, processing-error, or private-zero artifact is admitted.

Authoritative files are `history_inventory.json`,
`candidate_process_results.json`, the per-model JSON under `evidence/`,
`result.json`, and `winner_manifest.json`. Protected `submission.zip`,
`submission_base_8005.16.zip`, `best_score.json`, `all_scores.csv`, and `a.csv`
were not modified.
