# A42 — task196 strict exact-golf result

No candidate is eligible for promotion. The authoritative member remains cost
**1210** (`memory=1049`, `params=161`, score `17.901624361409212`) with SHA-256
`94e513b77cfce6a47ad2064a10c9c0c4ea545c46d1f3052b68d7e11f61fb3e7a`.
The source ZIP SHA-256 remains
`a2da30657f3798e861f369ac896f36722ff658ed3e468c4d55db9a04eefbccfc`.

## Decisive incumbent problem

The cost-1210 graph is already catalogued as a task196 private-zero/unsafe
incumbent. A new dual-ORT generator audit confirms the problem in both
`ORT_DISABLE_ALL` and optimized/default modes:

| model | known | fresh 5000 | runtime errors |
|---|---:|---:|---:|
| authority 1210 | 266/266 | 4789/5000 | 0 |
| historical 968 | 266/266 | 4490/5000 | 0 |

The authority misses 211 legal generator cases. The historical cost-968 graph
is worse, missing 510, and is raw-equal to the authority on only 4437/5000.
Thus a known-only pass cannot establish private correctness here.

## Truthful-shape gate

Both low-cost graphs are extensive `CenterCropPad` shape cloaks. Runtime tracing
on a real task196 case gives:

| model | false declarations | nominal cost | truthful cost |
|---|---:|---:|---:|
| authority | 56 | 1210 | 115233 |
| historical 968 | 59 | 968 | 151883 |

The authority's actual intermediate memory is 115072 bytes; the old lead's is
151747 bytes. Repairing every intermediate and output declaration gives zero
runtime-shape mismatches, passes full checker plus strict shape/data propagation,
and reproduces those costs. Full checker/strict inference also pass the original
cloaks, demonstrating that static checks alone do not prove truthful runtime
shapes.

## Exact local search

The authority contains no Identity, same-type Cast/CastLike, exact structural
CSE pair, duplicate or unused initializer, or dead node. The only remaining
one-element initializer opportunity was the bool type template:

| probe | dual known result | cost | decision |
|---|---|---:|---|
| `Cast(g_raw -> bool)` | 266/266 raw-equal, errors 0 | 1433 | worse |
| `Greater(g_raw,0)` | default 266/266; DISABLE_ALL 266/266 errors | 1433 | reject |

The exact Cast saves one parameter but exposes 224 bytes of real runtime output,
so cost rises by 223. The Greater variant additionally has an asymmetric runtime
allocator failure and is categorically invalid. Prior history already records
that replacing dynamic Shape inputs with constants exposes the large real
shapes and breaks the cloak, while removing the hidden clamp changes behavior.

## Historical and independent checks

The only archived actual-cost lead below 1210 is the cost-968 model. The
independent validator's arbitrary-grid random500 differential found 484/500
raw/threshold equality and 16 mismatches, with no one-sided failures. It was run
with `--allow-random-mismatch` only to retain diagnostics; any permissive label
in that output is not a promotion decision.

The best existing specification-derived fallback is the packed-bitset u16
model at cost **5573** (`memory=5368`, `params=205`). It is known-correct,
truthfully static, full-check clean, and has a recorded 3000/3000 fresh pass,
but is 4363 cost above the authority. Prior exact anchor and flood rebuilds are
also above the current bound; sampled CNN/hash reductions retain generator
counterexamples.

## Files and mutation scope

Primary evidence: `inventory.json`, `shape_audit.json`,
`truthful_costs.json`, `dual_known_fresh5000.json`,
`exact_probe_audit.json`, `external_summary.json`, and
`structural_audit.json`.

No root ZIP, CSV, score file, submission, or artifact was modified by A42. The
pre-existing `all_scores.csv` worktree modification was left untouched.
