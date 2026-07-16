# Latest-8005.16 low-cost extension audit (8 tasks)

## Outcome

No safe strictly cheaper candidate exists in this lane. The merge contract is
therefore empty: **0 adopted tasks, projected gain +0.0**. No submission ZIP,
score ledger, root artifact, or protected file was changed.

The authority is the exact `submission_base_8005.16.zip` payload (SHA-256
`73073208ec8b5b296f1fc13300a7e4df871135f0a9edd4a682ac7b48077aba00`).
Every one of the eight target members is byte-identical to the corresponding
member in the earlier exhaustive 7999.13 pool scan, so that scan's historical
coverage applies without a rebase ambiguity. This lane independently reran
checker/strict-data-propagation and both ORT modes on the latest bytes.

| task | cost | known DISABLE_ALL | known default | decisive result |
|---:|---:|---:|---:|---|
| 033 | 96 | 265/265 | 265/265 | incumbent uses 64-input Einsum; all 6 alternatives use prohibited 18-64-input Einsum |
| 282 | 96 | 265/265 | session creation fails | output is declared 1x10x1x1 but is actually 1x10x30x30; exact shave costs 27,994 truthfully |
| 084 | 92 | 175/175 | 175/175 | 32-input Einsum and false output shape; history has giant-Einsum or dominated graphs only |
| 362 | 92 | 267/267 | 267/267 | 28-input Einsum; sole alternative is still a prohibited 15-input Einsum |
| 381 | 92 | 265/265 | 265/265 | 33-input Einsum; sole alternative is cost-dominated |
| 001 | 90 | 268/268 | 268/268 | numeric cost-88 model is a prohibited 36-input Einsum; all other lower lineages are giant too |
| 352 | 90 | 266/266 | 266/266 | 85-input Einsum; three alternatives are 73-85-input Einsum and clean 3x3 Conv alone costs 910 |
| 283 | 89 | 265/265 | session creation fails | output is declared 1x10x1x1 but is actually 1x10x30x30; closest screened graph costs 91 |

All executable known runs above had zero runtime errors. Conv-family bias UB is
zero for every graph (none contains Conv/ConvTranspose).

## Exact/no-op screen

`task282` and `task283` each contain a boolean initializer used only to choose
the type of `CastLike`. I replaced those nodes with ordinary `Cast` and removed
the initializer. Both rewrites are raw-bit-identical to their incumbent over
the complete known set under `ORT_DISABLE_ALL`. They are still rejected:

- task282 candidate SHA
  `93d0cb4e9f3fad86e91162e67952489f0bf8bee6ccb235eecd08e6de74bf769f`
  profiles at memory 27,993 + params 1 = **27,994**;
- task283 candidate SHA
  `cc72e9906e91f5245705330446d771f5653ba2a9abd91d6a11f0d341a36d02fb`
  profiles at memory 30,684 + params 1 = **30,685**.

The jump is the expected effect of exposing the real runtime tensors hidden by
the incumbent metadata. Neither file is truthful-and-cheaper, and neither is a
merge candidate.

Initializer aliasing was also checked on all eight latest members. There are no
byte-identical same-shape initializers to share. The remaining compact lineages
encode nontrivial rules with giant tensor contractions: task001's Kronecker
mosaic, task033's panel propagation, task084's diagonal/bottom-border drawing,
task362's marker-driven line relocation, task381's inter-object fill, and
task352's 3x3 color-2 neighborhood. A clean standard-domain implementation of
each exceeds its current 89-96 cost floor.

## Gate disposition

Fresh two-seed testing was intentionally not run. It cannot rescue any model:
every sub-baseline attempt first fails actual cost, truthful runtime shape,
default-runtime, or the no-giant-Einsum structural prerequisite. There is no
private-risk candidate reaching the fresh gate, so the stricter private
known/fresh-100 policy is not invoked.

Machine-readable evidence:

- `audit_results.json`: latest payload hashes, full checker/strict data-prop,
  dual known counts, output-shape truthfulness, and historical stage counts;
- `exact_shave_audit.json`: complete-known raw equivalence and truthful actual
  profiles for the two exact rewrites;
- `result.json`: final eight-task decisions and empty adoption contract;
- `audit_lane.py` / `build_exact_candidates.py`: reproducible audit and builder.
