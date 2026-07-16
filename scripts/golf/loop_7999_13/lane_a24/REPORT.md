# A24 — task198 / task277 strict private-black audit

## Result

- Exact base: `submission_base_7999.13.zip`
- Base SHA-256: `a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`
- Accepted replacement: none
- Score delta: **+0.000000**
- Root ZIP / CSV / ledger / `artifacts/handcrafted`: untouched

No lower-cost model met the A24 admission rule: for these private-black/ambiguous
tasks, a result must be a truthful generator-rule SOUND rebuild or an exact
executable-equivalent rewrite.  Archive known/fresh success alone is not
sufficient.

## Exact baselines

| Task | SHA-256 | Actual cost | Known disabled/default | Structural caveat |
|---|---|---:|---|---|
| 198 | `4e37cca3fc86cd4781a9b1f55c080f13962273e803c4c45d6dda99f74ba95283` | 661 | 266/266, errors 0 in both | terminal 24-input Einsum + Hardmax lookup form |
| 277 | `a6d659f65b084bdbcd7e2cc287a6fc0901a0351863ac64f7b47a5420e251f71d` | 731 | 266/266, errors 0 in both | `g`/`u` declared `[1,1,1,1]`, runtime `[1,10,30,30]` |

The baseline caveats do not authorize carrying the same mechanisms into a new
candidate.

## Complete cheaper archive frontier

- task198: all eight retained lower-static models use a 24–26 input giant
  `Einsum`; one also has explicit `private0` quarantine lineage.
- task277 r01 (static 299): 16-input giant `Einsum` plus four
  `TfIdfVectorizer` nodes.
- task277 r02 (static 366): ten `TfIdfVectorizer` nodes and risky-archive
  lineage.

All ten are terminal policy rejects.  Their empirical validation was
deliberately not used as evidence of soundness.

## SOUND / exact-rewrite controls

| Task | Control | Actual cost | Both ORT known | Truthful shapes | Verdict |
|---|---|---:|---|---|---|
| 277 | exact graph, only `g`/`u` declarations corrected | 45,726 | 266/266; errors 0 | yes | exact but not cheaper |
| 277 | component-mass generator rule | 3,831 | 266/266; errors 0 | yes | SOUND control, not cheaper |
| 277 | component-width generator rule | 5,341 | 266/266; errors 0 | yes | SOUND control, not cheaper |
| 277 | historical behavioral flood | 1,256 | disabled 266/266; default session error | no | reject |
| 198 | generator runtime-basis control | static high | not run after terminal screen | n/a | forbidden 18-input Einsum |

Correcting task277's two false declarations exposes the actual intermediate
memory and raises the score cost from 731 to 45,726.  This demonstrates that a
truthful exact-equivalent metadata rewrite cannot improve the current cost.
The two genuinely truthful generator-rule implementations are also above the
base cost.

## Fresh and external gates

There were zero admissible, strictly cheaper pre-fresh finalists.  Therefore
dual independent fresh5000 and external validator runs were not started; those
gates are only meaningful after a candidate survives provenance, structure,
truthful-shape, both-known, and actual-cost checks.  No processing-error model
was accepted or written outside this isolated lane.

Machine evidence:

- `model_manifest.json`: exact lineage, hashes, full retained frontier
- `audit_rows.json`: structural, shape-trace, both-ORT, and actual-cost rows
- `fresh_dual_5000.json`: pending count 0
- `external_validation_summary.json`: pending count 0, accepted errors 0
- `winner_manifest.json`: no winner, delta 0
