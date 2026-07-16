# Lane C35 — task192 archive re-audit at 8000.46

No candidate is safe to promote. Projected gain is **+0.0** and the aggregate
was not changed.

## Baseline identity and true rule

The new archive member is exactly SHA-256
`e7f9a11b93b611acfa4bba39e90e1ddf24223d50add4277fe9716f21f6ede10c`,
cost **1609**. It is the retained LB-white h7902 fallback, not the later
documented black h7904 SHA `58d28573...`.

The raw rule chooses the most frequent nonzero color A, breaking ties toward
the lower color. A nonzero center is painted A iff A occurs in both its
center-inclusive horizontal radius-1 window and its center-inclusive vertical
radius-1 window. The default generator's equivalent output is the union of
3x3-or-larger nonoverlapping rectangles; sparse distractor pixels have no
orthogonal neighbors.

## Archive candidates

All five candidates pass full ONNX checker, strict shape inference with data
propagation, truthful runtime shapes, standard-domain-only checks, and known
**265/265** with zero errors in both `ORT_DISABLE_ALL` and default ORT. They
have no lookup, fixture table, nested graph, sparse initializer, banned op,
function, or shape cloak. Their dynamic Conv bias is strictly inferred as
length 10 against 10 output channels for every model, so Conv UB count is zero.

| cost | SHA prefix | known, each ORT | independent fresh early-stop | decision |
|---:|---|---:|---:|---|
| 403 | `ac136f26` | 265/265 | 88/100 (88%) | reject |
| 493 | `81a61b5a` | 265/265 | 49/100 (49%) | reject |
| 509 | `e49c31e7` | 265/265 | 80/100 (80%) | reject |
| 561 | `209e4785` | 265/265 | 70/100 (70%) | reject |
| 589 | `7217623b` | 265/265 | 78/100 (78%) | reject |

Seed was 19293501 and all early-stop runs used `ORT_DISABLE_ALL`; every run had
zero runtime errors. The candidates were tested in ascending cost order. Each
failed the 95% threshold within the first 100 valid fresh cases, so no failed
candidate was advanced to a wasteful 5000-case completion or second fresh ORT
mode. Known coverage already passed in both ORT modes.

Structurally, each candidate ends in a single grouped Conv over an asymmetric
kernel. That is a fitted linear threshold approximation to a predicate
requiring a nonlinear horizontal-AND-vertical conjunction. The observed fresh
mismatches prove non-equivalence. Exact-SHA inventory traces each file to loose
models or archive members, but supplies no leaderboard-white evidence for any
of these five SHAs. Thus they remain ineligible even under the explicit rule
that a >=95% non-equivalent model needs proven LB-white lineage or a
generator-derived structural justification.

The moved external validator independently confirms the cheapest candidate's
known 265/265 and cost 403, but rejects it: only 24/500 random threshold outputs
match the baseline. This differential is corroborating structural evidence;
the fresh-generator failure is the decisive rejection.

## Evidence

- `candidate_audit.json`
- `conv_bias_contract.json`
- `lineage.json`
- `fresh_task192_r0{1..5}_disabled_100.json`
- `external_task192_r01.json`
- `rejected_manifest.json`
- `winner_manifest.json`
