# 7999.13 strict optimization loop status

## Current audited aggregate (Wave 18 candidate)

- Exact leaderboard baseline: `submission_base_7999.13.zip`
- Baseline SHA-256: `a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`
- Metadata-preserving candidate ZIP: `submission_7999.13_wave18_candidate_meta.zip`
- Candidate ZIP SHA-256: `42db05f21fb3a3768a9491ccc5084601066b17af50e110f9015610416f0ccadb`
- Accepted tasks: 21 (18 strict, 3 exact-base-equivalent under the user-authorized 95% rule)
- Predicted cumulative gain: `+1.3429404589715328`
- Predicted score: `8000.472940458971`
- Requested completion threshold: `+20.0`
- Remaining predicted gain: `18.657059541028467`

| task | exact baseline cost | candidate cost | predicted gain | adoption evidence |
|---:|---:|---:|---:|---|
| 013 | 835 | 743 | +0.116735680133097 | known 267/267; fresh 5000/5000 under both disabled and default ORT; raw bitwise equal to Wave 16 on all 10000 comparisons after removing the shared all-one Einsum operand; actual/static cost agree; standard-domain full-check; external validator ACCEPT_STRICT |
| 051 | 283 | 279 | +0.014235115821872 | exact signed-power-of-two absorption; known 265/265 under both ORTs; candidate/base raw bitwise equal on 5000 fresh cases in each ORT mode; zero runtime errors; external random differential raw/threshold equal 500/500 and ACCEPT_STRICT; candidate retains the exact baseline's predictions and is accepted under the explicit >=95% rule |
| 063 | 26 | 24 | +0.080042707673536 | known 266/266 under both ORTs; independent fresh 5000/5000 under both ORTs; runtime errors 0; external validator ACCEPT_STRICT |
| 070 | 83 | 75 | +0.101352494260288 | exact algebraic inline; known 266/266 under both ORTs; fresh 4990/5000 disabled and 4996/5000 default; raw bitwise equal to the exact baseline on all 10000 cases; runtime errors 0; external validator ACCEPT_STRICT; accepted under explicit >=95% rule without changing predictions |
| 107 | 744 | 708 | +0.049596941139370 | known 266/266; fresh 5000/5000; independent raw differential 3000/3000 |
| 132 | 316 | 312 | +0.012739025777432 | exact two-axis gauge retie and initializer reuse; known 267/267 and fresh 5000/5000 under both disabled and default ORT; runtime errors 0; truthful strict shapes; external validator ACCEPT_STRICT; existing one-node Einsum remains at 47 inputs with no new operand or intermediate |
| 137 | 260 | 256 | +0.015504186535965 | known 266/266; independent fresh 5000/5000 under both disabled and default ORT; full verifier fresh 5000/5000; external strict known/differential clean; two in-Einsum factor reuses |
| 139 | 52 | 50 | +0.039220713153281 | known 265/265 under both ORTs; independent fresh 5000/5000 under both ORTs; runtime errors 0; external validator ACCEPT_STRICT |
| 158 | 7815 | 7627 | +0.024350380691339 | known 266/266 and fresh 5000/5000 under both ORTs; runtime errors 0; raw bitwise equal to Wave 13 on all 10000 comparisons after exact `coord2 = coord * coord` reuse; all 33 generator-reachable shapes covered; truthful static shapes; external validator ACCEPT_STRICT. Kept as a separately identifiable candidate because an older task158 processing-error incident has no recorded SHA |
| 168 | 432 | 416 | +0.037740327982846 | known 265/265; fresh 5000/5000 |
| 185 | 284 | 279 | +0.017762456339840 | dual known gold; two independent fresh seeds 5000/5000; default ORT 5000/5000 |
| 204 | 2560 | 2544 | +0.006269613013593 | known 268/268; fresh 3000/3000; independent differential 3000/3000 |
| 270 | 608 | 595 | +0.021613476420537 | known complete; two fresh seeds 5000/5000 each; exhaustive renderer audit |
| 275 | 432 | 428 | +0.009302392662313 | generator-exact gate-router absorption on the only reachable totals 18/32; known 266/266 under both ORTs; fresh 5000/5000 under both ORTs; runtime errors 0; truthful static/runtime shapes; external validator ACCEPT_STRICT; no new or enlarged Einsum |
| 279 | 404 | 397 | +0.017478597273960 | known 266/266; domain fresh 5000/5000 |
| 306 | 131 | 128 | +0.023167059281533 | exact signed-power-of-two factor retie; known 265/265 and fresh 5000/5000 under both disabled and default ORT; raw bitwise equal on all known and external random 500; errors 0; external validator ACCEPT_STRICT; existing 69-input Einsum not enlarged |
| 315 | 128 | 124 | +0.031748698314580 | known 266/266 and fresh 5000/5000 under both disabled and default ORT; errors 0; exhaustive all 3^9 generator grids 19683/19683; truthful one-node shapes; existing 43-input Einsum not enlarged |
| 324 | 442 | 439 | +0.006810469002527 | known 266/266; fresh 5000/5000; exact contraction proof |
| 333 | 449 | 447 | +0.004464293128686 | exact signed-factor absorption; known 265/265; fixed-seed fresh 5000/5000; runtime errors 0; minimum positive margin 0.5999981; external random differential raw/threshold equal 500/500; one fewer initializer and one fewer existing-Einsum operand |
| 344 | 401 | 197 | +0.710757698568582 | known 266/266; fresh 5000/5000 |
| 379 | 1955 | 1951 | +0.002048131796354 | exact tensor-mode plus initializer-slice reuse; known 266/266 and fresh 4995/5000 under both disabled and default ORT; raw bitwise equal to Wave 12 on all 10000 comparisons; runtime errors 0; external validator ACCEPT_STRICT; accepted under explicit >=95% rule without changing predictions |

The external validator accepts the Wave 9 additions (063/070/139), the
Wave 10 addition (379), the Wave 11 task158 candidate, the Wave 12
task051 exact rewrite, and the Wave 13 exact initializer-slice rewrites for
task013/task379, the Wave 14 task158 exact product reuse, and the Wave 15
task333 exact sign absorption, the Wave 16 task275 gate-router absorption,
and the Wave 17 task013/task306/task315 reductions, plus the Wave 18 task132
gauge-retie reduction.  The Wave 18 archive
audit reports 400 unique tasks, no missing/duplicate/oversize members,
preserved order/comment/member metadata, and zero Conv bias-length UB.
The root `submission.zip`, `best_score.json`, and `all_scores.csv` remain
unchanged.  The optimization loop is still active; this status is not a claim
that the requested `+20` target has been reached.

## Active next waves

- A34: task099/398 exact factor and initializer reuse
- B27: task055/159 exact factor and initializer reuse
- C33: task143/301 exact factor and initializer reuse

Every subsequent winner must remain cheaper, complete-known correct,
candidate-runtime-error-free, and free of the documented structural/UB
signatures. Fresh/domain-generator 100% remains the normal gate; an exception
is allowed only for an exact bitwise-equivalent rewrite that meets the user's
explicit >=95% threshold and introduces no new failure or runtime behavior.
