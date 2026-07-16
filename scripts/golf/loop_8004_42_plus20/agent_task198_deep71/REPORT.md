# task198 finite-generator exception audit

## Outcome

**SAFE 0.**  The complete lower, known-perfect lineage contains 16 unique
files.  Every file has at least one legal `task_83302e8f.generate(...)`
counterexample, so none can be guaranteed on the finite generator support.
No model is admitted, no ZIP is built, and the protected baseline/submission
files are untouched.

The current task198 member costs 661.  The lower files cost 554–655 and all
are 266/266 under both ORT modes on the retained corpus, but that corpus is not
a proof of the generator rule.

## Generator support and true rule

The source has exactly these top-level domains:

- `minisize m ∈ {3,4,5}`;
- `size s = r-m`, where `r ∈ {8,9,10}`;
- line color in `{1,2,5,6,7,8,9}`;
- grid side `n = s(m+1)-1`;
- a distinct subset of line pixels with size `k ∈ [s+m, sm]` is turned black.

For each black point on a horizontal border, both incident large cells become
yellow; the analogous rule holds for a vertical border.  All other large
cells become green.  The colored borders are retained and each black border
point becomes yellow.  The readable input-only reference is exact on all 266
stored cases and on both independent 1,000-case fresh seeds.

The parameterized support has

`7 * Σ(m=3..5,r=8..10) Σ(k=s+m..sm) C(L(m,s),k)`

inputs, where `L = n²-(sm)²`; this is
`64,652,967,950,963,564,767,518,564,435,358,347` inputs.  Exhaustive raw
enumeration is unnecessary for rejection: one exactly regenerable legal
counterexample disproves universal coverage.

## Candidate comparison

Fresh counts below are for each of four configurations: ORT
disabled/default optimization at threads 1 and 4.  All four configurations
produce the same right counts.  The “possible gain” is `ln(661/cost)` and is
not admitted.

| file | actual cost | fresh right / 2000 | rate | max Einsum arity | possible gain | first legal counter index |
|---:|---:|---:|---:|---:|---:|---:|
| r01 | 554 | 1792 | 89.60% | 24 | +0.176589 | 6 |
| r02 | 558 | 1792 | 89.60% | 26 | +0.169395 | 6 |
| r03 | 562 | 1616 | 80.80% | 24 | +0.162252 | 3 |
| r04 | 578 | 1813 | 90.65% | 24 | +0.134180 | 11 |
| r05 | 589 | 1619 | 80.95% | 24 | +0.115328 | 3 |
| r06 | 589 | 1810 | 90.50% | 24 | +0.115328 | 28 |
| r07 | 595 | 1619 | 80.95% | 24 | +0.105192 | 3 |
| r08 | 595 | 1906 | 95.30% | 24 | +0.105192 | 6 |
| r09 | 595 | 1742 | 87.10% | 24 | +0.105192 | 4 |
| r10 | 595 | 1780 | 89.00% | 24 | +0.105192 | 4 |
| r11 | 595 | 1684 | 84.20% | 24 | +0.105192 | 3 |
| r12 | 595 | 1929 | 96.45% | 24 | +0.105192 | 10 |
| r13 | 598 | 1501 | 75.05% | 24 | +0.100163 | 4 |
| r14 | 628 | 1961 | **98.05%** | 24 | +0.051214 | 89 |
| r15 | 645 | 1881 | 94.05% | 57 | +0.024504 | 14 |
| r16 | 655 | 1549 | 77.45% | 22 | +0.009119 | 5 |

The cheapest model is r01; the empirically strongest and most stable model is
r14.  Both are rejected.  r14 fails 39/2000 generated inputs, including the
exact legal parameter tuple stored in `counterexamples.json` (seed 47000199,
index 89: `m=5, s=4, color=1, k=9`).

## Equation and numerical audit

Fifteen files use:

1. a large variadic `Einsum` to score the three possible periods;
2. `Hardmax(axis=0)` to select a period;
3. a 22–26-input `Einsum` with dense learned coordinate/color factors to emit
   the output.

r15 replaces `Hardmax` with 30 repeated copies of `period_raw` in a 57-input
second `Einsum`.  Exact equations and arities for every file are recorded in
`counterexamples.json`.

On every captured counterexample the period selector's argmax equals
`minisize-3`.  The failure is therefore in the dense low-rank output
reconstruction, not in period identification.  All first mismatches are false
positive green logits; the same thresholded output occurs in disabled/default
ORT and with 1/4 threads on the exact counterexample.  Some factors have
absolute values from about `3e-37` to `1e10`, and r16 also produces values in
the unsafe `(0,0.25)` interval on fresh data.  The other 15 are near-margin
free on this run, but still wrong.

## Safety gates

- official-like actual cost: measured for all 16;
- known dual ORT: 266/266, runtime errors 0 for all 16;
- full checker / strict shape data propagation: pass for all 16;
- truthful runtime shapes: no declared/runtime mismatch for all 16;
- fresh dual seeds, disabled/default, threads 1/4: runtime errors 0;
- generator guarantee: **fail for all 16**;
- lookup-free semantic rule: **fail** — dense learned factor tables are not a
  proven generator-rule implementation;
- raw margin: r16 fails; irrelevant to the other files after correctness fail;
- final admission: **0/16**.

Machine evidence:

- `counterexamples.json`: exact formulas, costs, hashes, legal generator
  parameters, selector values, raw margins, and four-config reruns;
- `fresh_matrix.json`: all 16 × two seeds × disabled/default × threads 1/4;
- `result.json` and `winner_manifest.json`: safe0 decision.

