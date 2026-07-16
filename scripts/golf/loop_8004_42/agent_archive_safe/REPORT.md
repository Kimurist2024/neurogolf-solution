# Archive-safe audit against the fixed 8004.42 rebase

## Outcome

One historical model reaches the user's relaxed `fresh >= 95%` gate, but it is
**not an exact/sound promotion**:

| task | current cost | candidate cost | projected gain | disposition |
|---:|---:|---:|---:|---|
| 205 | 1042 | 937 | `+0.10621394007489116` | `RELAXED95_CANDIDATE_ONLY` |

No ZIP was built or merged. The fixed 27 tasks were not modified. If task205
were applied alone on top of the projected fixed-rebase score
`8004.532615455686`, the purely local projection would be
`8004.638829395761`; this is not LB-verified.

## task205 candidate identity and lineage

- Candidate:
  `scripts/golf/loop_7999_13/lane_a23/candidates/task205_r02.onnx`
- Candidate SHA-256:
  `bbfa8f5b79d2e8345a39a41f327ac1c2c851f3c7f388dd595c72ef951e1b3050`
- Byte-identical source:
  `others/2/7805/task205_rebuilt_top2_cost937.onnx`
- The source and retained candidate hashes are identical. The historical
  inventory records no quarantine/private-zero source for this SHA.
- Current fixed-rebase member SHA-256:
  `8a6acdc20a366ccbd32cf761285cbb2f1cbcf7d3d2ef8ea71d0fb5a3ed6f1468`

The current fixed-rebase member is byte-identical to the task205 member from
the older 7999.13 audit, so its existing generator evidence is directly
comparable.

## Cost and complete-known gate

The current external validator reproduced:

- baseline: memory 1031 + params 11 = cost 1042;
- candidate: memory 911 + params 26 = cost 937;
- reduction: 105;
- score gain: `ln(1042/937) = +0.10621394007489116`;
- known candidate: 266/266, wrong 0, skipped 0, runtime errors 0;
- known baseline: 266/266, wrong 0, skipped 0, runtime errors 0.

The same historical audit ran all 266 known cases under both
`ORT_DISABLE_ALL` and default ORT: 266/266 with zero errors in each mode.

## Dual-runtime generator-fresh evidence

Independent generator seed `93023205` produced exactly 5000 valid cases with
no generation errors and no conversion skips.

| model | ORT mode | right | wrong | runtime errors | rate |
|---|---|---:|---:|---:|---:|
| current baseline | disable-all | 4928 | 72 | 0 | 98.56% |
| current baseline | default | 4928 | 72 | 0 | 98.56% |
| candidate | disable-all | 4904 | 96 | 0 | 98.08% |
| candidate | default | 4904 | 96 | 0 | 98.08% |

The candidate exceeds the authorized 95% threshold in both modes and has no
generator-domain runtime errors. It is nevertheless weaker than the current
baseline on this seed: candidate and baseline decoded outputs agree on only
4873/5000 cases, and the candidate has 24 more wrong cases. This is why it is
classified as a relaxed95 candidate rather than a sound/fixed improvement.

## External and asymmetric-error audit

The team validator re-ran the candidate directly against
`submission_8004.42_fixed_rebase_meta.zip` on 500 arbitrary one-hot grids:

- executable on both sides: 500/500;
- candidate-only or baseline-only execution failures: 0;
- skipped-both-failed: 0;
- threshold equal: 22/500; threshold mismatches: 478/500;
- raw equal: 5/500;
- external cost verdict: cheaper and known-complete (`ACCEPT_STRICT` in the
  validator's cost/known terminology, with random mismatch explicitly allowed).

There were no asymmetric runtime errors, but ORT emitted three output-size
warnings on these out-of-generator arbitrary inputs: the candidate returned
widths 26, 29, and 28 while declaring width 30. All 5000 generator-valid fresh
cases recorded `[1,10,30,30]` outputs and zero errors. The warnings still make
the artifact unsuitable for calling platform-independent or shape-truthful on
all possible one-hot inputs.

Evidence: `task205_external_current500.json`.

## Static structure, bias, and lookup gates

- ONNX full checker: pass.
- Strict shape inference with `data_prop=True`: pass.
- All inferred tensors have positive static dimensions.
- Standard ONNX domain only; no functions, sparse initializers, external
  initializers, nested graphs, or banned/Sequence ops.
- Known-input runtime trace: zero declared/actual intermediate-shape
  mismatches; actual and static cost both 937.
- Conv-family bias audit: zero findings. The graph has no `Conv`,
  `ConvTranspose`, or `QLinearConv` node (its `ConvInteger` has no bias input).
- No `TfIdfVectorizer`, giant initializer, or spatial/example lookup bank.
  `Hardmax` is used as an algorithmic selector, not as a stored example table.

## 13-input float contraction risk

The largest node is a float32 `Einsum` with 13 inputs:

```text
bcrw,ri,ri,ri,ri,ri,ri,aiw,diw,eiw,fiw,giw,hiw->cadi
```

Its inputs are the float32 input grid, six repeated float32
`row_rough_counts` operands, and six float32 `col_counts` operands. The graph
also contains 11-, 10-, and 9-input float32 contractions. The largest
contraction is below the lane's hard giant-Einsum cutoff of 15 inputs, but it
is still numerically high-arity: repeated count powers are reduced in an
implementation-selected contraction order, and later rank/selection logic can
amplify float32 rounding differences. Both tested ORT modes produced the same
4904/5000 correctness count, which is useful local evidence, not a proof that
another provider/build will choose an equivalent floating contraction order.

Therefore this task should be submitted only as an isolated relaxed95 A/B
candidate. It should not be added to the fixed safe set until it receives LB
confirmation. The smaller task205 cost-1038 rewrite is decoded-identical to the
current baseline on 5000/5000 generator cases, but its gain is only
`+0.003846158587478315`; it is a lower-risk fallback if preserving existing
task205 semantics is preferred.

## Explicit exclusions

- task202 remains excluded. Its cost-28 candidate reaches 4993/5000 (99.86%)
  with zero runtime errors, but the sole float32 `Einsum` has 21 inputs. It is a
  giant floating contraction and does not pass this lane's platform-risk gate.
- task153 and all catalogued black/private-zero/lookup/shape-cloak/short-bias
  candidates remain excluded.
- Historical task322/task372 short-bias models, task254/task267/task323 giant
  contractions, task268 lookup/cloak model, and task366 error-minimized-policy
  reject were not reconsidered.

## Supporting files

- `recent_structural_leads.json`: post-all400 structural rescan inventory;
- `task063_known_compare.json`, `task137_known_compare.json`,
  `task270_known_compare.json`: previously safe candidates that are now more
  expensive than the current members;
- `task205_external_current500.json`: current-base external validation and
  asymmetric-error evidence.

