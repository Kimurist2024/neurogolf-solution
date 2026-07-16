# 8012.15 cost167..500 restart — 10-parallel handoff

Authority: `submission_base_8012.15.zip` (`1dce2bc9bd65497daf59575512972309a3d71b87ee406a97c21fe2ee16010231`), LB **8012.15**.

## Outcome

No unique new admissible candidate remains in this lane. Two valid normal-POLICY90
candidates were rediscovered, but both are byte-identical duplicates of lane404:

| task | authority -> candidate | known | fresh (two seeds) | gain | disposition |
|---:|---:|---:|---:|---:|---|
| 161 | 190 -> 186 | 265/266 | 99.24%, 99.35% | +0.021277 | POLICY90 duplicate lane404 |
| 355 | 250 -> 249 | 264/267 | 98.71%, 98.60% | +0.004008 | POLICY90 duplicate lane404 / public-overfit risk |

Duplicate conditional gain already represented by lane404: **+0.025285**.
This lane contributes **+0.000000 unique gain** and must not cause the two models
to be merged twice.

## Decisive rejections

- task048 379->142: fresh 61.10%; reject.
- task143 212->148: lookup carrier, fresh 2/5000 and 3/5000; reject.
- task168 414->166: fresh 30.35%; reject.
- task185 279->185: catalog black/private-zero and fresh 1/500 per seed; reject.
- task384 180->179: runtime shape cloak; reject regardless of 99.62% known.
- task070/task134/task202/task343: latest explicit LB-black list; unconditionally excluded.

## Coverage

- 101 current tasks with cost167..500.
- 144 finite low-cost/generic variants per task; no finalist.
- Exact current-graph initializer/Einsum/Gather/lookup/ConvTranspose-oriented shaves; no finalist.
- Loose+ZIP history: 9,792 ONNX paths, 378 ZIPs, 2,419 unique task/hash pairs,
  1,339 theoretical strict-lower profiles. Ten workers reached the 1,000-result
  checkpoint; pathological residual calls exceeded the bounded wait and are
  fail-closed, never admitted. Prior isolated exhaustive 101..500 inventories
  were cross-checked for the residual candidate families.

Every admission requires >=90% independently in each seed/config and zero
runtime errors, nonfinite values, output-shape mismatches, small-positive values,
UB, or runtime-shape cloak. `GUARANTEED_SAFE` remains empty and separate from POLICY90.

No root artifact or `others/` file was modified.
