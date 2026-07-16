# restart8012 pending 3-worker audit

Authority: `submission_base_8012.15.zip` (`1dce2bc9bd65497daf59575512972309a3d71b87ee406a97c21fe2ee16010231`)

Workers: 3 spawned processes, PIDs [54237, 54238, 54239]

| task | authority | candidate | gain | classification | min accuracy |
|---:|---:|---:|---:|---|---:|
| 161 | 190 | 186 | +0.021277 | POLICY95_NONEXACT | 99.3500% |
| 175 | 166 | 145 | +0.135254 | POLICY95_NONEXACT | 98.4962% |
| 355 | 250 | 249 | +0.004008 | PUBLIC_OVERFIT_RISK_POLICY95 | 98.5000% |

Conditional total gain: **+0.160539**
Conditional projected LB: **8012.310539**

`GUARANTEED_SAFE` and `POLICY95` are intentionally separate. A POLICY95 pass
permits known/fresh mismatches up to 5% and is not an LB guarantee. task355
also retains the documented public-overfit-risk tag. No candidate is listed in
the maintained private-zero/unsound operational set.

The root authority and `others/` were not modified.
