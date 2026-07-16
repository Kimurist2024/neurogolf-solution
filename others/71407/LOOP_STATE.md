# +50 loop state — baseline 8009.46

- Target: `8059.46` (`+50.0`)
- Staged candidates: 23
- Projected staged gain: `+0.7881256396521608`
- Projected score: `8010.248125639652`
- Remaining to target: `49.21187436034784`
- LB-counted gain from this directory: `0.0` (no root merge/submission)

Latest promoted residual:

- task355 `250 -> 249`, normal-POLICY90 gain
  `ln(250/249) = +0.004008021397538641`. Candidate SHA-256 is
  `7ca617858a19310a433010e6e50da46b4d562d76f3d0688665c8387bdf6f24d8`.
  Known is264/267; primary fresh is9872/10000 and9852/10000; independent
  fresh is9871/10000 and9860/10000. All four ORT configurations have
  errors/nonfinite/shape mismatch/small-positive/config differences0 and
  minimum positive margin1.0. This is a clean normal-POLICY90 admission, not
  an exact rule proof; task355 is public overfit-risk but not private-zero.
- task161 `190 -> 186`, margin-repaired normal-POLICY90 gain
  `ln(190/186) = +0.02127739844728488`. Candidate SHA-256 is
  `57487cce1b40cc7df6097cdf1e82e7bfa53b9bcb6f5be954329ea10d132ced81`.
  It keeps the clean cost186 graph and changes only the terminal `poly`
  initializer to exact float32 `source*8`, proving a positive uniform raw
  scale. Known is265/266; primary fresh is9925/10000 and9947/10000;
  independent fresh is9924/10000 and9935/10000. All four ORT configurations
  have errors/nonfinite/shape mismatch/small-positive/config differences0.
- task007 `70 -> 68`, normal-POLICY90 gain
  `ln(70/68) = +0.028987536873252187`. Candidate SHA-256 is
  `fa22f345634e3f059b0b2d334e6b9d85d60973d5cc2a6c92003b8f7cfc60486a`.
  It is a clean output-only 10-input Einsum with no lookup/cloak/giant graph.
  Known is260/266, primary fresh is9775/10000 and9726/10000, and independent
  fresh is9745/10000 and9752/10000. All four ORT configurations have identical
  signs, errors0, nonfinite0, shape mismatch0, and forbidden small positives0.
- task012 `710 -> 650`, normal-POLICY90 gain
  `ln(710/650) = +0.08829260714567821`. Candidate SHA-256 is
  `9aea31a6c01f7af21d893f6e5dde16dc947cdb17088686654f3f568845fbb947`.
  It is a truthful output-only group10 Conv with no lookup/cloak. Known is
  252/265, complete generator geometry is186/196, primary fresh is
  9478/10000 and9499/10000, and independent fresh is9502/10000 and9472/10000.
  All four ORT configurations have identical signs/raw output, errors0,
  nonfinite0, shape mismatch0, and minimum positive margin0.39035797.
- task205 `1042 -> 1038`, authority-relative gain
  `ln(1042/1038) = +0.003846158587478315`. Candidate SHA-256 is
  `43c963c46bda5b444fb830b5495b4d71fb9dcf958e108954cdb9ef1064d9f9a8`.
  ORT reduction-order inspection plus exact binary `[30,1]` row-mask support
  proves bitwise pass-through for all `2^30` masks. Known266×4, micrograph372,
  and exposed generator2000 have zero raw/error/nonfinite/shape differences.
- task158 `7578 -> 7498`, total authority gain
  `ln(7578/7498) = +0.010612994282688715`.
- Candidate SHA-256:
  `e7101699bfc022fa794e15d7f374a8febe3e2680b8388c67b9a81cdc9962ced0`.
- Complete 48-configuration generator-support proof and independent known/fresh
  four-runtime review passed with zero raw differences, errors, nonfinite
  values, or runtime shape mismatches.

Current exploration wave:

- tasks159/199/259/301 exact-regolf; previously unassigned 150--400 range
  targets.
- tasks153/161/200/316 exact-regolf; previously unassigned 150--400 range
  targets.
- tasks225/228/388/400 latest-authority exact-regolf; task228/388 were changed
  by the 8009.46 rebase and are rescanned from their current SHAs.

Latest rejected residuals:

- task209 `2085 -> 2083` is raw-equivalent but inherits 16 runtime-shape
  mismatches and two shape cloaks; the cheapest truthful control costs 2650.
- task366 `7985 -> 7984` repairs all observed OOB runtime errors and is
  raw-equivalent whenever the parent returns, but still has 98 runtime-shape
  mismatches; the truthful repair costs 9465.
- The promoted task158 cost-7498 payload was re-run through 25 fixed-point
  optimizer/fusion/cleanup profiles.  All 25 changed serialization, but none
  profiled below 7498.
- tasks218/394/397 produced no strict-lower result in 66 exact-cleanup profiles.
  Their authorities respectively have a fresh true-rule miss plus internal
  cloak, four runtime-shape contradictions, and default-ORT/lookup/cloak
  failures; truthful controls are all more expensive.
- tasks206/212/247/273 produced no exact mechanical shave.  task247 is sound
  but already at its exact floor; the other three carry false shapes, giant
  contractions, or lookup structure and cannot seed a safe descendant.
- task222's cost280 projection removal and all eight cost348 rank drops fail
  known data.  More fundamentally, its generator has positive-probability
  identical inputs with different planted outputs, so a deterministic
  all-support correctness guarantee is impossible; only authority-equivalent
  rewrites could qualify, and none was found.

Pinned authority and observed concurrent root state:

- immutable `submission_base_8009.46.zip` SHA-256:
  `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
- root `submission.zip` observed SHA-256:
  `d772399d4535176b95039690eca59808059add3c0ca2d42e2124f17c705ec2e6`
- root `all_scores.csv` observed SHA-256:
  `e58e81abdf5c0705ab8a7acb506173ee03f6383aa1c9a31adc1ab29898e4a630`
- The root divergence was first observed after independent task012 review;
  this loop did not create or overwrite it and preserves it as concurrent
  external state. All stage deltas remain measured against the pinned base.

The loop must not merge into the root archive or score ledger.  Every promoted
file belongs directly under `others/71407`; failed/private probes stay in a
quarantine directory.
