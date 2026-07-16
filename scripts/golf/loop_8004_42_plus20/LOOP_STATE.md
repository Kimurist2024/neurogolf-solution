# 8009.46 -> +50 continuous loop

## Objective

- Active LB-verified immutable baseline: `8009.46`
- Active target verified score: `8059.46`
- Required gain from the latest baseline: `+50.0`
- The earlier `8004.42 -> 8024.42` +20 target remains an intermediate
  historical milestone; its fixed improvements and provenance are preserved.
- Never revert any LB-fixed task payload; in particular preserve all 37 new
  71405 white payloads and all24 new8009.46 white payloads on top of the
  earlier fixed improvements.

## Current LB-verified baseline

- ZIP: `submission_base_8009.46.zip`
- SHA-256: `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
- MD5: `2dc6d412ddd8bd3102f42775155e4a38`
- LB-verified score: `8009.46`
- Current verified gain from the requested 8004.42 origin: `+5.04`
- Current verified gain from the active8009.46 origin: `+0.0`
- Staged exact/POLICY90 projected gain in `others/71407`: `+0.784117618255`
- Remaining to active8059.46 target before staged LB confirmation: `50.0`
- Projected remaining after all current staged candidates: `49.215882381745`
- Archive integrity: PASS; Conv-family bias-length UB: 0

### Active +50 memory census 119

- All400 current8009.46 payloads have canonical aggregate cost192,234,
  comprising memory155,413 and parameters36,821.
- The current target list contains94 tasks with memory>=300 and theoretical
  half-memory gain>=0.15. Their summed half-memory opportunity is
  `+53.7045885244`; this is a prioritization upper opportunity, not counted
  progress.
- Evidence: `root_mem_census_119/REPORT.md`, `canonical_costs.json`, and
  `mem_targets_8009_46.json`.

### 8009.46 exact re-golf and residual archive wave

- The24 newly LB-white payloads were split into two exact lanes. Lane A
  generated18 transformations across tasks029/031/036/075/079/091/092/124/
  137/153/159/169; all were rejected by strict structure or official runtime,
  and task153 had no valid reduction. Lane B found no safe winner across
  tasks178/228/234/264/325/344/357/387/388/392/397/398. Its only nominal
  reduction, task264 344->343, retained authority runtime failure and44
  truthful-shape mismatches.
- A separate scan deduplicated110 task/file candidates from `others/71406`
  including71409. The only strict-lower survivor was task382 820->813, but it
  had20 runtime-shape mismatches, failed default ORT, and lacked an all-input
  proof; it was rejected as shape-cloak risk.
- Safe adoptees: 0. Evidence: `agent_8009_exact_A_115/REPORT.md`,
  `agent_8009_exact_B_116/REPORT.md`, and
  `agent_71406_residual_117/REPORT.md`.

### Memory-target waves 120--125

- task145: exact vertical-slope transpose alias removes40 parameters but adds80
  scored bytes; 5129->5169, rejected.
- task364: a domain-exact Boolean compilation reduced468->300 nodes but exposed
  real tensor shapes and cost685->115431. float8 is checker-unsupported, and
  no two-op `CenterCropPad` shift equals the incumbent three-op shifts.
- task009/task076: no strict-lower safe winner. task009 is already exact and
  mechanically irreducible; task076 has a reproduced non-injective generator
  witness plus default-ORT failure.
- task173: mathematically exact TopK score narrowing to 8/16-bit integer types
  is unusable because ORT has no TopK(11) CPU kernel for those dtypes in either
  mode. No candidate was admitted.
- Safe adoptees: 0. Evidence: `root_mem145_120/REPORT.md`,
  `root_mem364_122/REPORT.md`, `agent_high009_076_121/REPORT.md`, and
  `root_mem173_125/REPORT.md`.

### task205 all-support ReduceSum/scalar fusion 247/248

- Authority SHA
  `8a6acdc20a366ccbd32cf761285cbb2f1cbcf7d3d2ef8ea71d0fb5a3ed6f1468`
  costs1042. `ReduceSum(row_mask[30,1])` followed by scalar `Mul` is replaced
  with `Einsum(row_mask, scalar, equation="ri,->")`, removing the four-byte
  scalar intermediate. Candidate memory/params/cost is1027/11/1038, giving
  projected gain `+0.003846158587`.
- Candidate SHA
  `43c963c46bda5b444fb830b5495b4d71fb9dcf958e108954cdb9ef1064d9f9a8`.
  ORT's CPU implementation reduces the first-only r/i labels before a K=1
  pairwise MatMul. Since `row_mask` is exact binary, every reduction result
  0..30 is exactly representable; both paths then perform the same one float32
  product. This proves bitwise equality for all `2^30` masks.
- Micrograph93 masks x4 configs and exposed generator2000 have zero
  row_mask/colq/final raw differences. Retained exact-SHA and independent
  sibling audits cover known266 in disabled/default x threads1/4, with
  error/nonfinite/shape mismatch/UB all zero.
- Promoted inside `others/71407/task205.onnx`; root submission and score ledgers
  remain unchanged. Evidence: `root_task205_cost1038_proof_247/REPORT.md` and
  `scripts/golf/root_reduce_scalar_fusion_scan_248/REPORT.md`.

### Continuous exact/rebuild waves 240--259

- All400 exact constant-Einsum absorption (32,946 actions), linear-kernel
  factoring, integer carriers, domain identities, and loose-history rescreen
  produced no additional truthful strict-lower survivor.
- task023's best two-QLinearConv spatial morphology is cost1621 and only
  known253/266; a broader48-layout search reached91.733% fresh but failed the
  complete-known prerequisite. task012's exhaustive7x8/7x9 retrain reaches
  only235/265 known and89.796% complete generator support, below POLICY90.
- task138's truthful normalization costs5568; even impossible free mask and
  terminal deletion bounds remain3428, above authority2705. task367 was
  independently confirmed as the already-closed shape-cloak lineage.
- Boolean neutral, unsigned-comparison, Split/Concat, Einsum-unit-factor, and
  ScatterElements full-overwrite scans found no new lower candidate. Identity
  Gather and arithmetic Gather-to-Slice scans found nominal lower probes only
  in task090/191/285; each fails truthful runtime/default-ORT gates.
- Safe adoptees from these waves other than task205 cost1038: **0**. Evidence:
  `root_task023_spatial_morphology_246/`,
  `scripts/golf/root_task012_sub710_retrain_250/`,
  `scripts/golf/root_task138_truthful_floor_249/`,
  `scripts/golf/root_identity_gather_scan_254/`, and
  `scripts/golf/root_gather_slice_scan_257/`.

### task012 8x8 normal-POLICY90 promotion 272/273

- A previously unsearched balanced8x8 depthwise-Conv layout reaches the proven
  case-level MILP optimum186/196 over all7x7 column-pairs x4 gravity states.
  The truthful output-only one-Conv graph costs650 versus authority710, for
  projected gain `ln(710/650)=+0.088292607146`.
- Candidate SHA is
  `9aea31a6c01f7af21d893f6e5dde16dc947cdb17088686654f3f568845fbb947`.
  It scores252/265 known, primary fresh9478/10000 and9499/10000, and disjoint
  independent fresh9502/10000 and9472/10000. Disabled/default x threads1/4
  have identical prediction and raw digests, runtime/shape/nonfinite errors0,
  and minimum positive margin0.39035797.
- Channels1..9 have byte-identical weights and biases under group10 Conv, so
  the complete fixed-color196-state census extends by permutation equivariance
  to every ordered pair of distinct nonzero generator colors. The ten misses
  are disclosed, so this is normal POLICY90 rather than exact.
- Promoted as `others/71407/task012.onnx`; cumulative staged gain is now
  `+0.733852682934`. Evidence: `root_task012_h8w8_policy90_272/REPORT.md` and
  `agent_review_task012_h8w8_policy90_273/REPORT.md`.
- At promotion time, root `submission.zip` and `all_scores.csv` were observed
  to have concurrently diverged from the pinned authority. This loop did not
  overwrite that external state; all deltas remain based on immutable
  `submission_base_8009.46.zip` SHA `4eb324d7...`.

### task007 and task161 normal-POLICY90 promotions 277--280

- task007 uses the clean output-only cost68 historical Einsum against the
  cost70 authority. Primary known/fresh results are260/266, 9775/10000, and
  9726/10000; independent results are260/266, 9745/10000, and9752/10000.
  Disabled/default x threads1/4 have error/nonfinite/shape mismatch/
  small-positive/sign-difference counts0. Projected gain is
  `ln(70/68)=+0.028987536873252187`.
- task161 starts from a clean cost186 graph whose only blocker was positive
  output values below0.25. Multiplying its terminal `poly` initializer by
  exact float32 eight uniformly scales every raw output by positive eight,
  adds no node/parameter/memory, and preserves every sign. Primary
  known/fresh are265/266, 9925/10000, and9947/10000; independent fresh are
  9924/10000 and9935/10000. All four configurations have zero runtime,
  nonfinite, shape, small-positive, and config-stability findings. Projected
  gain is `ln(190/186)=+0.02127739844728488`.
- Both are normal POLICY90 candidates, not exact/private-zero claims. They are
  staged as `others/71407/task007.onnx` and `task161.onnx`; root remains
  untouched. The stage now has22 tasks with cumulative projected gain
  `+0.7841176182546227` and projected score `8010.244117618255`.
- Evidence: `root_task007_policy90_277/`,
  `agent_review_task007_policy90_278/`, `root_task161_margin_repair_279/`,
  and `agent_review_task161_margin8_280/`.

### Current Identity census 126

- Re-ran the exact no-op census against all400 current8009.46 members. The six
  highest nominal deltas were task269/289/262/214/102/353, totaling about
  +0.9504 if their Identity outputs were truly removable.
- Every rewired candidate fails full checker/strict inference: the identities
  deliberately hide constant shape arity or window length from downstream
  inference. They are structural shape witnesses, matching the earlier
  task191 failure mode, not executable no-ops under the grader contract.
- Safe adoptees: 0. Evidence: `root_exact_identity_126/REPORT.md` and
  `build.json`.

### Exact nonnegative Selu expansion 127

- Scanned every positive scalar `Mul` in the current8009.46 archive and found
  five additional cases where every source is provably finite nonnegative and
  the initializer becomes unused: tasks013/090/134/209/366.
- Replacing Mul nodes with `Selu(alpha=1,gamma=g)` and positive-source
  `Div(x,d)` with `Selu(alpha=1,gamma=1/d)` removes one parameter from
  tasks013/090/134, two from task209, and two from task366. Official costs are
  357->356, 1050->1049, 423->422, 2087->2085, and 7987->7985, for combined
  projected gain `+0.007333961806`.
- The exact numerical identity was exhaustively checked for every one of the
  31,744 nonnegative finite float16 bit patterns under both ORT modes for all
  admitted task/scalar pairs. Known outputs are raw-bitwise-equivalent in four
  configs. Task013 adds500 fresh cases per mode plus its prior complete37,800
  support proof; task090 adds1000, and tasks134/209/366 add3000 fresh cases per
  task per mode. Runtime errors are zero and minimum truth accuracy95.733%.
- The authority members contain pre-existing shape metadata cloaks, but the
  candidates add none: graph changes are limited to the proven bitwise Mul/Selu
  identities and removal of scalar initializers, giving an all-valid-input
  pass-through guarantee. Staged as `others/71407/task013.onnx`,
  `task090.onnx`, `task134.onnx`,
  `task209.onnx`, and `task366.onnx`; root submission remains immutable.
- Evidence: `root_selu_scan_127/REPORT.md`, `audit.json`,
  `exhaust_float16.json`, and `exhaust_div_float16.json`.
- A sixth exact task233 constant passed the exhaustive operator check, but its
  authority has default-ORT runtime failures; it was rejected and not staged.

### task101 exact Boolean regolf 118

- Replaced `And(all-true[1,1,3,6], scalar_bool)` with
  `Expand(scalar_bool, [1,1,3,6])`. This is an all-input Boolean identity and
  preserves the exact tensor shape while replacing18 stored Boolean values
  with a four-element int64 shape.
- Official cost is5655->5641, projected gain `+0.002478754810`; candidate SHA
  is `a57a944d958be1945563a7d55320239bb0f36b4ba25af1041f589a904cc7b81e`.
- Full checker/strict data propagation/UB0 pass, runtime shape mismatches0,
  known266 raw-equivalent in both ORT modes, and two independent5000-case
  streams are raw-equivalent to the authority in both modes. Minimum shared
  truth accuracy is98.98%, above the user's90% prior.
- Staged as `others/71407/task101.onnx`; root submission remains immutable.
  Evidence: `agent_high101_133_118/REPORT.md` and `winner_manifest.json`.

### CastLike witness attribute scan 135

- Scanned all400 authority members for initializers used exclusively as the
  fixed type witness of `CastLike`;47 tasks yielded mechanical Cast rewrites.
  Forty-six lost deliberate shape cloaks and reprofiled above their authority.
- The sole nominal strict-lower candidate was task071 188->187. It was raw
  equivalent in default ORT but failed all265 known and all3000 fresh cases in
  ORT_DISABLE_ALL with a Cast buffer-reuse shape error, so it was rejected.
- Safe adoptees:0. Evidence: `root_castlike_scan_135/build.json` and
  `audit_task071.json`.

### Discrete Selu and reduction-carrier probes 137/139

- task133: proved `scale_m1` is restricted to `{-1,0,1,2,3}` and built a
  tuned Selu that is bitwise-equal to multiplication by0.5 on all five values.
  The model is strict-lower4393->4392 and known267 raw-equal, but both authority
  and candidate share43/40 runtime errors on two2500-case streams because of30
  existing shape contradictions. Rejected under the no-error gate.
- task377: the binary-input `ReduceL1` can be replaced exactly by
  `Einsum('abcd->ab')`, removing two parameters, but runtime memory increases
  344->362 and cost worsens409->425. Rejected before fresh validation.
- Safe adoptees:0. Evidence: `root_task133_discrete_selu_137/REPORT.md` and
  `root_task377_einsum_139/REPORT.md`.

### High-memory waves 124, 128, 129

- task037/089/279: no safe strict-lower winner. task089's apparent1340->1171
  cleanup fails all known executions; the other tasks had no lower valid
  rewrite.
- task077/361: no safe strict-lower winner. task077's current rule is below100%
  fresh and its sound rebuild costs17653; task361 fails default ORT and its
  historical cost810 candidate only reaches198/266 known.
- task018/286 remains in progress. Evidence for completed lanes:
  `agent_high037_089_279_124/REPORT.md` and
  `agent_high077_361_129/REPORT.md`.

The 8009.46 archive is the latest external/user-confirmed LB update and is
immutable. It is byte-identical to protected `submission.zip`. It preserves
8008.14 and adds24 LB-white payloads from the71406/71409 sweep. The initial
25-candidate merge regressed to7990.09; individual probes isolated task185 as
black and confirmed task091 and task344@137 as white. All fixed payloads must
never be reverted.

## Local screening and LB fixation gate

- Complete known-set correctness.
- Record independent fresh accuracy for ranking; 90%+ is preferred for normal
  candidates, but lower fresh accuracy alone does not remove an otherwise
  valuable candidate from an isolated LB-probe pool.  Task254 cost42 was only
  412/500 in one local Wave38 audit and nevertheless proved LB-white.
- Candidate runtime errors: 0 on admitted generated cases.
- Full checker and strict shape inference with data propagation.
- Static positive shapes and actual runtime shapes used for cost.
- Conv/ConvTranspose/QLinearConv bias-length UB: 0.
- Strictly lower official-like cost than the current fixed member.
- Preserve ZIP order, metadata, archive comment, and every unchanged payload.

Local gates are candidate-screening evidence, not an LB guarantee.  Candidates
below 100% fresh are isolated and explicitly labeled policy90 until LB
verification. Lookup/private-zero/shape-cloak and platform-dependent failures
are not fixed into the champion merely to increase the projected number.

The user lowered the normal fresh-prior threshold from 95% to 90% after Wave 2.
Previously rejected clean candidates in the 90-95% band are therefore
re-screened.  The fresh prior is not an LB oracle: both false accepts (twelve
black tasks in the second 8006.47 wave) and false rejects (task254) are now
known.  Runtime, cost, shape, and UB evidence is still recorded, while actual
LB-white payloads override local heuristic rejection.

Private-zero lineage is now conditionally allowed into an LB-probe pool only
when pass-through can be demonstrated against the decoded true-rule reference:
complete known accuracy, 100% on multiple independent fresh seeds in both ORT
modes, runtime errors 0, strict shapes, and UB 0. A 90-99.x% approximation is
not treated as guaranteed.  No new candidate is marked fixed until its LB
classification is known: the 8006.47 candidate sweep produced twelve black
tasks even after local admission (018/048/112/134/168/198/233/251/277/286/
365/366), while only 013/070/158/379 were white.

## Wave 1 root screening

- task009 archive candidate `64cfd428...` appeared perfect on the first 100
  generated cases and reduced cost 2619 -> 2072, but failed 4625/5000 on the
  full fresh audit (7.5% accuracy): rejected.
- task054 and task138 archive candidates passed 100/100 fresh, but actual-runtime
  costs were 2372 and 2762, above incumbents 2291 and 2729: rejected.
- task076 archive candidate could not initialize under target ORT because its
  TopK dtype/kernel combination is unsupported: rejected.
- task096 candidate cost 1198 -> 1111 passed known and 4840/5000 fresh, but its
  QLinearConv has 10 output channels with a length-1 bias initializer. This is
  Conv-family bias UB and was rejected before ZIP construction.
- task219 candidate `6fffa0b8...` reduced actual cost 1479 -> 1081 and passed
  known plus 4999/5000 fresh with runtime errors 0, but static audit found a
  743 KB TfIdfVectorizer lookup graph. The isolated ZIP
  `submission_8004.50_wave1_task219_policy95_meta.zip` is evidence-only and is
  not promoted or counted toward the target.

### High-gain archive lane

- Rebased 645 unique primary candidate SHA values for tasks 145/191/204/205/285
  and rechecked nine additional high-cost tasks.
- Safe adoptees: 0; gain counted: `+0.0`.
- task205 1042 -> 937 was rejected because fresh accuracy regressed from the
  incumbent's 4928/5000 to 4904/5000.
- task286 (~+0.049) and task366 (~+0.044) did not pass the soundness/fresh/shape
  gates. Evidence: `agent_high_gain/RESULTS.json`.

### Full history miner

- Rebased 13,591 unique historical SHA values plus 643 newer SHA values against
  the actual 8004.50 payloads.
- Safe adoptees: 0; gain counted: `+0.0`.
- task204's apparent static 573 candidate costs 2544 at runtime versus incumbent
  2240; task023's cheaper candidate scored only 2/500 fresh.
- task202/205/344 were private-zero/high-arity approximations and task153 has
  QLinearConv bias 9 for 10 outputs. None were admitted.
- Evidence: `agent_history_miner/candidate_manifest.json`.

#### task153 UB repair audit

- The 236-cost historical candidate differs from the safe 237-cost incumbent
  only by truncating QLinearConv bias `B` from 10 values to 9 values while the
  weight still has 10 output channels.
- It passes known and fresh execution only through out-of-bounds bias behavior;
  restoring the required tenth bias (`-80`) raises parameters from 84 to 85 and
  returns official-like cost to 237, exactly tying the incumbent.
- Therefore no strictly cheaper defined candidate exists in this lineage. It is
  excluded even though its observed fresh agreement is 100%.

#### task382 shape repair audit

- Historical candidate `ac0d47cf...` claims 820 -> 814 and is perfect with ORT
  optimizations disabled, including two new 5000-case seeds.
- It fails every known/fresh case in default ORT because stale intermediate
  declarations cause QLinearConv buffer-reuse shape mismatches.
- Removing stale value-info, declaring the true `[1,10,30,30]` output, and
  rerunning strict shape/data propagation fixes default ORT (known 266/266;
  fresh smoke 100/100 on two seeds), but the official-like scorer cannot profile
  the truthful model and returns no cost. It therefore cannot prove a strict
  reduction and is rejected.
- Evidence: `root_task382/verification.json`,
  `root_task382/verification_truthful100.json`, and
  `root_task382/cost_comparison.json`.

### Exact rewrite Wave 2

- Scanned 400/400 models for initializer alias, dead code, no-op, duplicate
  producer, and unused optional-output reductions.
- Safe adoptees: 0; gain counted: `+0.0`.
- task124's one-byte Split-output shave crashed the validator with SIGSEGV.
- task165's apparent 592 -> 551 CastLike reuse failed all 265 known cases with
  runtime errors. Both were rejected; evidence: `agent_exact_wave2/RESULTS.json`.

## Admitted policy95 candidates

### Wave 2 — task009

- Candidate SHA-256: `b265f7f83d8fbf66c9388b9edfe0111d2b77a4b610377a3994a9c483fb445d28`
- Cost: 2619 -> 2586; gain: `+0.01268028517590921`
- Known: 265/265 in both ORT modes; runtime errors 0
- Independent fresh: seed A 4752/5000 (95.04%), seed B 4778/5000
  (95.56%); combined 9530/10000 (95.30%) in both ORT modes; runtime errors 0
- Ops: Cast/Concat/Conv/Equal/Greater/Where; no lookup or giant contraction
- Strict shape/data propagation: PASS; Conv-family bias UB: 0
- Metadata-safe ZIP: `submission_8004.50_wave2_task009_policy95_meta.zip`
- ZIP SHA-256: `56ef885cd9c0139c363098973c7a2f4f1247b07efc6296000d89c25d6a10d089`
- Projected score from LB 8004.50: `8004.512680285176`
- Projected gain from requested 8004.42 origin: `+0.092680285176`
- Remaining to 8024.42: `19.907319714824`

### Policy90 evidence-only exclusions

- task396 cost 1019 -> 947 passed 4865/5000 fresh, runtime 0, strict shape,
  and bias UB checks, but its lineage/graph is private-zero. The generated Wave
  3 ZIP is evidence-only and not counted.
- task048 cost 379 -> 378 passed two independent seeds at 90.06% and 91.08%
  with runtime 0, but is explicitly listed in the private-zero catalog. The
  generated Wave 4 ZIP is evidence-only and not counted.
- task365 also passed two seeds at 91.92% and 91.26% but is private-zero and was
  not integrated.

### Wave 5 — task343 private-catalog exclusion

- Candidate SHA-256: `6ada3c411cf90b4bcb42ff69e47eee35ed1c1b7d8b842c96c5c02c0eb06bec9e`
- Cost: 173 -> 172; nominal gain: `+0.005797117684327446` (not counted)
- Known: 266/266 in both ORT modes; runtime errors 0.
- Prior fresh: 4975/5000 (99.50%). New independent seeds: 4971/5000
  (99.42%) and 4973/5000 (99.46%) in both ORT modes; runtime errors 0.
- Full checker, strict shape/data propagation, truthful runtime shapes, standard
  domains, bias UB0, no lookup, and no giant contraction all pass. However,
  task343 is explicitly in `PRIVATE_ZERO_CATALOG` even though the candidate
  path itself has no private marker.
- Because its fresh rate is below 100%, it fails the strengthened guarantee for
  private-zero lineage and is not admitted. The combined ZIP is evidence-only:
  `submission_8004.50_wave5_task009_343_policy90_meta.zip`
- ZIP SHA-256: `f0f3116ce1cdc5ae3337b923e9b11e13c100b5741f258dff951d72740c35f70c`
- Full-ZIP Conv-family bias UB: 0; archive integrity and metadata parity: PASS,
  but this does not override the private guarantee failure.
- Counted projected score remains task009-only: `8004.512680285176`.
- Counted projected gain from requested 8004.42 origin: `+0.092680285176`.
- Remaining to 8024.42: `19.907319714824`.
- Evidence: `root_task343/verification.json`,
  `wave5_zip_manifest.json`, and `loop_7999_13/lane_c39/final_audit.json`.

## True-rule rebuild waves

### Wave 1

- Audited tasks 125/145/187/192/196/204/208/340/344 from their generators.
- Safe adoptees: 0; gain counted: `+0.0`.
- task192's only cheaper known-perfect candidate cost 1609 -> 1322 but scored
  440/500 fresh (88%), so it was rejected.
- Evidence: `agent_rebuild/RESULTS.json`.

### Wave 3 — tasks 005/080/101/133

- All healthy exact controls passed known and fresh 2000/2000 but cost no less
  than the incumbents; safe adoptees: 0.
- Cheaper task101 artifacts were fixture lookup/private-zero and task133's
  incumbent itself produced 34/2000 runtime errors, while its healthy control
  cost more. Evidence: `agent_rebuild_high2/winner_manifest.json`.

### Wave 5 — tasks 054/077/118/173

- Safe adoptees: 0; gain counted: `+0.0`.
- Healthy rewrites passed complete known, dual-ORT runtime, strict shape, margin,
  and bias checks, but cost substantially more than the fixed members:
  task054 2291 -> 49618, task077 3364 -> 17653, task118 3665 -> 51350, and
  task173 3525 -> 18494.
- Exact fresh agreement reached 100% for task054/task077/task173 controls;
  task118 reached 97.4%/97.5%. Cost alone rejects all four.
- Evidence: `agent_rebuild_high4/REPORT.md` and `agent_rebuild_high4/result.json`.

### Wave 6 — tasks 023/187/209/367

- Safe adoptees: 0; gain counted: `+0.0`.
- Healthy controls were more expensive: task367 2197 -> 3915, task209
  2150 -> 2372, and task187 1814 -> 56264.
- The task023 compact reference reached 1996/2000 and 1991/2000 on independent
  fresh runs, below the 100% guarantee required for private-zero lineage.
- Evidence: `agent_rebuild_mid5/REPORT.md` and
  `agent_rebuild_mid5/final_manifest.json`.

### Wave 7 — task036 safe adoption

- Candidate SHA-256: `fc83bef42ce52ddd5c726323bacca5c4bf59ecaa55ef2aa55b1571243e9b5738`.
- Cost: 1477 -> 1428; gain: `+0.033738139631850204`.
- Known: 265/265 in both ORT modes; runtime errors 0.
- Two independent fresh seeds: 5000/5000 each in both modes (20,000/20,000
  aggregate executions), generation errors 0, runtime errors 0.
- Full checker, strict data propagation, static positive and truthful runtime
  shapes, standard domains, bias UB0, no lookup/shape cloak/giant contraction,
  and no private-zero lineage.
- Metadata-safe combined ZIP with task009:
  `submission_8004.50_wave7_task009_036_safe_meta.zip`.
- ZIP SHA-256: `aac6119be045c7a62b4d72ef6592489c387d6d562d90b99b578e10ca89c2ef98`.
- Archive integrity/metadata parity: PASS; full-ZIP Conv-family bias UB: 0.
- Projected score from LB 8004.50: `8004.546418424808`.
- Projected gain from requested origin 8004.42: `+0.126418424808`.
- Remaining to 8024.42: `19.873581575192`.
- Evidence: `agent_rebuild_mid8/task036_audit.json` and
  `wave7_zip_manifest.json`.

### High3 — tasks 018/233/286/366

- Safe adoptees: 0; gain counted: `+0.0` across 21 attempts.
- task018/task233 private candidates scored 0/32 fresh; task286's clean lead
  reached only 86.35% (below policy90), while its 92.24% lead was rcorr lookup;
  task366 cheaper leads retained 100+ runtime-shape mismatches and truthful
  repair cost more than the incumbent.
- Evidence: `agent_rebuild_high3/REPORT.md` and
  `agent_rebuild_high3/result_manifest.json`.

### Private guarantee Wave 6 — tasks 035/066/090/377

- Audited 21 candidates; guaranteed adoptees: 0; gain counted: `+0.0`.
- Rejections: task035 known 0/266; task066 lookup or 61-input Einsum; task090
  lookup/shape cloak/no actual reduction; task377 cost408 candidates fail
  default ORT with shape buffer mismatch.
- Evidence: `agent_private_guarantee6/REPORT.md` and
  `agent_private_guarantee6/result.json`.

### Repair Wave 7 — tasks 096/205/328/343

- Safe adoptees: 0; gain counted: `+0.0`.
- task096's compact graph relies on a nontruthful output declaration; its
  truthful repair costs more than the incumbent. task205 and task343 are
  private-lineage approximations below 100% fresh. task328 is a 58-input
  Einsum with a near-zero decision margin and fails the giant-contraction gate.
- None satisfies the private pass guarantee or the clean structural gate.

### Clean mid Wave 10 — tasks 370/182/201/251

- Safe adoptees: 0; gain counted: `+0.0`.
- Historical leads for task370/task182 are nonstatic or more expensive after
  truthful profiling. task201 leads are lookup/private lineage and its sound
  control costs 7898 versus 793. task251's cheaper leads fail default ORT and
  its sound control costs 24708 versus 755.
- Evidence: `agent_clean_mid10/REPORT.md` and
  `agent_clean_mid10/result.json`.

### Clean mid Wave 9 — tasks 089/002/088/191

- Audited 2,407 historical paths representing 148 distinct SHA values.
- Safe adoptees: 0; gain counted: `+0.0`.
- task089's cheap histories use nontruthful shapes and its sound control costs
  2620 versus 1361. task002 has a direct same-input/different-output
  counterexample and its candidate uses a 66-input Einsum. task088 histories
  are shape-invalid or cost 1026 versus 902; its sound control costs 5412.
  task191 is private-zero lineage and the decoded true-rule control costs
  10829 versus 897, so no pass guarantee exists at a lower cost.
- Evidence: `agent_clean_mid9/REPORT.md`, `agent_clean_mid9/result.json`, and
  `agent_clean_mid9/audit_results.json`.

### Target mid Wave 11 — tasks 156/284/363/368

- Safe adoptees: 0; gain counted: `+0.0`.
- All four are outside the private-zero catalog and their incumbents pass the
  complete known dual-ORT and bias-UB gates. Exhaustive history deduplication
  found no strictly cheaper truthful runtime profile: task156 bottoms out at a
  cost-556 tie, task284 at 521 versus 518, task363 at a cost-513 tie, and
  task368 at 522 versus 521.
- task284/task363/task368 incumbents also rely on runtime shape cloaks; truthful
  profiles cost 57069/90998/45516, so local shaves are not promotable. task156
  is clean but already at the practical floor.
- Evidence: `agent_target_mid11/REPORT.md` and
  `agent_target_mid11/result.json`.

### Target mid Wave 12 — tasks 330/280/364/310

- Safe adoptees: 0; gain counted: `+0.0`.
- task330 uses a shape cloak, fails default ORT, and has a legal generator
  counterexample; its sound control costs 5525 versus 897. task280 has 22
  runtime-shape mismatches, default-ORT failure, a 24-input Einsum and sub-100%
  fresh agreement; its sound control costs 2161 versus 828. task364 uses an
  `input_fake` cloak and truthful profiling costs 46741 versus 685. task310
  combines TfIdf with a 23-input Einsum and reaches only 4993/5000 and
  4998/5000 fresh; its exact control costs 633 versus 566.
- Evidence: `agent_target_mid12/REPORT.md`, `agent_target_mid12/result.json`,
  and `agent_target_mid12/audit.json`.

### Private exact-alias audit — task233

- A formally exact scalar-initializer alias reduced measured cost 7432 -> 7431
  and passed a fresh 100-case smoke run, but it is not promotable.
- Complete known disable-all was 266/266 with raw-bitwise equality to the base.
  Default ORT inherited the incumbent's instability and produced two runtime
  errors; candidate/base raw equality on the remaining 264 cases confirms the
  rewrite itself is exact but does not repair the unsafe incumbent behavior.
- The two-seed fresh run was stopped after the mandatory runtime0 gate failed;
  gain counted: `+0.0`.
- Evidence: `root_private_exact/REPORT.md`,
  `root_private_exact/build_report.json`, and
  `root_private_exact/verification.json`.

### Static Shape-fold probe

- Probed current tasks 233/131/107/137/234/278/308/397/081 for exact folding
  of strict-static Shape outputs into initializers.
- Safe adoptees: 0; all potential folds either exposed pre-existing shape-cloak
  contradictions during strict inference or had no foldable Shape node.
- Evidence: `root_shape_fold/build_manifest.json`.

### Target mid Wave 13 — tasks 238/354/237/378

- Safe adoptees: 0; gain counted: `+0.0`.
- task237's cost-529 incumbent is sound (known 266/266 and two independent
  fresh 5000-case seeds in both ORT modes). New exact attempts cost 532, while
  the theoretical cost-523 sparse form fails strict validation. task238,
  task354, and task378 incumbents rely on runtime shape cloaks; their nearest
  sound controls cost 7682/6337/1651 and are all more expensive.
- Evidence: `agent_target_mid13/REPORT.md`, `agent_target_mid13/result.json`,
  and `agent_target_mid13/task237_fresh_2seed.json`.

### Private exact Wave 15 — tasks 077/102/169/187/191/216/285/366

- Safe adoptees: 0; gain counted: `+0.0`.
- Rechecked private-lineage Identity eliminations and duplicate-producer merges
  under the user's conditional pass guarantee. No candidate reached fresh:
  every rewrite either exposed strict shape contradictions, retained runtime
  shape mismatches, failed session construction, or caused allocator/runtime
  failures. task169's nominal static 248 -> 246 and task187's nominal
  1582 -> 1574 therefore remain rejected.
- Evidence: `agent_private_exact15/REPORT.md` and
  `agent_private_exact15/result.json`.

### Target mid Wave 14 — tasks 034/374/025/250

- Safe adoptees: 0; gain counted: `+0.0`.
- A truthful policy-clean task250 rebuild reduced measured cost 468 -> 464,
  but scored only 33/265 known cases in each ORT mode and was rejected before
  fresh validation. Sound controls for task034/task374/task025 cost
  3626/3361/370205 versus incumbents 511/481/474.
- Evidence: `agent_target_mid14/REPORT.md`, `agent_target_mid14/result.json`,
  and `agent_target_mid14/audit_results.json`.

### Exact Size Wave 16 — tasks 177/387/069/367

- Safe adoptees: 0; gain counted: `+0.0`.
- Direct Size-to-scalar folds nominally reduce task177 81 -> 74 and task387
  337 -> 330, but constant propagation exposes invalid CenterCropPad/HannWindow
  shape declarations and prevents both ORT sessions from loading. task069 and
  task367 match the incumbents under disable-all but fail strict/truthful-shape
  and default-ORT gates.
- Evidence: `agent_exact_size16/REPORT.md` and
  `agent_exact_size16/result.json`.

### Target mid Wave 17 — tasks 324/338/268/184

- Safe adoptees: 0; gain counted: `+0.0`; all four member SHAs are unchanged
  in the 8005.16 rebase.
- Incumbents fail truthful/dual structural gates through TopK/default-ORT,
  shape-cloak/allocator, TfIdf, or runtime-shape issues. Sound decoded-rule
  controls cost 16550/37101/18665/1996 versus 439/426/422/421. The only cheap
  task268 cost-327 lead reached just 2187/5000 and 2141/5000 fresh.
- Evidence: `agent_target_mid17/REPORT.md` and
  `agent_target_mid17/RESULT.json`.

## Current promotion state

- Promotion ZIP:
  `submission_8005.17_wave15_task013_158_254_267_323_333_safe_meta.zip`
- Promotion ZIP SHA-256:
  `cc051fdc8d8ab1c86f40a4caf4617e900d0fcd2ae12c033f696e7082c1b5a820`
- LB-fixed task: `226`; admitted pending-LB tasks: `013`, `158`, `254`, `267`, `323`, `333`.
- Projected score: `8006.483532199815`.
- Projected gain from requested origin 8004.42: `+2.063532199815`.
- Remaining to 8024.42: `17.936467800185`.
- Archive order/comment/member metadata parity and integrity: PASS; exactly six
  intended members changed. Full-ZIP Conv-family bias audit: 400 networks,
  short/long bias findings 0/0.
- Active target-range lanes continue in parallel. Private-zero lineage may be
  promoted only after decoded-rule evidence and 100% agreement on complete
  known data plus multiple independent fresh seeds in both ORT modes; 99.x%
  remains a rejection.

### Wave 10 — task158 and task323 safe adoption

- task158 SHA `3bfa7341...` reduces7615→7612 for gain
  `+0.00039403691322208417`. It passes known266/266 dual and two independent
  fresh3000-case seeds in both modes, reference agreement100%, runtime0,
  positive margin1, checker/strict/data_prop, truthful shapes, domains,
  lookup0, and bias UB0.
- task323 SHA `db773b15...` reduces106→104 for gain
  `+0.01904819497069441`. Its decoded generator support is finite and fully
  exhausted (169/169) in both ORT modes at1/4 threads; see Wave58 for the
  guarantee exception and numerical margin proof.
- Metadata-safe Wave10 ZIP and full-ZIP audit:
  `wave10_8005_16_zip_manifest.json` and `wave10_full_zip_audit.json`.

### Wave 11 — task267 finite-support guaranteed adoption

- Candidate r02 SHA `4ca7f921...` reduces cost60→30 for gain
  `+0.6931471805599453`.
- Known is264/264 dual. The decoded generator reduces algebraically to
  creature count N=12..15 and72 ordered distinct color pairs; every one of288
  support states passes disabled/default ORT at1/4 threads, runtime0,
  nonfinite0, near-margin0. Minimum positive raw is775.5849 and max absolute
  raw is1.40932e13.
- Checker, strict/data_prop, static/truthful shapes, standard domain, lookup0,
  bias UB0 pass. Its73-input contraction is allowed only under the user's
  guarantee exception after complete support exhaustion and algebraic proof
  that arbitrary creature placement is pointwise/placement-independent.
- Metadata-safe Wave11 ZIP and full-ZIP audit:
  `wave11_8005_16_zip_manifest.json` and `wave11_full_zip_audit.json`.

### Wave 12 — task254 finite-support guaranteed adoption

- Superseded packaging evidence only: after the protected 8005.17 LB update,
  current promotion was rebuilt as Wave13. Wave12's task009/task036 payloads
  are not part of the current promotion.

- Candidate SHA `814ece45...` reduces cost76->42 for gain
  `+0.5930637220029628`.
- The generator's complete support is exactly21,168 ordered height tuples:
  `P(9,4)+P(9,5)+P(9,4)`. Every tuple passes disabled/default ORT at1/4
  threads, totaling84,672/84,672 exact masks with wrong/runtime/nonfinite/
  near-positive all0. Minimum true raw is1.0000171661 and maximum false raw
  is0.0; known is265/265 in all four configurations.
- Checker, strict/data_prop, static/truthful shapes, standard domain, lookup0,
  and UB0 pass. Its33-input contraction is allowed only under the user's
  finite-support guarantee exception.
- Metadata-safe Wave12 ZIP and full-ZIP audit:
  `wave12_8005_16_zip_manifest.json` and `wave12_full_zip_audit.json`.
- Evidence: `agent_task254_deep70/REPORT.md`, `result.json`,
  `winner_manifest.json`, and `exhaustive_audit.json`.

### Wave 13 — 8005.17 safe rebase

- Rebased directly on immutable `submission_base_8005.17.zip`; task226 is
  already LB-white in that base. task009 is excluded by its recorded LB-black
  result, and task036 is excluded because cost1428 regresses from actual325.
- Pending-LB replacements are only task158/254/267/323, for aggregate gain
  `+1.3056531344468246` from8005.17 and projected score
  `8006.4756531344465`.
- Exactly four intended members changed; archive order/comment/metadata parity,
  integrity, and full400-network Conv-family bias audit all pass with0 findings.
- Evidence: `wave13_8005_17_zip_manifest.json` and
  `wave13_full_zip_audit.json`.

### Wave 14 — task013 finite-support guaranteed adoption

- Candidate SHA `ad4eb359...` removes an exact `[1,0]` initializer by reusing
  an equal diagonal, reducing actual cost638->636 for gain
  `+0.0031397200046678463` without changing real-valued semantics.
- The complete37,800-state structural support passes disabled/default ORT at
  1/4 threads:151,200/151,200 exact executions, runtime/nonfinite/near-positive
  all0. A mechanically checked colour-equivalence proof expands this to all72
  ordered colour pairs, covering all2,721,600 generator parameter states.
- Known267/267 in all four configurations, truthful55/55 runtime shapes,
  checker/strict/data propagation, lookup0 and Conv-family UB0 pass. Its giant
  contraction is accepted only under the user's complete-support guarantee.
- The current five-task Wave14 archive changes exactly task013/158/254/267/323;
  metadata, CRC, 400-model loading and full Conv-family bias audit pass.
- Evidence: `agent_finite_support74/REPORT.md`, `result.json`,
  `winner_manifest.json`, `wave14_8005_17_zip_manifest.json`, and
  `wave14_full_zip_audit.json`.

### Wave 15 — task333 all-input algebraic adoption

- Candidate SHA `0628a573...` absorbs `GE=[1,-1]` into a shared factor and
  compensates its other use with `GE^2=1`, reducing actual cost423->421 for
  gain `+0.004739345363896568`.
- Every complete Einsum monomial is unchanged for every possible input tensor.
  All80 changed-factor entries match exactly in four ORT configurations;
  known265/265x4 and whole-model two-seed testing totals8000/8000 with raw
  difference0, runtime/nonfinite/near-positive/sign differences all0.
- Full checker, strict/data propagation, truthful output shape, standard
  domains, lookup0 and UB0 pass. The giant35-input contraction is accepted
  only under this all-input proof and complete changed-factor audit.
- Wave15 changes exactly task013/158/254/267/323/333; metadata, CRC, 400-model
  loading and full Conv-family bias audit pass.
- Evidence: `agent_task333_finite81/REPORT.md`, `result.json`,
  `winner_manifest.json`, `wave15_8005_17_zip_manifest.json`, and
  `wave15_full_zip_audit.json`.

### Remaining-target Wave 67 — tasks 071/137/183/210/234/241/267/308

- Safe adoptees: task267 only, counted above; gain
  `+0.6931471805599453`. Other targets yielded no admissible candidate after
  retained and repository-wide history screening.
- Evidence: `agent_high67/REPORT.md`, `agent_high67/result.json`,
  `agent_high67/task267_exhaustive.json`, and
  `agent_high67/winner_manifest.json`.

### High-gain deep Wave 68 — tasks 285/366/286/233

- Safe adoptees: 0; gain counted: `+0.0`. task285's new exact/truthful
  initializer-sharing result costs14685 versus8623. task366 truthful costs9465
  versus7987. task286 cost7122 reaches only86.36% fresh, and task233 cost4936
  is lookup-based with0/100 fresh.
- Evidence: `agent_deep68/REPORT.md`, `agent_deep68/result.json`, and
  `agent_deep68/winner_manifest.json`.

### High-gain deep Wave 69 — tasks 349/138/076/107

- Safe adoptees: 0; gain counted: `+0.0`. task349/task138 healthy candidates
  exceed their baselines, task076's generator is non-injective for identical
  inputs, and task107 lower histories retain shape cloaks plus giant/lookup
  machinery.
- Evidence: `agent_deep69/REPORT.md`, `agent_deep69/result.json`, and
  `agent_deep69/winner_manifest.json`.

### Wave 9 — task226 safe adoption

- Candidate SHA-256:
  `852b6091385d97df6899e21304bf194440fb5cd3343385693093c24be0cb8203`.
- Runtime cost 375 -> 372; gain `+0.008032171697264253`.
- Complete known 133/133 in both ORT modes, runtime errors 0.
- Independent fresh seeds 22650001 and 22650002: each 5000/5000 in both ORT
  modes (20,000 correct executions total), runtime errors 0.
- Exhaustive valid generator width/height domain 136/136 in both modes, raw
  equal to the incumbent; full checker, strict data propagation, all 65
  runtime intermediates truthful, standard domains, bias UB0, stable margin.
- The member SHA is unchanged between 8004.50 and 8005.16, so the candidate is
  rebase-compatible. Evidence: `agent_target_mid19/task226_strict_audit.json`
  and `wave9_8005_16_zip_manifest.json`.

### 8005.16 changed-member exact scan — remaining 15 members

- Scanned tasks 133/145/182/187/201/204/216/233/255/319/330/349/361/367/370
  for byte-identical initializer aliases, output-unreachable payload, and
  isolated internal Identity removal.
- Safe adoptees: 0; gain counted: `+0.0`.
- The sole strict-static lead, task187 Identity removal, reduces the static
  estimate 1566 -> 1558 but cannot create the mandatory default optimized ORT
  session after sanitization (`TopK` axis shorter than k). It is rejected before
  known/fresh scoring.
- Evidence: `root_rebase_exact22/REPORT.md` and
  `root_rebase_exact22/scan_report.json`.

### Expanded target Wave 18 — tasks 099/279/345/239/075/392/387/225

- Safe adoptees: 0; gain counted: `+0.0`; all eight member SHAs are unchanged
  between 8004.50 and 8005.16.
- All eight independent true-rule references reached complete known 100% and
  two fresh 5000-case seeds at 100%. No lower-cost candidate satisfies the
  strict/truthful policy: the incumbents or history leads rely on shape cloaks,
  TfIdf lookup, runtime instability, or are no cheaper after truthful rebuild.
- Evidence: `agent_target_mid18/REPORT.md`, `agent_target_mid18/result.json`,
  `agent_target_mid18/model_audit.json`, and
  `agent_target_mid18/reference_audit.json`.

### New low-cost exact scan — tasks 020/030/161/175/189/193/195/281/302/304/376/384

- Safe adoptees: 0; gain counted: `+0.0`.
- All twelve 8005.16 baselines pass the strict structural precheck, but none
  contains a byte-identical initializer alias, output-unreachable payload,
  removable Identity/no-op Cast/no-op Reshape, duplicate deterministic
  producer, or unused optional output.
- Evidence: `root_newtarget_exact23/REPORT.md` and
  `root_newtarget_exact23/scan_report.json`.

### Additional mid-cost exact scan — tasks 046/157

- Safe adoptees: 0; gain counted: `+0.0`.
- Both 8005.16 members pass the strict structural precheck, but contain no
  mechanically exact reduction opportunity covered by the current scanner.
- Evidence: `root_newtarget_exact24/REPORT.md` and
  `root_newtarget_exact24/scan_report.json`.

### Full 8005.16 exact Einsum rescan

- Safe adoptees: 0; gain counted: `+0.0`; all 400 members scanned.
- The two task048 outer-product fusions are exact to a private-risk lineage
  with only 90.06%/91.08% fresh accuracy, below the private guarantee's 100%
  requirement. The sole task333 sign-absorption lead retains a giant floating
  Einsum contraction and fails the no-giant gate.
- No new initializer alias or truthful metadata reduction was found.
- Evidence: `root_exact_einsum25/REPORT.md` and
  `root_exact_einsum25/scan_report.json`.

### Full 8005.16 exact no-op/dead-code rescan

- Safe adoptees: 0; gain counted: `+0.0`; all 400 members scanned.
- Five cheaper artifacts were emitted for tasks 039/089/111/122/183, all in
  allocator/liveness-risk lineages that previously failed the complete known
  set at runtime. Because task089 changed in 8005.16, it was rechecked: both
  default optimized sessions fail to load, and disable-all differential probes
  produce runtime shape/buffer errors on all 100 inputs. It therefore still
  fails runtime0 and truthful-shape gates despite nominal cost 1349 -> 1180.
- Evidence: `root_exact_noop26/REPORT.md` and
  `root_exact_noop26/manifest_pre_differential.json`.

### 8005.16 changed-member Wave 21 — tasks 013/018/054/080/089/096/101/131

- Safe adoptees: 0; gain counted: `+0.0`.
- task080 is fully healthy at cost 3050 (known 231/231 dual, two independent
  fresh 5000-case seeds at 100% in both ORT modes, runtime0, truthful, UB0), but
  no cheaper representation was found.
- task101 is truthful but private-risk fresh accuracy is only 994/1000 and
  987/1000, below the required private guarantee of 100%. task013 retains a
  51-input giant Einsum. Tasks 018/054/089/096/131 fail runtime-shape or dual
  ORT gates. The task089 dead-code lead at cost 1180 is known 0/267 with 267
  runtime errors, default-session failure, and 49 runtime-shape contradictions.
- Evidence: `agent_rebase_new21/REPORT.md`,
  `agent_rebase_new21/result.json`, and
  `agent_rebase_new21/task089_root_candidate_audit.json`.

### Full 8005.16 optional-default scan

- Safe adoptees: 0; gain counted: `+0.0`; all 400 members scanned.
- No graph contained a removable exact-default optional input that also made an
  initializer dead (zero Conv-family bias/zero point/Pad value or unit Slice
  steps).
- Evidence: `root_optional_defaults27/REPORT.md` and
  `root_optional_defaults27/build_manifest.json`.

### Full 8005.16 Reduce-axes downgrade scan

- Safe adoptees: 0; gain counted: `+0.0`; all 400 members scanned.
- No constant Reduce-axes conversion to opset-17 remained whole-model
  schema-valid because affected graphs also require opset-18+ operator schemas.
- Evidence: `root_opset17_axes28/REPORT.md` and
  `root_opset17_axes28/build_manifest.json`.

### New low-cost Wave 22 — tasks 123/316/212/301/055/086/163/206

- Safe adoptees: 0; gain counted: `+0.0`.
- task123 is fully policy-clean at cost 266 but has no cheaper candidate.
  task316 is fresh-perfect on 500 probes, but its new collapse probe fails the
  first known case in both ORT modes. Tasks 212/301/055/163 use giant Einsum
  contractions; tasks 086/206 have false runtime shape declarations. task055's
  compatible fresh subset is only 482/493 (97.77%) and remains disallowed.
- The task163 latent-prune artifacts cost 196 -> 184 but score known 0/267 in
  both ORT modes and retain a 53-input giant Einsum.
- Evidence: `agent_new_mid22/REPORT.md`, `agent_new_mid22/result.json`, and
  `agent_new_mid22/task163_root_prunes_audit.json`.

### 8005.16 changed-member Wave 23 — tasks 133/145/182/187/201/204/216/233

- Safe adoptees: 0; gain counted: `+0.0`.
- Every incumbent has runtime/declaration shape mismatches (6 to 65 tensors).
  Tasks 145/182/187/204 fail default ORT session construction; task233 produces
  49/266 default-correct cases plus two runtime errors. task133's truthful
  control costs 5570 > 4393, task201 uses a 51,241-entry TfIdf lookup and its
  sound control costs 7898, and task216's sound control costs 9135.
- task187 Identity and task233 alias candidates inherit mandatory default ORT
  failures and remain rejected.
- Evidence: `agent_rebase_new23/REPORT.md`,
  `agent_rebase_new23/result.json`, `agent_rebase_new23/baseline_audit.json`,
  and `agent_rebase_new23/baseline_runtime_shapes.json`.

### Expanded target Wave 20 — tasks 051/064/185/200/245/264/394/397

- Safe adoptees: 0; gain counted: `+0.0`.
- task200 is the only fully clean incumbent (cost 346), but both new cost-344
  candidates score known 0/84 in both ORT modes. The older cost-345 variants
  use a length-1 bias for a two-channel Conv and are UB-dependent.
- Tasks 051/064 use 64/58-input giant Einsums, task185 uses eight TfIdf lookup
  nodes, tasks 245/264/394/397 have false runtime shape declarations, and
  tasks 264/397 additionally fail default ORT session construction. Independent
  true-rule solvers are 100% on known plus two 5000-case fresh seeds, but their
  safe controls do not beat the incumbents.
- Evidence: `agent_target_mid20/REPORT.md`,
  `agent_target_mid20/result.json`, `agent_target_mid20/current_audit.json`,
  and `agent_target_mid20/current_anatomy.json`.

### Full-sweep candidate Wave 30B — tasks 199/070/333/165/169/328/379/013

- Safe adoptees: 0; gain counted: `+0.0`; 41 candidates audited.
- Thirty-nine candidates retain giant Einsums (16 to 58 inputs), including all
  leads for tasks 199/070/333/328/379/013. task165 cost 551 and task169 cost
  246 pass superficial checker/strict/UB0 gates but fail every known case at
  runtime under disable-all and cannot create a default ORT session.
- Evidence: `agent_sweep_wave30b/REPORT.md`,
  `agent_sweep_wave30b/result.json`, and
  `agent_sweep_wave30b/audit_sweep.py`.

### Extended 8005.16 exact-rewrite sweep

- Safe adoptees: 0; gain counted: `+0.0`; 14 additional full-payload passes.
- task071 nominal 188 -> 187 is default-known 265/265 but disable-all runtime
  fails all 265 cases and it retains a 39-input giant Einsum. task397 nominal
  364 -> 362 inherits false shapes, CenterCropPad/TfIdf usage, and default ORT
  failure. task333 nominal 423 -> 421 retains a 36-input giant Einsum.
- All other exact-rewrite passes emitted no lower-cost candidate.
- Evidence: `root_sweep33/REPORT.md` and per-pass JSON manifests under
  `root_sweep33/`.

### Latent-prune candidate Wave 30A — tasks 010/028/060/175/229/232/304/315

- Safe adoptees: 0; gain counted: `+0.0`; 32 lower-cost variants audited.
- No candidate reached complete known 100% in either ORT mode. Best results by
  task were 010 0/265, 028 0/265, 060 0/265, 175 262/266, 229 90/267,
  232 0/266, 304 1/266, and 315 19/266. Runtime errors were zero, but the
  mandatory known gate failed; tasks 010/060/175/229/304/315 also retain giant
  Einsum contractions.
- Evidence: `agent_prune_wave30a/REPORT.md` and
  `agent_prune_wave30a/result.json`.

### New low-cost Wave 32 — tasks 221/136/278/230/327/391/097/027

- Safe adoptees: 0; gain counted: `+0.0`; all eight members are byte-identical
  between 8004.50 and 8005.16.
- Tasks 136/278/230/097/027 only have same-cost or more-expensive safe history.
  task221's nominal 144 -> 142 edit fails the complete known set at runtime
  under ORT_DISABLE_ALL. The proposed cost-46 task327 one-node ConvTranspose is
  mathematically infeasible across all 51 legal placements because identical
  local patches require opposing labels. task391's cheaper cost-85/87/88
  artifacts are TfIdf lookup/private-zero models; its smallest table-free
  true-rule control costs 139 versus the incumbent 104.
- Evidence: `agent_new_low32/REPORT.md`, `agent_new_low32/result.json`,
  `agent_new_low32/history_audit.json`, and
  `agent_new_low32/evidence/task327_one_node_infeasible.json`.

### New low-cost Wave 34 — tasks 033/282/084/362/381/001/352/283

- Safe adoptees: 0; gain counted: `+0.0`; all eight payloads are unchanged from
  the exhaustive 7999.13 scan and were independently rerun against 8005.16.
- Tasks 033/084/362/381/001/352 have only giant-Einsum or cost-dominated lower
  history. Tasks 282/283 pass complete known raw equivalence after an exact
  CastLike-to-Cast rewrite, but exposing their true runtime tensors increases
  actual cost to 27,994/30,685 and task282/283 also fail the default-runtime and
  truthful-output-shape gates. They are therefore not promotion candidates.
- Evidence: `agent_new_low34/REPORT.md`, `agent_new_low34/result.json`,
  `agent_new_low34/audit_results.json`, and
  `agent_new_low34/exact_shave_audit.json`.

### New mid/low Wave 31 — tasks 046/157/161/189/384/193/195/281

- Safe adoptees: 0; gain counted: `+0.0`; all eight exact 8005.16 members were
  independently structurally and dual-runtime audited.
- task193 was the sole credible clean lead. Every 1x1 through 4x4
  kernel/padding family with actual cost below 170 was tested against all 266
  stored cases: all smaller kernels have contradictory identical-patch labels,
  and the only conflict-free 4x4 no-bias placement is not linearly separable.
  The remaining tasks have no lower history or fail private-risk, truthful
  runtime-shape, default-runtime, giant-Einsum, or fresh true-rule gates.
- Evidence: `agent_new_mid31/REPORT.md`, `agent_new_mid31/result.json`,
  `agent_new_mid31/baseline_audit.json`, `agent_new_mid31/true_rule_audit.json`,
  and `agent_new_mid31/task193_conv_search.json`.

### New low-cost Wave 35 — tasks 050/329/350/356/371/360/214/083

- Safe adoptees: 0; gain counted: `+0.0`; decoded rules reproduce the complete
  known sets for all eight tasks, and all current members are byte-identical to
  the audited 8004.50 base.
- task050's four cost-84 factorizations fail the first training example in both
  ORT modes. task350/356 cost-60 rank-one approximations each score 0/100 in
  both modes. task214's static cost-75 history is known-false and fresh 0/20.
  The other tasks have no strict lower actual-cost graph or only same-cost
  giant-Einsum history.
- Evidence: `agent_new_low35/REPORT.md`, `agent_new_low35/result.json`,
  `agent_new_low35/baseline_audit.json`, `agent_new_low35/true_rule_audit.json`,
  and `agent_new_low35/history_audit.json`.

### New low-cost Wave 37 — tasks 320/154/393/290/336/003/058/072

- Safe adoptees: 0; gain counted: `+0.0`; all eight decoded rules reproduce
  their complete known sets and the latest payloads are unchanged from 8004.50.
- The only archive entries numerically below the current floor were task290
  static-cost 73/75/88 models, but official runtime profiling gives actual
  costs 91/93/97 versus the incumbent 91. All other histories are same-cost,
  cost-dominated, shape-cloaked, or giant-Einsum lineages.
- Evidence: `agent_new_low37/REPORT.md`, `agent_new_low37/result.json`,
  `agent_new_low37/baseline_audit.json`,
  `agent_new_low37/known_baseline_dual.json`, and
  `agent_new_low37/history_audit.json`.

### New low-cost Wave 38 — tasks 141/004/254/049/287/078/095/007

- Safe adoptees: 0; gain counted: `+0.0`; all eight decoded rules reproduce the
  complete known sets and latest member bytes remain unchanged from 8004.50.
- task254 cost-42/68 variants retain giant contractions and the former differs
  on 412/500 external cases; task049's apparent static 69 floor profiles at
  actual cost 88 versus incumbent 75; task287 cost30 is known 263/267; task007
  cost68 is known 260/266 in both ORT modes. Other targets have no lower graph.
- Evidence: `agent_new_low38/REPORT.md`, `agent_new_low38/result.json`,
  `agent_new_low38/baseline_audit.json`,
  `agent_new_low38/history_audit.json`, and
  `agent_new_low38/evidence/task007_cost68_known_dual.json`.

### New low-cost Wave 36 — tasks 149/390/272/147/040/176/252/127

- Safe adoptees: 0; gain counted: `+0.0`; decoded rules reproduce the complete
  known sets and all latest payloads are unchanged from 8004.50.
- Exact sparse-initializer rewrites for tasks040/176/252 would nominally reduce
  the parameter count, but all three fail full ONNX checking because shape
  inference exposes the sparse Einsum operand as rank zero. task272's lone
  cost-82 lower history cannot create an ORT session due to mixed int64/int32
  inputs to Max. Other histories are same-cost, giant, or dominated.
- Evidence: `agent_new_low36/REPORT.md`, `agent_new_low36/result.json`,
  `agent_new_low36/baseline_audit.json`, `agent_new_low36/known_dual.json`,
  `agent_new_low36/history_audit.json`, and
  `agent_new_low36/sparse_build_manifest.json`.

### New low-cost Wave 41 — tasks 380/242/298/026/261/351/274/317

- Safe adoptees: 0; gain counted: `+0.0`; decoded rules reproduce every known
  pair and all latest payloads are byte-identical to 8004.50.
- No one of 13,591 unique archive graphs or 1,134 focused graphs is strictly
  below the current floor. task380 is structurally truthful at cost60 but its
  dense 30x2 contraction shape is required and alternative history starts at
  cost99. The remaining seven incumbents are shape-cloaked, lookup-based, or
  nondeterministic and have only ties or more-expensive alternatives.
- Evidence: `agent_new_low41/REPORT.md`, `agent_new_low41/result.json`,
  `agent_new_low41/baseline_audit.json`,
  `agent_new_low41/true_rule_audit.json`, and
  `agent_new_low41/history_audit.json`.

### New low-cost Wave 39 — tasks 032/041/215/211/120/235/258/292

- Safe adoptees: 0; gain counted: `+0.0`; all decoded rules reproduce their
  complete known sets and latest payloads are unchanged from 8004.50.
- task032 cost46 fails every known case at runtime; task211 cost64 retains a
  25-input giant Einsum and is only 9/266; task120's apparent static cost41 is
  actual cost2738; task292 cost50/50/54 variants are each 0/28 in both ORT
  modes. The remaining tasks have no sub-baseline history.
- Evidence: `agent_new_low39/REPORT.md`, `agent_new_low39/result.json`,
  `agent_new_low39/baseline_audit.json`,
  `agent_new_low39/lower_leads_dual.json`, and
  `agent_new_low39/history_audit.json`.

### New low-cost Wave 40 — tasks 022/181/104/294/128/152/203/236

- Safe adoptees: 0; gain counted: `+0.0`; all eight latest members remain
  byte-identical to 7999.13/8004.50 and have no below-baseline graph among the
  13,591 unique archive entries.
- task294's apparent duplicate ConstantOfShape lead uses three different shape
  constants (31/29/30) and cannot be merged. Tasks128/203 are truthful one-node
  contractions already at a 60-parameter, zero-intermediate floor; task152 is
  giant, and the others retain runtime shape contradictions with no lower safe
  history.
- Evidence: `agent_new_low40/REPORT.md`, `agent_new_low40/result.json`,
  `agent_new_low40/baseline_audit.json`,
  `agent_new_low40/exact_candidate_scan.json`, and
  `agent_new_low40/history_audit.json`.

### New low-cost Wave 42 — tasks 339/126/021/171/346/227/318/332

- Safe adoptees: 0; gain counted: `+0.0`; decoded rules reproduce all stored
  pairs and no strictly lower graph appears in the 13,591-entry archive,
  focused harvest, or new exact scan.
- Tasks339/021 are runtime-shape truthful but already at their 53/51 structural
  floors. The other six are shape-cloaked; task171 additionally fails all 54
  default-runtime cases, while task346 belongs to a private-risk lineage.
- Evidence: `agent_new_low42/REPORT.md`, `agent_new_low42/result.json`,
  `agent_new_low42/baseline_audit.json`,
  `agent_new_low42/true_rule_audit.json`, and
  `agent_new_low42/history_audit.json`.

### New low-cost Wave 43 — tasks 006/334/244/249/347/386/146/291

- Safe adoptees: 0; gain counted: `+0.0`; all current members and decoded true
  rules pass their complete known sets.
- task006 cost30/38/40/40 lower models score 0/266, runtime-fail all cases,
  runtime-fail all cases, and 27/266 respectively. task291 cost30 is 0/265 in
  both modes. task146 static38 histories profile at actual cost67 versus the
  incumbent40 and fail default session construction. Other tasks have no lower
  archive, focused-harvest, or exact lead.
- Evidence: `agent_new_low43/REPORT.md`, `agent_new_low43/result.json`,
  `agent_new_low43/baseline_audit.json`,
  `agent_new_low43/lower_leads_dual.json`, and
  `agent_new_low43/history_audit.json`.

### New low-cost Wave 44 — tasks 303/098/395/167/289/038/262/269

- Safe adoptees: 0; gain counted: `+0.0`; decoded rules reproduce every known
  pair, and the archive plus focused harvest contains no strictly lower model
  for any target.
- Exact Identity removal on tasks289/262/269 is rejected because full checker
  and strict data-propagating shape inference expose inherited CenterCropPad or
  HannWindow shape contradictions. The remaining five targets have no exact
  rewrite hit.
- Evidence: `agent_new_low44/REPORT.md`, `agent_new_low44/result.json`,
  `agent_new_low44/baseline_audit.json`,
  `agent_new_low44/true_rule_audit.json`, and
  `agent_new_low44/history_audit.json`.

### New low-cost Wave 45 — tasks 024/113/385/389/296/399/359/110

- Safe adoptees: 0; gain counted: `+0.0`; all eight baselines are known100 in
  both ORT modes and pass full checker, strict/data_prop, runtime-shape,
  standard-domain, and Conv-bias audits.
- The complete history contains numeric lower entries only for task385 (five)
  and task389 (one); all six score 0/20 on both known and fresh screens. The
  remaining six tasks have no strictly lower historical or exact lead.
- Evidence: `agent_new_low45/REPORT.md`, `agent_new_low45/result.json`,
  `agent_new_low45/baseline_audit.json`,
  `agent_new_low45/known_baseline_dual.json`,
  `agent_new_low45/true_rule_audit.json`, and
  `agent_new_low45/history_audit.json`.

### High-cost history pre-screen 49 — tasks 037/297/014/092/398/218/132/388

- Safe adoptees: 0; gain counted: `+0.0`; every retained numeric lower history
  lead was re-profiled and rerun on the complete known set in both ORT modes.
- task297 is the only lower model that is known100 dual, actual361 versus371,
  and otherwise structurally clean. It is nevertheless out of schema because
  its Conv has negative end padding `-24`; standard Slice/Split repairs cost
  484/511. Other leads are actual-more-expensive, runtime-invalid, known-false,
  shape-cloaked, giant, or only tie the latest actual cost.
- Evidence: `root_high49/REPORT.md`, `root_high49/result.json`, and
  `root_high49/history_lead_audit.json`.

### High-cost history pre-screen 50 — tasks 222/228/159/029/178/148/341/357

- Safe adoptees: 0; gain counted: `+0.0`; all retained numeric lower history
  leads were re-profiled and run on the complete known set.
- task228's lower actual294 and task159's lower actual291 are known0; task178
  actual261 is only68/268; task222 candidates reach at most128/266; task357
  candidates are all0/13. task029 is runtime-invalid, and tasks148/341 have no
  lower retained history.
- Evidence: `root_high50/REPORT.md`, `root_high50/result.json`, and
  `root_high50/history_lead_audit.json`.

### High-cost history pre-screen 51 — tasks 355/174/325/042/143/247/079/065

- Safe adoptees: 0; gain counted: `+0.0`; all retained lower history was
  re-profiled and complete-known screened.
- task143 actual148 versus212 is known100 dual, but uses TfIdf lookup plus a
  17-input giant Einsum and independently scores only2/5000 and3/5000 fresh.
  task355/task042 lower models miss known cases; task174/task079/task065 known
  survivors tie or exceed the base actual cost; tasks325/247 have no lower lead.
- Evidence: `root_high51/REPORT.md`, `root_high51/result.json`,
  `root_high51/history_lead_audit.json`, and
  `scripts/golf/loop_7999_13/lane_c11/fresh_audit.json`.

### High-cost history pre-screen 52 — tasks 115/114/273/105/259/031/263/300

- Safe adoptees: 0; gain counted: `+0.0`; all retained numeric lower history
  was re-profiled and complete-known screened.
- task273 actual192 versus193 is known0/266. task105's known-perfect history
  costs198 versus188. task031's lowest known-perfect history costs186 versus
  base183, and its reported static185 entry actually costs209. The other five
  targets have no retained numeric lower lead.
- Evidence: `root_high52/REPORT.md`, `root_high52/result.json`, and
  `root_high52/history_lead_audit.json`.

### High-cost expanded Wave 48 — tasks 008/275/134/112/168/109/160/170

- Safe adoptees: 0; gain counted: `+0.0`; 49 retained/SOUND models were
  runtime-reprofiled, including 16 actual-lower entries, with zero safe
  pre-fresh finalist.
- task134 cost320/322 is known100 but lookup-based and reaches only about96%
  on independent fresh5000 dual. task168 cost166/285 is lookup plus giant
  contraction. task275 is128/266, task160 lower models are known0, and the
  other tasks have no truthful actual-lower safe graph.
- Evidence: `agent_high48/REPORT.md`, `agent_high48/result.json`,
  `agent_high48/history_audit.json`, and
  `agent_high48/winner_manifest.json`.

### High-cost history pre-screen 53 — tasks 383/068/400/224/240/059/358/190

- Safe adoptees: 0; gain counted: `+0.0`; all retained numeric lower history
  was runtime-reprofiled and complete-known screened.
- Actual-lower leads for tasks383/224/059 are known0. task240 reaches69/266
  and task190 reaches56/266. task068's known-perfect histories all exceed its
  base actual167 cost. Tasks400/358 have no retained numeric lower lead.
- Evidence: `root_high53/REPORT.md`, `root_high53/result.json`, and
  `root_high53/history_lead_audit.json`.

### High-cost history pre-screen 54 — tasks 243/162/119/180/295/074/093/271

- Safe adoptees: 0; gain counted: `+0.0`; task243/task162 histories reprice
  far above their baselines, task119 is higher and only109/266, and four tasks
  have no retained lower lead.
- task271 actual126 versus135 is known100 dual, but uses nine TfIdfVectorizer
  nodes and independently passes only2/5000 fresh in both modes with4998
  failures. It fails the private guarantee gate.
- Evidence: `agent_high54/REPORT.md`, `agent_high54/result.json`,
  `agent_high54/task271_private_gate.json`, and
  `agent_high54/winner_manifest.json`.

### High-cost history pre-screen 55 — tasks 349/138/076/255/319/361/107/396

- Safe adoptees: 0; gain counted: `+0.0`.
- task255's apparent zero-input cost878 lead costs1342 on the full official-like
  trace versus baseline1336 and has18 shape cloaks plus a non-functional
  generator. task107's apparent638 lead has a66-input giant Einsum, GatherND,
  and13 shape cloaks. task396 lower histories remain private/lookup candidates
  without the required fresh100 guarantee. The other five targets have no
  safe, known-perfect, strictly lower retained lead.
- Evidence: `root_high55/REPORT.md`, `root_high55/result.json`,
  `root_high55/history_lead_audit.json`, and
  `scripts/golf/loop_7999_13/lane_b18/candidate_audit.json`.

### Mid-cost history pre-screen 57 — tasks 015/313/256/321/057/011/085/194

- Safe adoptees: 0; gain counted: `+0.0`; tasks256/194 lower leads are known0
  and task085 is5/265. task057 known-perfect histories all reprice above its
  baseline. Four targets have no retained numeric lower lead.
- Evidence: `root_high57/REPORT.md`, `root_high57/result.json`, and
  `root_high57/history_lead_audit.json`.

### High-cost expanded Wave 47 — tasks 044/012/198/277/117/270/019/062

- Safe adoptees: 0; gain counted: `+0.0`; 4,872 history observations and312
  unique variants produced21 numeric-lower candidates, all rejected.
- task012 is235/265; task198's16 leads use22–57-input giant Einsums; task277
  uses4/10 TfIdfVectorizers; task117 has95 shape mismatches; task019 exits ORT
  with signal11. Tasks044/270/062 have no actual-lower history.
- Evidence: `agent_high47/REPORT.md`, `agent_high47/result.json`, and
  `agent_high47/winner_manifest.json`.

### Mid-cost expanded Wave 56 — tasks 348/369/306/106/091/121/108/265

- Safe adoptees: 0; gain counted: `+0.0`; 247 unique history models yielded31
  actual-lower entries and only three known100 dual candidates.
- All three survivors are task091 cost122/117/124 variants with
  ScatterElements and8/14/14 runtime-shape mismatches. task348/task306/task121
  lower entries fail known; the remaining targets have no valid lower lead.
- Evidence: `agent_high56/REPORT.md`, `agent_high56/result.json`,
  `agent_high56/history_lead_audit.json`, and
  `agent_high56/winner_manifest.json`.

### Mid-cost expanded Wave 59 — tasks 151/213/122/094/220/260/342/331

- Safe adoptees: 0; gain counted: `+0.0`; 76 nonbaseline candidates yielded21
  strict-lower entries, all failing known. task260 reaches only23/266 and
  task122 has266/266 runtime errors.
- Evidence: `agent_high59/REPORT.md`, `agent_high59/result.json`, and
  `agent_high59/winner_manifest.json`.

### Mid-cost expanded Wave 60 — tasks 353/081/111/248/266/375/231/142

- Safe adoptees: 0; gain counted: `+0.0`; task353's known-perfect candidates
  cost99/104 versus base93. task111 candidates exceed base89 and fail all265
  default runs. task231 cost59 versus64 is known0/266. Five targets have no
  retained lower lead.
- Evidence: `agent_high60/REPORT.md`, `agent_high60/result.json`, and
  `agent_high60/winner_manifest.json`.

### Mid-cost Wave 58 — task323 exhaustive guaranteed adoption

- task323 robust candidate SHA `db773b15...` reduces official-like cost
  106→104 for gain `+0.019048194970697097`.
- Known is172/172 dual. The decoded generator's complete fixed support—all169
  size13 seed placements—passes disabled/default ORT at1/4 threads with zero
  runtime/nonfinite failures. Positive raw margin is at least1.018e25 and every
  nonzero raw value has absolute magnitude at least2.599e18.
- Checker, strict/data_prop, truthful shapes, standard domains, bias UB0,
  lookup0, and cloak0 pass. The inherited giant contraction is accepted only
  under the user's guarantee exception after moving its sensitive exact-zero
  coefficient ten ULPs negative and exhausting every generator state.
- Evidence: `root_high58/REPORT.md`, `root_high58/result.json`,
  `root_high58/task323_robust_audit.json`, and
  `root_high58/winner_manifest.json`.

### Low-cost history pre-screen 61 — tasks 100/139/144/202/188/039/186/293

- Safe adoptees: 0; gain counted: `+0.0`.
- task202 cost20/28 histories are private-zero approximations; the best reaches
  only412/422 fresh (97.63%) and fails the required100% guarantee. task139 and
  task039 lower histories are known0. Five targets have no retained lower lead.
- Evidence: `root_high61/REPORT.md`, `root_high61/result.json`, and
  `root_high61/history_lead_audit.json`.

### Low-cost history pre-screen 62 — tasks 116/056/082/150/155/043/045/047

- Safe adoptees: 0; gain counted: `+0.0`; task056 cost18 is35/46 known and
  task150 lower histories are known0 or unscorable. Six targets have no
  retained numeric lower lead.
- Evidence: `root_high62/REPORT.md`, `root_high62/result.json`, and
  `root_high62/history_lead_audit.json`.

### Low-cost history pre-screen 63 — tasks 052/063/166/299/322/314/164/172

- Safe adoptees: 0; gain counted: `+0.0`; task322 cost19 candidates use a
  nine-element bias for ten ConvTranspose outputs and produce nonfinite values.
  task172 lower entries are known0; task063's known-perfect history is dearer.
  Five targets have no retained lower lead.
- Evidence: `root_high63/REPORT.md`, `root_high63/result.json`, and
  `root_high63/history_lead_audit.json`.

### Lowest-cost history pre-screen 66 — tasks 140/223/307/326/135/129/067/179

- Safe adoptees: 0; gain counted: `+0.0`; task326/task135 zero-cost histories
  are known0/266. Six targets have no lower retained history, including the
  already cost0 tasks067/179.
- Evidence: `root_high66/REPORT.md`, `root_high66/result.json`, and
  `root_high66/history_lead_audit.json`.

### Lowest-cost expanded Wave 65 — tasks 276/305/309/312/337/373/053/087

- Safe adoptees: 0; gain counted: `+0.0`; 4,520 path aliases reduced to41
  unique task/model pairs (33 nonbaseline), with zero strict-lower candidate.
  Several models tie the baseline, while task373/task087 alternatives are
  substantially dearer.
- Evidence: `agent_high65/REPORT.md`, `agent_high65/result.json`, and
  `agent_high65/winner_manifest.json`.

### Lowest-cost expanded Wave 64 — tasks 103/372/073/130/016/017/061/197

- Safe adoptees: 0; gain counted: `+0.0`; 4,622 paths reduced to137
  nonbaseline candidates and13 strict-lower entries, with zero safe finalist.
- task372 cost12 entries have nonfinite initializers plus a9/10 ConvTranspose
  bias. task103/task073 lower entries fail known/runtime/shape gates. The other
  five targets have no strict-lower history.
- Evidence: `agent_high64/REPORT.md`, `agent_high64/result.json`, and
  `agent_high64/winner_manifest.json`.

### Expanded private-lineage guarantee pre-screen 73 — three files

- Safe adoptees: 0; gain counted: `+0.0`.
- Two task048 exact-fusion files (379->378) reproduce legal generator
  counterexamples on both independent 5000-case seeds (4503/5000 and
  4554/5000). task365 cost1337 versus1369 likewise has counterexamples
  (4596/5000 and4563/5000). Their combinatorial generator supports have not
  been exhausted, so the user's private-zero guarantee exception does not
  apply despite all three clearing90%.
- Evidence: `root_expanded73/REPORT.md` and `root_expanded73/result.json`.

### task198 finite-generator guarantee audit 71 — sixteen files

- Safe adoptees: 0; gain counted: `+0.0`; all16 strict-lower, known266/266
  dual candidates were rerun on two independent1000-case seeds under
  disabled/default ORT and1/4 threads with runtime0.
- Every file has an exactly regenerable legal generator counterexample. The
  best empirical candidate is cost628 versus661 and1961/2000 (98.05%), but it
  still fails legal seed47000199 index89; the cheapest cost554 model is only
  1792/2000. Universal coverage is therefore disproved for all16.
- Evidence: `agent_task198_deep71/REPORT.md`, `result.json`,
  `winner_manifest.json`, `counterexamples.json`, and `fresh_matrix.json`.

### Giant/private finite-support cross-scan 72 — at least forty-four files

- Safe adoptees: 0; gain counted: `+0.0`; 1,136 report/result/winner files
  were searched and candidates across tasks013/051/064/070/199/202/328/333/
  379 were re-audited, excluding dedicated task198/254/267/323 lanes.
- task202 cost28 versus48 has a legal20-cell generator counterexample;
  task070/task199 lower candidates are known0. task379 is4999/5000 and not a
  guarantee. task328 cost554 versus558 is sign-correct on the finite audit but
  has only7.3e-11 minimum positive raw and is held for a robust-margin repair.
  task013/task333 pass known in four configurations but lack complete support
  guarantees; task051/task064 have no cheaper qualifying candidate.
- Evidence: `agent_giant_scan72/REPORT.md`, `result.json`,
  `winner_manifest.json`, `audit.json`, and `official_costs.json`.

### Private/lookup expanded bundle 75 — thirty-three files

- Safe adoptees: 0; gain counted: `+0.0`; 33 byte-distinct strict-lower files
  across tasks134/202/219/271/343/396 were official-profiled against immutable
  8005.17, with26 reaching known dual100%.
- All are rejected by legal generator counterexamples, lookup/Scatter,
  runtime-shape cloak, or an exact true-rule control that exceeds the baseline.
  Notably task343's clean cost172 pair remains only about4965--4976/5000;
  task396's cost1014 occupancy attempt is4963/5000; task202 all three giant
  candidates have generator counterexamples. None meets the private guarantee.
- Evidence: `agent_private_bundle75/REPORT.md`, `result.json`,
  `winner_manifest.json`, `authority_members.json`,
  `extended_candidate_audit.json`, and `retained_audit.json`.

### task328 robust-margin repair 76 — two boundary files

- Safe adoptees: 0; gain counted: `+0.0`.
- The cost554 source needs at least a2^32 output scale to raise its retained
  minimum true value7.31687e-11 to0.25, while a legal source witness already
  reaches1.44181e34 and limits finite float32 scaling to at most2^14. The
  bounds are disjoint.
- Fixed candidates at2^14 and2^32 both destroy output signs on legal generator
  witnesses in disabled/default ORT at1/4 threads. The former creates false
  positives/negative true cells before overflow; the latter likewise flips28
  false and28 true cells. No uniform power-of-two repair is safe.
- Evidence: `agent_task328_robust76/REPORT.md`, `result.json`,
  `winner_manifest.json`, `witness_audit.json`, and `orbit_checkpoint.json`.

### task343 exact-under-floor search 77 — expanded classifier space

- Safe adoptees: 0; gain counted: `+0.0`; authority remains cost173.
- The known cost172 approximation is only4976/5000 fresh and therefore still
  fails the private-lineage guarantee. The exact rule model is5000/5000 but
  costs178, so it is a regression.
- Exhaustive bounded searches covered930 local features,56,074 affine scalar
  Conv features, shared constants,129 direct anchors, Cast+relation anchors,
  300x600 multi-Cast/relation combinations, and11,978 truth signatures without
  finding an exact cost<=172 expression.
- Evidence: `agent_task343_sound77/REPORT.md`, `result.json`,
  `winner_manifest.json`, and `candidate_comparison.json`.

### task192 retraining search 79 — private-zero fail-closed

- Safe adoptees: 0; gain counted: `+0.0`; task192 is explicitly listed in the
  private-zero catalog and was previously LB-black in the h7904 pool.
- A new cost1309 model (baseline1609) passes known265/265, strict/data_prop,
  truthful runtime shapes, standard domains, Conv UB0 and independent fresh
  seeds at92.2%/92.4% with runtime0. It still has39/38 legal fresh errors, so
  it is evidence-only and cannot use the private guarantee exception.
- The exact generator-SOUND control passes known and prior fresh5000/5000 but
  costs3325, so it is a regression.
- Evidence: `agent_task192_retrain79/REPORT.md`, `result.json`,
  `winner_manifest.json`, and `final_audit.json`.

### task023 known-constrained rank search 82 — threshold miss

- Safe adoptees: 0; gain counted: `+0.0`.
- The best cost1541 kernels preserve known266/266 in disabled/default ORT,
  strict/data_prop, truthful runtime output, standard domains, UB0 and
  lookup/custom/giant0, but independent fresh5000 seeds remain below policy90:
  root coordinate2=88.94%/89.18%, new global integer=88.92%/89.20%.
- Structured and all-pair losses, three-seed worst-shard optimization, and
  known-hard global integer search all plateau below90%. The generator is also
  non-injective, so no universal claim is possible.
- Evidence: `agent_task023_rank82/REPORT.md`, `result.json`,
  `winner_manifest.json`, `audit.json`, and `integer_search_report.json`.

### Algebraic twenty-file expansion 83 — thirty files audited

- Safe adoptees: 0; gain counted: `+0.0`; thirty high-operand task150--400
  files and34 latent-prune candidates were audited for exact factor absorption.
- task328 558->554 is algebraically exact but retains a7.31687e-11 positive,
  failing the robust margin gate. task379 1949->1947 is algebraically exact,
  known266/266x4 and near-positive0, but retains12,896 negative infinities per
  configuration and fails the nonfinite0 gate.
- Every latent-prune candidate across tasks163/175/199/229/232/304/315 has a
  deterministic real-input counterexample.
- Evidence: `agent_algebraic20_83/REPORT.md`, `result.json`, and
  `winner_manifest.json`.

### Mid-cost twenty-task expansion 84 — 878 SHA candidates

- Safe adoptees: 0; gain counted: `+0.0`; tasks008/014/037/062/092/099/109/
  112/160/168/245/250/275/279/297/345/374/394/397/398 were scanned across
  every loose ONNX and ZIP member, yielding878 non-authority SHA values.
- Ninety candidates reached actual profiling and28 reached complete known
  scoring; none passed all gates. Authority profiling corrected stale costs for
  task014/109/245 to370/405/387.
- task297 cost361 versus371 passed known and two fresh5000 seeds100%, but uses
  schema-invalid negative Conv padding `[0,0,0,-24]`. Legal Slice and Split
  repairs cost484/511, both regressions, so it remains quarantined.
- Evidence: `agent_mid20_84/REPORT.md`, `result.json`,
  `winner_manifest.json`, and `audit/inventory_summary.json`.

### Mid-cost second twenty-task expansion 86 — 1,140 SHA candidates

- Safe adoptees: 0; gain counted: `+0.0`; 36,145 observations across loose
  files and1,259 ZIPs reduced to1,140 nonbaseline SHA values for tasks025/048/
  102/132/134/170/184/200/222/228/234/239/264/268/308/324/338/377/387/388.
- Ninety-eight reached actual profiling,24 were actually lower,8 passed known
  in four configurations, and7 had truthful shapes. All seven are task048
  cost378 versus379 variants with43/500 legal fresh counterexamples; task048
  is private-zero, so91.4% cannot satisfy its guarantee requirement.
- task134 cost412 versus423 passed known x4 but had six runtime shape
  contradictions. Other actual-lower candidates failed known or an ORT mode.
- Evidence: `agent_mid20b_86/REPORT.md`, `result.json`,
  `winner_manifest.json`, and `audit/final_decisions.json`.

### Mid-cost third twenty-task expansion 87 — 800 SHA candidates

- Safe adoptees: 0; gain counted: `+0.0`; tasks029/051/064/091/123/124/137/
  148/153/169/174/178/199/212/301/316/325/341/355/357 were exhaustively
  inventoried across loose ONNX and ZIP members, yielding800 nonbaseline SHA.
- Sixty-five reached actual profiling and59 complete known profiling; none was
  both known-correct and strictly cheaper. The authority cost for task091 was
  corrected from stale126 to official265; known-correct histories start at266.
- task124's apparent one-byte Split-output shave reproducibly exits139/SIGSEGV
  from an ORT allocator/liveness mismatch. Private tasks169/174/178/325 lacked
  a complete true-rule proof and remained excluded.
- Evidence: `agent_mid20c_87/REPORT.md`, `result.json`,
  `winner_manifest.json`, and `audit/task124_runtime_crash.json`.

### Mid-cost fourth twenty-task expansion 88 — 803 SHA candidates

- Safe adoptees: 0; gain counted: `+0.0`; tasks031/042/055/065/071/079/086/
  088/105/114/115/143/161/163/189/206/247/259/273/344 were inventoried across
  12,030 loose ONNX observations and 23,671 members from 1,259 ZIP files.
- 803 distinct nonbaseline SHAs produced 28 actual strict-lower candidates;
  22 passed all known examples in disabled/default ORT at one/four threads, but
  none had truthful runtime shapes.  Task088 accounts for all 22 known-perfect
  candidates: 21 have 11--18 declared/actual shape contradictions and the last
  has a duplicate `label_scale` node that prevents truthful tracing.
- task071 cost188->186 misses one of 265 known cases in every configuration;
  task161 cost190->186 misses one of 266.  No candidate entered fresh testing.
- Evidence: `agent_mid20d_88/REPORT.md`, `result.json`,
  `winner_manifest.json`, and `audit/final_decisions.json`.

### Mid-cost fifth twenty-task expansion 89 — 738 SHA candidates

- Safe adoptees: 0; LB-probe candidates: 0; gain counted: `+0.0`; tasks020/
  030/059/068/175/183/190/193/195/224/240/281/300/302/304/358/376/383/
  384/400 were rebased to 8006.61.  All twenty authority members are
  byte-identical between 8005.17 and 8006.61.
- 738 distinct non-authority SHAs produced 92 actual-cost profiles and two
  strict-lower finalists.  task193 cost100 is only154/266 known; task384
  cost179 is265/266.  Neither can enter fresh or an LB-probe pool.
- All 53 policy-only rejects were reopened under the task254/task379
  false-reject lesson.  Nineteen are schema-invalid; 33 private-lineage rows
  have no lower static floor.  The remaining task302 cost150 cloak profiles at
  truthful cost52,346, has76 declared/runtime shape contradictions, and fails
  default ORT session creation.  Thus no policy-only probe candidate remains.
- Evidence: `agent_mid20e_89/REPORT.md`, `result.json`,
  `probe_manifest.json`, `authority_rebase_proof.json`, and
  `audit/policy_reopen.json`.

### Expanded sixth twenty-task scan 90 — 1,025 SHA candidates

- Safe adoptees: 0; LB-probe candidates: 0; gain counted: `+0.0`; tasks012/
  075/107/131/157/159/182/185/201/218/225/251/263/280/330/361/364/370/
  382/392 were scanned across12,283 loose observations and23,712 ZIP members
  from1,261 archives.  All target authority members are byte-identical from
  8005.17 to8006.61.
- Standard and reopened giant/lookup/private paths produced49 actual-lower
  candidates,13 known-perfect x4, and6 truthful survivors.  The six survivors
  are all task185 lookup variants.  SHA819bc2f cost185 andd3da20d cost186 are
  known LB-black and only1/500 fresh on each seed; d21f1db cost273 is500/500
  on both fresh seeds but is also known partner-probe LB-black.  The remaining
  cost186/190/191 variants are each1/500 and the same local false-accept family.
- task107 cost706/638, task131 cost627/596, and task201 cost785/682/543 pass
  known x4 but are runtime-shape cloaks.  task251 cost709/582 passes disabled
  ORT but fails every default-ORT case.  Nothing remains for LB probing.
- Evidence: `agent_expand20f_90/REPORT.md`, `result.json`,
  `probe_manifest.json`, `inventory/candidate_inventory.json`, and `audit/`.

### Expanded seventh twenty-task scan 91 — task066 isolated probe

- Safe adoptees: 0; LB-probe candidates: 1; gain counted: `+0.0`.  Twenty
  targets were scanned across 285,747 loose ONNX files and 1,259 ZIPs,
  yielding 1,034 distinct non-authority SHAs and 99 isolated actual profiles.
- Four task066 payloads were strict-lower and known-complete x4.  Exact-SHA
  history proves the cost368/582/636 payloads LB-black.  The remaining new SHA
  `3a31ce1c...` is cost583 versus677 (`+0.149484` projected), truthful-shape,
  UB0/runtime-clean, and reached96.2%/95.8% on two independent fresh500 seeds.
  It remains `LB_PROBE_REQUIRED`; no local score is counted.
- Evidence: `agent_expand20g_91/REPORT.md`, `result.json`,
  `probe_manifest.json`, `winner_manifest.json`, and
  `evidence/authority_binding.json`.

### Expanded eighth twenty-task scan 92 — twelve isolated probe payloads

- Safe adoptees: 0; LB-probe candidates: 12 across tasks009/205/219/396;
  gain counted: `+0.0`.  A 1,374-SHA inventory produced43 ordinary and25
  policy-reopened strict-lower profiles.  Exact history and two fresh500 seeds
  reduced the initial19 probes to12 probes,5 exact known-black payloads, and2
  severe false-accept payloads.
- Best-per-task projected gain is `+0.086852` only if every isolated probe is
  LB-white.  task009 has one cost2616 candidate; task205 has cost1038/1041;
  task219 has cost1445/1453/1454 but all are LOW priority at87.2% minimum
  fresh; task396 has six cost961--1017 candidates, with cost1017 the
  safer98.4%-fresh fallback.  The task365 cost1337 SHA was matched to the
  direct705-pool black report and removed.
- Evidence: `agent_expand20h_92/REPORT.md`, `result.json`,
  `probe_manifest.json`, `audit/probe_classification.json`, and
  `audit/lb_history_exact_sha.json`.

### Authority alias drift during scans 91--92

- The canonical 8006.61 bytes remain intact in
  `submission_base_8006.61.zip` and
  `others/71403/lb_verified_8006.61/submission.zip`, both SHA-256
  `9085e2f795c0a73d44d27d712ad8fbaad67a3f37b1d8363a0f33305fbafa4118`
  and MD5 `c90b23f514fe47c36f1d032bd0924662`; both pass `unzip -t`.
- The auxiliary alias `submission_base_8006.61.zip` was temporarily replaced
  after the scans by a68-byte empty-ranking JSON, then externally restored to
  the exact canonical SHA above.  The loop did not repair or modify protected
  root files.  Every later lane binds to the exact SHA rather than trusting a
  path name, so the transient alias drift could not enter candidate results.
- On 2026-07-14 at 20:40 JST, the shared root `submission.zip` was externally
  changed to SHA-256 `50b3215030cf506f692af50e41203d805256a250bd29882cc777749767a350c6`.
  It differs from the fixed checkpoint at 37 tasks, all drawn from the new
  71405 pool.  It was initially quarantined while unclassified, then the
  matching `best_score.json`, campaign ledger, immutable alias, MD5, and user
  confirmation established it as the LB-verified8008.14 authority.  This loop
  did not write or restore the protected root.

### Isolated LB probe queue on 8006.61

- This queue is now historical evidence only.  Its ZIPs must not be submitted,
  because they would revert the37 new LB-white members fixed at8008.14.
- Ten single-replacement archives are staged under
  `lb_probe_queue_8006_61/`.  Every archive has exactly400 members, preserves
  canonical order, passes CRC, and differs from SHA `9085e2f7...` at exactly
  one named task.  They are probes, not fixed winners.
- The strongest probe is the exact true-rule task192 rebuild cost1307
  (`+0.207878`, fresh10000/10000 plus exhaustive163-case sign proof).  The
  safer approximate queue is task046 cost627 (`+0.006359`, fresh6000/6000),
  task396 cost1017 (`+0.001965`, fresh98.4%), task205
  cost1038 (`+0.003846`, fresh97.4%), and task009 cost2616 (`+0.001146`,
  fresh95.6%).  Higher-risk probes are task310 cost501 (`+0.121988`, one fresh
  miss in10000 and lookup/giant lineage), task066 cost583/562
  (`+0.149484`/`+0.186169`), and task396 cost961 (`+0.058603`).  task205
  cost1041 is retained only as a fallback.
- task219 was deliberately not packaged because all three surviving SHAs have
  a fresh minimum of87.2%, below the current90% threshold.  Evidence and exact
  archive/candidate hashes are in `lb_probe_queue_8006_61/MANIFEST.json` and
  the recommended order is in `lb_probe_queue_8006_61/README.md`.

### Expanded ninth and tenth twenty-task scans 94--95

- Safe adoptees: 0; LB-probe candidates: 0; gain counted: `+0.0`.
- Scan94 reduced seven delta SHAs to six apparent savings, but task062/170/
  245 were shape-invalid, task308/338 failed default ORT, and task377 used an
  unsupported TopK.  The task297 negative-padding repair still cannot fit
  below the cost370 authority.
- Scan95 examined71 delta SHAs.  The only official-lower task014 cost360 and
  task037 cost320 payloads failed truthful/default runtime gates; task036
  regressed and task124 reproduced the allocator SIGSEGV lineage.
- Evidence: `agent_expand20i_94/` and `agent_expand20j_95/`.

### 71405 new-pool audit 96--99

- The new `others/71405` pool contains77 ONNX files across71 tasks.  Official
  profiling found46 preliminary strict-lower/static-clean payloads; this first
  stage deliberately did not imply admission.
- Priority audit97 retained task046 cost627 and task066 cost562 only as
  isolated probes.  task013 cost357, task023 cost1479, task044 cost1076, and
  task069 cost524 failed true-rule, truthful-shape, or default-runtime gates.
- Tail audit99 retained task310 cost501 as a high-risk isolated probe
  (known x4, truthful, fresh9999/10000).  task396 cost960/974 remain unbuilt,
  low-priority alternatives because fresh is971/1000 and the task has
  version-dependent LB-black history.  Nine other tail candidates were
  rejected for shape/runtime/oversize-lookup faults.
- Subsequent LB bisect over the42-candidate clean merge proved37 exact payloads
  white and five tasks black:023/198/201/208/396.  The white37 are now fixed
  in8008.14, including locally risky task046/066/096/138/157/209/310.  Exact
  LB classification overrides the earlier heuristic rejection only for those
  exact SHAs; it does not whitelist future variants.  Evidence:
  `root_71405_96/`, `agent_71405_priority_97/`, and
  `agent_71405_tail_99/`, `agent_71405_mid_98/`, and
  `agent_71405_residual_100/`.

### Sound task192/task344 rebuild 93

- task192 was rebuilt from the decoded true rule at cost1609 ->1307,
  projected `+0.207878433368`.  It is known265/265 in four configurations,
  fresh10000/10000 over two seeds, runtime-shape truthful, standard-domain,
  Conv-UB0, and passes an exhaustive163-case local sign proof.  Maximum
  Einsum arity is12; no stored-example lookup is used.
- The exact candidate SHA is
  `c3cbaf44d962ca72e15514da1b32c121ee489d153ef39d38b7101f09576e92b6`.
  It is the highest-confidence isolated LB probe, not yet a fixed winner.
- task344 cost188 no-S rewrite scored0/266 known and was rejected; authority
  cost197 remains.
- The task192 probe was rebuilt on the immutable8008.14 checkpoint.  Current
  p00 SHA-256 is
  `79382b86e280462df208376806f26db19926b7ebe8f9fffedbdb2e0443658d7d`
  and differs only at task192 while preserving all37 new fixed members.
- Evidence: `agent_sound192_344_93/` and
  `lb_probe_queue_8008_14/p00_task192_sound_cost1307.zip`.

### Isolated LB probe queue on 8008.14

- Seven archives are staged under `lb_probe_queue_8008_14/`: task192 cost1307,
  task349 cost3556, task205 cost1038/1041, task009 cost2616, and task396
  cost1017/961.
- Every archive has400 unique members, exact authority order/comment, CRC pass,
  Conv-family short-bias0, and exactly one named task difference from SHA
  `50b32150...`.  This was checked across2,800 model instances including the
  authority and six probes.
- Obsolete task046/066/310 probes were removed from the live queue because
  their exact payloads are already fixed in8008.14.  The old8006.61 queue is
  never a submission source.
- Evidence and exact hashes: `lb_probe_queue_8008_14/MANIFEST.json` and
  `lb_probe_queue_8008_14/README.md`.

### task349 exact-equivalence golf 104

- The LB-white authority cost3564 payload has an exhaustive11-row identity
  `top_offset = hstart_offset + hend_offset - 1`.  Eliminating the redundant
  table, reusing a broadcast scalar zero, and gathering an already computed
  signature equality reduces official cost to3556 (`+0.002247191957`).
- The rewrite has an all-input integer proof and preserves raw output on two
  independent5,000-case seeds in both disabled/default ORT modes:
  20,000/20,000 exact equality, runtime errors0.  Checker, strict/data_prop,
  known correctness, margin, domains, nonfinite, and Conv-UB gates pass.
- Candidate SHA-256:
  `179bbed5bd313a1f6ec62f573fd725ab71ff55a9509daaceff3f40274ac514c7`.
  It is staged as `p06_task349_exact_cost3556.zip`, not counted until exact-SHA
  LB confirmation.
- Evidence: `root_task349_exact104/REPORT.md`, `winner_manifest.json`, and
  `lb_probe_queue_8008_14/p06_task349_exact_cost3556.zip`.

### Exact re-golf of the 37 new LB-white payloads 102

- All37 exact SHAs were scanned for dead initializers/value-info, no-op and
  Identity removal, CSE, optional-output removal, constant folding, and exact
  Add/Mul absorption.  Sixty-nine candidates produced no admissible winner:
  40 failed structure/schema/UB and29 were not strict-lower after profiling.
- Direct profiling of13 ambiguous candidates also admitted0.  The apparent
  task089 1340->1171 and task165 587->546 savings are incorrect allocator/
  runtime-shape variants and were rejected.
- Evidence: `agent_8008_exact_white102/REPORT.md`, `result.json`, and `audit/`.

### Staged cheap-task pool 71407

- `others/71407/` contains three active root `*.onnx` files: SOUND task158
  cost7525, SOUND task192 cost1197, and exact-equivalent task349 cost3556.
  Combined projected gain is `+0.305060135353`; none is added to verified LB
  gain before exact-SHA LB.
- Five policy90/95 approximations for tasks009/205/396 are stored only as
  `*.onnx.quarantine` under `PROBE_ONLY_DO_NOT_MERGE/`; ordinary ONNX globbing
  cannot merge them accidentally.
- All three active candidates were reprofiled directly against their authority
  tasks and pass
  correctness, strict-lower, checker, and Conv-UB gates. The three authority
  payloads are unchanged in8009.46, so all candidates were rebased without
  altering their proofs. Root submission and score files were not modified.
- Evidence: `others/71407/README.md`, `MANIFEST.json`, and
  `REBASE_8009_46.json`.

### task192 exact relation-factor golf 111

- The prior generator-SOUND polynomial candidate cost1307 stored a
  `relation[2,10,10]` tensor. Because the selected color is one-hot, its
  contraction is identically the two-row mask `[all_colors, selected]` for
  every input. Constructing those rows directly removes190 net parameters at
  the cost of80 bytes of intermediate memory.
- New candidate SHA
  `40244ab462644481407ebb7200984dfdff1475c0d8e6ff731ba2d588ec92ea09`
  costs1197 versus authority1609, projected `+0.295794441434`.
- Checker/strict/data_prop/truthful shapes/UB0 pass; known265/265 in four
  configs; both independent5000-case seeds are perfect in disabled and
  default ORT with runtime errors0 and cross-mode threshold equality10000/
  10000. The readable rule and exhaustive163-case sign proof also pass.
- It replaces the older cost1307 task192 file in `others/71407/`; not counted
  as LB-fixed until exact-SHA confirmation.
- Evidence: `root_task192_exact111/REPORT.md`, `fresh_dual.json`, and
  `winner_manifest.json`.

### SOUND rebuild of tasks216/255 103

- No strict-lower exact candidate was found.  task216 history contained58
  unique SHAs and24 apparent sub-1499 payloads; none survived.  The authority
  itself has53 runtime-shape contradictions and errors on about one third of
  legal fresh cases.  The truthful exact control is fresh10000/10000 in both
  ORT modes but costs31511, far above1499.
- task255 history contained60 unique SHAs and11 sub-1307 payloads; none
  survived.  A finite generator witness produces one identical legal input
  with two different outputs (15 differing cells), proving that no
  deterministic input-only ONNX can be universally exact.  The authority is
  only95.16%/94.44% fresh.
- Evidence: `agent_sound216_255_103/REPORT.md`, `RESULT.json`,
  `fresh_two_seeds.json`, and `task255_ambiguity.json`.

### SOUND task285 reconstruction 105

- Current 8008.14 authority cost is **8623**. The smallest complete SOUND
  reconstruction found costs **14685**, or 6062 more than the authority.
- The SOUND model is known-exact in both ORT modes, shape-truthful, UB0, and
  bitwise-equal to the exact source rebuild, but it cannot satisfy the strict
  lower-cost gate. Fifteen optimizer passes produced no lower SOUND graph.
- Safe adoptees/probes: 0; gain counted: `+0.0`.
- Evidence: `agent_sound285_105/REPORT.md`, `final_audit.json`, and
  `winner_manifest.json`.

### task191 exact shape rewrite 109

- The 8008.14 authority costs **3436**. Removing its shape Identity with the
  original scalar target makes two-axis `CenterCropPad` fail to load; changing
  the target to explicit `[30,30]` instead makes a one-axis `CenterCropPad`
  fail to load.
- The apparent Identity is part of the authority's dynamic scalar reuse across
  incompatible axes arities, so neither static rewrite is valid.
- Safe adoptees/probes: 0; gain counted: `+0.0`.
- Evidence: `root_task191_exact109/REPORT.md`, `identity_result.json`, and
  `identity_shape2_result.json`.

### SOUND tasks319/367 audit 107

- task319 authority cost1003 is known-perfect but only4885/5000 and4868/5000
  fresh; the generator also produces identical valid inputs with different
  outputs, so universally exact deterministic ONNX behavior is impossible.
- task367's truthful exact control was deduplicated from3915 to3913 and passes
  known/fresh20,000 dual-ORT with truthful shapes and UB0, but remains far
  above the2179 authority.
- Safe adoptees/probes: 0; gain counted: `+0.0`.
- Evidence: `agent_sound319_367_107/REPORT.md` and `result.json`.

### task344 deep local-rule audit 110

- Five new clean/no-S/non-giant structures cost170--188, but all fail the
  complete-known gate; the best cost170 model reaches only224/266. Historical
  lower candidates are also inexact or fresh-imperfect.
- The minimum verified clean SOUND control costs910 versus authority197.
- Safe adoptees/probes: 0; gain counted: `+0.0`.
- Evidence: `agent_task344_deep110/REPORT.md`, `final_audit.json`, and
  `winner_manifest.json`.

### task158 SOUND Scatter repair 108 / independent review 113

- The current authority cost7578 payload has invalid object slots with
  arbitrary out-of-range Scatter indices. The repaired candidate forces only
  invalid `obj_base` to -1 and keeps their updates at zero. With
  `p_mag in [0,12.5]` and local offsets in `[0,52]`, every invalid index is
  formally in `[-1,649]`, inside the axis650 ONNX range; zero updates under
  `reduction=max` are inert.
- Candidate SHA
  `9d9a3ca8fb39856125925ea464ed1cc80f0301bd785ff7b60da37bd1c2b6b9d1`
  costs7529 versus7578, projected `+0.006487081728`.
- Author audit: known266 and two independent5000-case seeds are perfect/raw
  equal to the trusted cost7612 reference in both ORT modes, runtime errors0.
  Independent review used two different2000-case seeds and reproduced all
  structural, formal-bound, strict-shape, UB0, and raw-equivalence results.
- It is staged in `others/71407/task158.onnx`; not counted until exact-SHA LB.
- Evidence: `agent_task158_current_108/REPORT.md` and
  `agent_review_task158_113/REPORT.md`.

### task158 exact regolf 114

- Starting from the independently reviewed cost7529 SOUND repair, removing a
  uniform +146 Conv bias, doubling the bounded integer Conv sum, and shifting
  all phase/role thresholds preserves every comparison, TopK ordering, and
  tie. Reachable magnitudes remain exactly representable in float16.
- Two shifted constants then alias existing equal typed initializers. Final
  SHA `127984c6807d84559bbf74fd58e3b09a66459d142cef65a8635647e64f5e59fd`
  costs7525 (memory6662, parameters863), projected authority gain
  `+0.007018501961`.
- Known266 and two fresh2000-case seeds are raw-equal to the trusted SOUND
  reference in both ORT modes; checker/strict/truthful/UB0/runtime gates pass.
- It supersedes only the staged task158 candidate in `others/71407`; root
  submission and score ledgers remain unchanged.
- Evidence: `agent_task158_regolf_114/REPORT.md` and `winner_manifest.json`.

## LB-verified supersession: 8008.14 -> 8009.46

- The new immutable archive adds24 LB-white task payloads:
  029/031/036/075/079/091/092/124/137/153/159/169/178/228/234/264/325/
  344/357/387/388/392/397/398.
- The25-candidate merge first scored7990.09; individual probes confirmed185
  as the sole black member and task091/task344@137 as white. The white-only
  archive scored8009.46.
- ZIP SHA-256 is
  `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`,
  MD5 `2dc6d412ddd8bd3102f42775155e4a38`; it is byte-identical to protected
  `submission.zip`, has400 unique members, and preserves order/comment.
- Verified cumulative progress from8004.42 is `+5.04`; `14.96` remains to
  8024.42.

## LB-verified supersession: 8006.61 -> 8008.14

- A79-file directory scan produced43 candidate winners.  After excluding a UB
  task251 payload, the42-candidate merge scored7917.91.  Seven-group bisect and
  exact subset-sum arithmetic identified five black tasks with matching score
  deficit:023/198/201/208/396.
- The remaining37 exact payloads scored8008.14, an LB-verified `+1.53` over
  8006.61.  The champion SHA-256 is
  `50b3215030cf506f692af50e41203d805256a250bd29882cc777749767a350c6`,
  MD5 `db4da5cc59186b26572a380725bc2fdf`.
- Verified cumulative progress from the requested8004.42 origin is `+3.72`;
  `16.28` remains to the8024.42 target.

## LB-verified supersession: 8005.17 -> 8006.61

- Wave A: 8005.17 + tasks158/254/267/323 = 8006.47.  The large gains at
  task254 (76->42, +0.593) and task267 (60->30, +0.693) were individually
  probed LB-white.  This overrides the earlier local Wave38 rejection of the
  task254 cost42 payload.
- Wave B: a 705-candidate deduplicated sweep yielded 23 local winners.  After
  excluding known black/UB/policy95 cases, the 16-candidate merge fell to
  7792.73.  Four-way bisect plus exact subset arithmetic classified twelve
  black tasks (018/048/112/134/168/198/233/251/277/286/365/366) and four white
  tasks (013/070/158/379).  The white-only merge is LB 8006.61.
- The final 8006.61 archive differs from 8005.17 at seven unique members:
  task013/070/158/254/267/323/379.  Protected root score pointers and archive
  are externally synchronized and are not modified by this loop.
- Operational lesson: local policy90/policy95 is useful for screening but did
  not predict private correctness.  New savings are kept as probe candidates
  until LB-white; the immutable 8006.61 payloads are never replaced by a
  locally preferred regression.

### Binary Add/Sum carrier scan 141

- All binary `Add`/two-input `Sum` carrier substitutions in the400 immutable
  8009.46 payloads were profiled in both directions.
- 113 profiles were checked; none was strictly cheaper, so no semantic/runtime
  admission was needed.
- Safe adoptees/probes: 0; gain counted: `+0.0`.
- Evidence: `root_add_sum_scan_141/REPORT.md`, `scan.json`, and `scan.py`.

### Reduce nonnegative/binary carrier scan 143

- All49 `ReduceL1`/`ReduceSumSquare` nodes in the400 immutable8009.46
  payloads were profiled as shape-identical `ReduceSum` replacements.
- None was strictly cheaper; the ORT memory penalty was neutral or positive in
  every case, so no domain-semantic admission was needed.
- Safe adoptees/probes: 0; gain counted: `+0.0`.
- Evidence: `root_reduce_nonnegative_scan_143/REPORT.md`, `scan.json`, and
  `scan.py`.

### Attribute and Boolean carrier scans 144--147

- Nonpositive `Mul` to `LeakyRelu` and binary affine to `HardSigmoid` scans
  found no structurally eligible chain in all400 payloads.
- Four `Where(bool,1,0)` Cast rewrites and16 proved-nonnegative
  `Greater(x,0)` Cast rewrites passed structural gates, but none was strictly
  cheaper under official-like runtime profiling.
- Safe adoptees/probes: 0; gain counted: `+0.0`.
- Evidence: `root_leaky_nonpositive_scan_144/REPORT.md`,
  `root_hardsigmoid_affine_scan_145/REPORT.md`,
  `root_bool_where_cast_scan_146/REPORT.md`, and
  `root_positive_cast_scan_147/REPORT.md`.

### Exact sparse-storage and repeated-slice factor scans 149/151/152

- Sixty-five direct sparse-initializer candidates all failed strict ONNX type
  or rank inference.  The alternative sparse-Constant form passed full/strict
  gates for256/262 attempts but materialized the full tensor as scored memory;
  even task158's650-cell zero seed worsened cost by one.
- Every Einsum-only initializer axis was checked for exact one-hot selector x
  unique-slice factorization; no factor used fewer elements than its source.
- Safe adoptees/probes: 0; gain counted: `+0.0`.
- Evidence: `root_sparse_nondot_scan_149/REPORT.md`,
  `root_sparse_constant_scan_151/REPORT.md`, and
  `root_exact_unique_factor_scan_152/REPORT.md`.

### task333 exact shared-sign absorption re-admission 81/153

- Rebased candidate SHA `0628a573302f...` against the byte-identical current
  task333 authority SHA `5bb4ddf301f1...`; official cost is423->421, projected
  gain `+0.004739345364`.
- This SHA is distinct from the historical cost627 private-zero SHA
  `a76f2a80ea58...`.  `GE=[1,-1]` is absorbed into `HC`, while the shared
  second use is compensated in `GHHT`; `GE^2=1` proves every Einsum monomial
  unchanged for every input.
- Complete changed-factor support80/80, known265/265 in four configs, and8000
  fresh whole-model runs are raw-identical, with runtime/nonfinite/margin
  failures all zero.  The inherited35-input Einsum is accepted only under
  this all-input termwise proof.
- Staged as `others/71407/task333.onnx`; protected root ZIP and score ledgers
  remain unchanged. Evidence: `agent_task333_finite81/REPORT.md` and
  `winner_manifest.json`.

### Exact deep audit 148 — task014/239/255

- Twelve archive SHA values plus optimizer/initializer/CSE and seven task014
  algebraic variants produced no strict-lower admissible candidate.
- task239 is runtime-shape truthful but has no lower graph. task014/task255
  retain authority shape contradictions; task255 also has a finite
  same-input/different-output witness, so no deterministic exact rebuild
  exists for the full generator relation.
- Safe adoptees/probes:0; gain counted:`+0.0`. Evidence:
  `agent_high014_239_255_148/REPORT.md` and `winner_manifest.json`.

### Attribute-carrier exact scan 155

- Scanned728 initializer-backed CastLike nodes across all400 payloads for
  exact `CastLike(x,ref)->Cast(x,to=dtype(ref))`; no eligible scalar PRelu
  attribute absorption existed.
- Five preliminary lower profiles were found. Tasks071/133/216/285 exposed
  3/30/53/57 runtime shape contradictions and fail an ORT configuration with
  every known case or at session creation. Task388 is raw-equal and known
  266/266 in four configurations, but competition actual cost regresses
  305->1599 instead of the declared85->84.
- Safe adoptees/probes:0; gain counted:`+0.0`. Evidence:
  `root_attr_constant_scan_155/REPORT.md`, `scan.json`, and `audit.json`.

### Exact deep audits 150/153

- task025/131/363 produced no current-only or CastLike-subset strict-lower
  winner. Historical task025 lows are0/266; task131 lows are TfIdf lookup and
  raw-different despite boolean fixtures; task363's generator is non-injective.
- task044/117/330 also produced no winner. The authorities retain2/10/38
  runtime shape contradictions respectively; all truthful controls cost more.
- Safe adoptees/probes:0; gain counted:`+0.0`. Evidence:
  `agent_high025_131_363_150/REPORT.md` and
  `agent_high044_117_330_153/REPORT.md`.

### Optional-default initializer scan 158

- Scanned equal Split sizes, default Slice axes/steps, and zero/all-axis Pad
  optional inputs across all400 payloads. Forty variants were profiled.
- Twenty-four fail full checker/strict data propagation after omission; none
  of the remaining sixteen is strict-lower.
- Safe adoptees/probes:0; gain counted:`+0.0`. Evidence:
  `root_optional_default_scan_158/REPORT.md` and `scan.json`.

### Unary carrier scan 159

- Scanned `Div(1,x)->Reciprocal`, `Sub(0,x)`/`Mul(-1,x)->Neg`, and
  `Max(0,x)->Relu` across all400 payloads.
- The only36 eligible sites are unsigned `Sub(0,x)` or already malformed
  lineages; every Neg candidate fails full/strict ONNX type or shape inference.
- Safe adoptees/probes:0; gain counted:`+0.0`. Evidence:
  `root_unary_carrier_scan_159/REPORT.md` and `scan.json`.

### task086 shared negative-PRelu scan 161

- Exhausted all31 nonempty subsets of five shared slope=-1 PRelu nodes with
  both exact Abs and LeakyRelu replacements, 62 profiles total.
- All pass full/strict structure, but no subset is cheaper; replacing all five
  and dropping the shared initializer still loses to added runtime memory.
- Safe adoptees/probes:0; gain counted:`+0.0`. Evidence:
  `root_preluneg_scan_161/REPORT.md` and `scan.json`.

### Zero optional-bias scan 162

- Scanned all400 payloads for all-zero optional biases on Conv,
  ConvTranspose, QLinearConv, and Gemm. Only two eligible sites exist, both
  shared task233 QLinearConv scalar biases.
- Omitting either passes structural analysis, but the shared initializer stays
  live and official cost remains7308->7308.
- Safe adoptees/probes:0; gain counted:`+0.0`. Evidence:
  `root_zero_bias_scan_162/REPORT.md` and `scan.json`.

### Schema-default attribute scan 164

- Compared every explicit node attribute in all400 payloads with the active
  ONNX schema default and profiled combined plus isolated removals:1,171
  variants total.
- Sixty-five inherit structural/strict failures. None of the1,106 valid
  variants is cheaper because schema-default attributes are not counted by the
  competition memory/parameter profiler.
- Safe adoptees/probes:0; gain counted:`+0.0`. Evidence:
  `root_schema_default_scan_164/REPORT.md` and `scan.json`.

### CastLike independent audit 156

- Independently reproduced the five lower-profile CastLike leads from scan155.
  Tasks071/133/216/285 fail disabled-ORT operational gates; task388 is raw
  equal but its competition actual cost regresses305->1599.
- Safe adoptees/probes:0; gain counted:`+0.0`. Evidence:
  `agent_castlike_exact_156/REPORT.md` and `result.json`.

### Exact deep audit 157 — task187/191/319

- Exhausted397 current CastLike/Identity subsets plus322 unique historical
  SHAs. Four historical strict-lower leads all fail default ORT or raw equality
  and lack an all-input proof.
- Safe adoptees/probes:0; gain counted:`+0.0`. Evidence:
  `agent_high191_187_319_157/REPORT.md` and `deep_audit.json`.

### Scalar-Pow exact scan 163

- Scanned18 Pow nodes in11 tasks. No node has a scalar initializer exponent
  equal to2,0.5,or1; the only scalar-initializer sites are task250 exponent
  0.412499994.
- Safe adoptees/probes:0; gain counted:`+0.0`. Evidence:
  `agent_pow_exact_163/REPORT.md` and `result.json`.

### Consecutive exact-chain scan 167

- Scanned all400 payloads for consecutive idempotent unary operations, double
  involutions, same-target Cast, Reshape chains, and composable Transposes.
- No eligible single-use consecutive site exists in the current authority.
- Safe adoptees/probes:0; gain counted:`+0.0`. Evidence:
  `root_unary_chain_scan_167/REPORT.md` and `scan.json`.

### Unit DequantizeLinear scan 168

- Scanned all nine DequantizeLinear nodes in the authority; none has an
  all-one initializer scale, so no all-input exact Cast replacement exists.
- Safe adoptees/probes:0; gain counted:`+0.0`. Evidence:
  `root_dequant_cast_scan_168/REPORT.md` and `scan.json`.

### Inference/default exact scan 165

- Across all400 payloads there are no Dropout sites and no Clip bounds equal
  to the infinite defaults. Three explicit Reshape allowzero=0 sites yielded
  four variants, all exact cost ties.
- Safe adoptees/probes:0; gain counted:`+0.0`. Evidence:
  `agent_inference_defaults_165/REPORT.md` and `scan.json`.

### task118 true-rule rebuild 166

- Proved the generator relation non-injective with one all-gray input mapping
  to two different cyan-plus outputs. Healthy observable reconstructions pass
  known267/267 x4 and fresh96%+, but cost9142/51350 versus authority3665.
- The authority itself is fresh86.38% with37 shape contradictions, so its
  defects were not inherited into a new candidate.
- Safe adoptees/probes:0; gain counted:`+0.0`. Evidence:
  `agent_task118_rebuild_166/REPORT.md` and `result.json`.

### task192 duplicate-constant regolf 170 / staged optimizer scan 172

- All11 staged SOUND candidates were profiled through319 conservative
  optimizer variants. The sole strict-lower result is task192's byte-identical
  float32 `[0,1]` initializer alias.
- New task192 SHA `51a7d65491f3...` reduces the staged exact model1197->1195
  and the immutable authority1609->1195. The all-input proof is literal tensor
  identity; graph computation/order otherwise does not change.
- Full/strict/truthful/UB0/no-lookup gates pass, known265/265 in four configs,
  fresh5000/5000 on two seeds, runtime/nonfinite failures0, finite sign proof
  163/163.
- Replaced `others/71407/task192.onnx`; root submission and score ledgers stay
  unchanged. Incremental projected gain:`+0.001672241192`; checkpoint staged
  cumulative gain:`+0.322244592223`. Evidence: `root_task192_hist_170/REPORT.md`,
  `audit/task192_exact_poly.json`, and `root_stage_optimizer_scan_172/REPORT.md`.

### task192 exact shared-basis factorization 178

- Replaced two separate `[all,selected]`/`[background,selected]` Concat
  outputs with one shared `[nonzero,background,selected]` basis. Three 2x3
  integer maps recover the center masks, neighbor masks, and output route
  inside the final Einsum exactly.
- The safe ArgMax+OneHot version costs1149 (memory208, params941), versus the
  prior staged1195 and immutable authority1609. SHA is
  `19fbdce89a5c...`; authority-relative projected gain is
  `+0.336720869144`, an incremental `+0.039254186517`.
- Full/strict/truthful/UB0/no-lookup gates pass, known265/265 in four ORT
  configurations and fresh5000/5000 on two seeds pass with runtime/nonfinite
  failures0. Every valid-grid product and accumulation is a small exactly
  representable integer, so the163-case local sign proof remains exhaustive.
- A lower Hardmax cost1138 control was rejected because the SOUND gate treats
  Hardmax as lookup. Only the safe ArgMax+OneHot SHA is staged. Current staged
  cumulative gain:`+0.361498778740`. Evidence:
  `root_task192_basis_178/build_safe_argmax.json` and
  `audit/task192_exact_poly.json`.

### task328 exhaustive-support exact coefficient shave 175

- The immutable authority costs558 (memory200, params358). Existing exact554
  first removes four parameters through `e[4]=sum_t J[t,a,a]`; the new
  candidate removes one more scalar by replacing `one=1` with existing
  `ninvB=-1/3` and scaling `CoreB[:,:,0]` by -3. The serialized float32
  compensation product is exactly1.
- Candidate SHA `cc2718047fec...` costs553 (memory200, params353), projected
  gain `+0.009000960859`. Full/strict/truthful/static/standard/finite/UB0
  gates pass; known267/267 in four ORT configurations.
- Although this is private-zero lineage and positive margins can be tiny, the
  full71,136-state generator support is exactly reduced by nonzero-color
  permutation equivariance to143 representatives. Every representative passes
  in disable/default x threads1/4 with wrong/error/nonfinite/false-positive0;
  two fresh10000 seeds map entirely into the certified support and see all143
  orbits. This satisfies the user's explicit guaranteed-private-zero exception.
- Staged as `others/71407/task328.onnx`; cumulative projected gain is now
  `+0.370499739599`. Evidence: `agent_task328_exact_175/REPORT.md`,
  `final_audit.json`, and `winner_manifest.json`.

### Exact deep audit 160 — task002/012/107

- task002 has no lower historical/current graph, a66-input giant Einsum, and a
  generator same-input/different-output witness. task012's healthy one-node
  authority cost710 has only a cost500 frontier that fails30/265 known cases.
- task107's cost638 frontier is giant/nonfinite with13 shape mismatches and raw
  authority equality0/266 despite threshold-known agreement; the healthy-shape
  control costs1286 versus664.
- Safe adoptees/probes:0; gain counted:`+0.0`. Evidence:
  `agent_high002_012_107_160/REPORT.md` and `result.json`.

### task192 sparse-initializer audit 173

- Converted the30x30 dense adjacency (900 values,88 nonzero) to both legal COO
  index encodings across ten metadata/type variants. ORT can execute eight
  variants bit-identically, but every one fails the official full/strict type
  gate because Einsum accepts tensor, not sparse_tensor.
- Constant(sparse_value) controls pass and are bit-identical but materialize
  dense runtime memory, regressing cost1195->3983. The theoretical cost383
  SparseInitializer graph is unscorable and not admissible.
- Safe adoptees/probes:0; gain counted:`+0.0`. Evidence:
  `agent_task192_sparse_173/REPORT.md` and `probe_results.json`.

### task343 exact period compiler search 169

- The clean period/reflection compiler cost178 passes known266/266 x4,
  fresh10000/10000 x4, and an exhaustive3,144,312-state normalized generator
  proof with zero mismatches and positive margin1.
- Authority cost173 is fresh4975/5000; both historical cost172 classifiers have
  deterministic fresh counterexamples. Exhaustive affine threshold AND/OR and
  z11 relation searches found no exact cost<=172 construction.
- Safe strict-lower adoptees:0; gain counted:`+0.0`. Evidence:
  `agent_task343_exact_169/REPORT.md`, `finite_rule_proof.json`, and
  `threshold_search.json`.

### task257/task310 mask-factor absorption 174

- task257's global feat masking gives114->84 but breaks source-coordinate
  features; task310's global P0/P1/P2 masking gives501->471 but breaks repeated
  source periods. Both have explicit dual-ORT threshold counterexamples.
- Output-only cloned factors are algebraically exact and known/synthetic raw
  bit-identical in both modes, but cost144 and531 respectively, above114/501.
- Safe strict-lower adoptees:0; gain counted:`+0.0`. Evidence:
  `agent_mask_absorb_174/REPORT.md` and `results.json`.

### task344 current-authority compact-factor audit 171

- Rebased against the immutable8009.46 authority: task344 cost137, member SHA
  `05bedf3ca834...`. The compact `G = H.T @ S @ H` Einsum candidate costs132,
  SHA `c5272a42bee4...`, for a potential `+0.037179003242`.
- It passes official/team validation, full/strict/truthful/UB0 gates, known
  266/266 in four ORT configurations, and two fresh10000 runs with zero
  candidate-authority sign differences. Candidate and authority share the
  same30 true-rule misses; sampled maximum raw delta is `5.32e-5`.
- Float32 factor residual is `3.30e-8` and the derived global real-logit error
  bound is `1.24802e-4`. A subsequent full-support search found an explicit
  generator-reachable10x10 witness: authority channel8@(5,5) is
  `+0.000245497`, while cost132 is `-0.000113848` in all four ORT modes.
  The candidate is therefore rejected, remains quarantined only as evidence,
  and counted gain remains `+0.0`. Evidence:
  `agent_task344_margin_184/REPORT.md`, `audit/margin_counterexample.json`,
  and `agent_task344_rebase_171/audit/final_audit.json`.

### task246/task335/task348 selector-factor absorption 177

- task246/task335 selector-bond fusion is raw-bitwise exact on all266 known
  cases but ties cost109; `S@C` costs116 and `C^2` costs139.
- task348's exact `C1/C2 @ D` absorption is raw-bitwise exact on all265 known
  cases but regresses130->148. Its apparently low-rank `D` is exact rank3 and
  the factored representation uses99 elements versus90 direct.
- All controls pass full/strict, truthful shape, and dual-ORT gates. There is
  no strict-lower candidate, so SAFE/PROBE_ONLY counts and gain are all0.
  Evidence: `agent_selector_absorb_177/REPORT.md`, `result.json`, and
  `audit/selector_factor_audit.json`.

### task398/task324 selector absorption 176

- task398's global Q4 compensation gives346->343 but is wrong; shared-use
  parity proves the same factor must appear with both exponent0 and1. The
  all-input exact cloned-K construction costs367 and is rejected.
- task324 has a formal all-input one-hot identity that removes two parameters,
  439->437, with known266 and fresh10000 raw equality. It is nevertheless
  ineligible: default ORT cannot build the inherited TopK graph, six runtime
  shapes are untruthful, and the candidate emits3,636,153 nonfinite fresh
  values. Truthful repair has a >54KB intermediate floor.
- Safe/probe adoptees0; gain counted:`+0.0`. Evidence:
  `agent_selector_absorb_176/REPORT.md`, `screen_results.json`, and
  `deep_audit_324.json`.

### Shared-Concat to single-basis audit 181

- Exhausted72 shape classes and192 concrete selector-anchor expansions for
  task013/055/099/281, the four authority/staged models whose overlapping
  Concat outputs feed only Einsum.
- Every rewrite passes full/strict and threshold equivalence, but the best
  actual costs regress: task013 356->378, task055 234->270, task099 398->582,
  and task281 161->213. task013/281 also inherit baseline shape/nonfinite
  defects; no new defect is accepted.
- Strict-lower candidates0, so known4/fresh gates are not run and gain is0.
  Evidence: `agent_shared_concat_181/REPORT.md`, `screen_results.json`, and
  `concrete_costs.json`.

### All-authority Einsum precontraction inventory 182

- Scanned400/400 members and found524 locally smaller two-initializer
  contractions across60 tasks, including172 integer-valued pairs.
- None has both source factors single-use. Graph-level reuse reverses most
  local savings; e.g. task304's shared H/SF/SG factorization uses102 elements
  versus120 after full precontraction.
- The largest integer lead is task328 CoreB/e. It was forwarded to lane175,
  whose independent exhaustive-support winner is recorded above. Remaining
  integer families task074/200/211 were forwarded to lane183.
  Evidence: `root_einsum_precontract_scan_182/REPORT.md` and `scan.json`.

### Full-archive exact fusion scan 185

- Ran21 exact ONNX optimizer pass sets across all400 immutable8009.46 members
  (8,400 task/pass profiles), including MatMul/Gemm, Conv bias/BN/Pad,
  CSE, shape/slice, Einsum/MatMul, and combined fixed-point rewrites.
- The only strict-lower profiles are known failure lineages:
  task039/111/122/183 dead allocator witnesses, task089 allocator cloak,
  task165 CastLike CSE allocator failure, and task264 initializer alias with
  default-ORT/44-shape failures.
- The extra11 previously unscanned pass families produced no new strict-lower
  task. Safe adoptees0; gain counted:`+0.0`. Evidence:
  `root_full_fusion_scan_185/REPORT.md` and `scan.json`.

### Integer Einsum graph-level precontraction 183

- Exhausted98 graph classes and140 concrete candidates for task074/200/211,
  including all initializer uses, selector absorption, channel permutations,
  and task211 left/right aliases.
- No strict-lower graph exists in the enumerated exact families: task074
  135->135, task200 346->348, task211 66->66. All98 profiled classes execute
  with equal threshold signs;89 are also raw-equal, while the remainder differ
  only by task211 float contraction order and are nonlower.
- Safe adoptees0; gain counted:`+0.0`. Evidence:
  `agent_integer_precontract_183/REPORT.md` and `winner_manifest.json`.

### task226 exhaustive-support Boolean carrier shave 187

- Replaced two `Cast(a) AND NOT Cast(b)` row conditions with `Greater(a,b)`.
  Every valid one-hot probe is exactly0 or1, so the two-bit truth table is an
  exact identity and removes two Boolean bytes: cost372->370.
- Candidate SHA `aebca4b2e7c...`, projected gain `+0.005390848635`.
  Full/strict/truthful/UB0/standard/no-lookup gates pass; known133/133,
  complete17x8=136 generator states, and fresh2x5000 all pass in four ORT
  modes with raw authority equality and error/nonfinite0.
- Staged as `others/71407/task226.onnx`; cumulative projected gain is now
  `+0.375890588234`. Evidence: `agent_task226_regolf_187/REPORT.md` and
  `audit/final_audit.json`.

### task192 fixed-threshold POLICY90 shave 188--191

- Replaced exact `ArgMax+OneHot` selection with a standard `HardSigmoid`
  threshold at count33. Cost improves1609->1138, including1149->1138 over the
  previously staged exact candidate, for an additional projected
  `+0.009619663162`.
- Independent actual-ONNX audit passes known265/265 in four ORT configurations
  and fresh4998/5000 plus4997/5000 (`9995/10000=99.95%`). Runtime errors,
  nonfinite values, near-positive values, shape mismatches, lookup ops, and
  Conv-family UB are all0.
- This candidate is deliberately `POLICY90`, not exact: reachable cases
  `B=26,D=1` and `B=48,D=37` prove false-negative and false-positive behavior.
  It is active only under the user's explicit >=90% normal-candidate rule.
- The exact cost1149 fallback is preserved outside the active ONNX glob as
  `others/71407/FALLBACK_EXACT_DO_NOT_AUTO_MERGE/task192_exact1149.onnx.fallback`.
  Root submission and score ledger remain immutable. Cumulative staged gain is
  `+0.385510251397`. Evidence:
  `agent_task192_threshold_k33_191/REPORT.md` and `result.json`.

### task205 cost937 private-zero proof audit 192

- The historical cost937 candidate SHA `bbfa8f5b...` is hard-rejected. A
  generator-reachable seed93023205/case11 input reproduces12 differing one-hot
  cells in both disabled/default ORT with truthful shapes and nonfinite0, so
  this is a semantic failure rather than a runtime artifact. The exact SHA is
  also the recorded task205 LB-black payload from `others/7805`.
- Lowering the internal threshold to1.98 repairs that witness but fails one of
  266 known cases and13/1000 fresh cases. No exact repair below the staged1041
  frontier was found. The separate staged1041 candidate remains safe because
  it is an all-valid-input nonnegative Mul->Selu identity against authority.
- New adoptees0; gain0. Evidence:
  `agent_task205_private_proof_192/REPORT.md`, `counterexample.json`, and
  `model_audit.json`.

### task396 quarantined-candidate support audit 191

- Both quarantined policy candidates have generator-reachable semantic
  counterexamples in all four ORT configurations. Cost1017 underflows
  `uint8(0)-1` to255 and gives48 mismatched cells versus authority's16 on
  seed92000396/case23. Cost961 keeps only three TopK rows and drops the
  decisive fourth row, giving4 mismatched cells on seed94000396/case93.
- Known266/266 still passes for both candidates, demonstrating why the known
  set alone is insufficient. Runtime errors, nonfinite values, and shape
  mismatches are0; these are true semantic failures. A cost1018 local repair
  inherits authority's generator failure, while the sound control costs1245.
- Both candidates are marked `REJECTED_DO_NOT_MERGE`; gain0. Evidence:
  `agent_task396_support_191/REPORT.md` and `audit/result.json`.

### task125 current-authority exact regolf 193

- Profiled nine exact local-transform families against current cost1045.
  Sharing `hW/vW` only ties1045 and is nontruthful; the optimistic truthful
  lower bound of the current topology is1096. Independent generator reference
  streams pass2000/2000 each, but no strict-lower exact graph was found.
- Adoptees0; gain0. Evidence: `agent_task125_regolf_193/REPORT.md`.

### task245 exact positive-Selu shave 196

- Retained the static singleton batch label in four coordinate `Einsum`
  reductions and replaced rank-broadcast `Div(log,2)` with
  `Selu(log,alpha=1,gamma=0.5)`. Generator facts prove all four Log values are
  positive: the 5x5 Conway sprite keeps every row/column and the green corners
  are six cells apart. The now-unused float16 scalar is removed.
- Cost385->384, candidate SHA `1b777a51c...`, projected gain
  `+0.002600781700`. Div/Selu is bitwise-equal on all31,744 nonnegative finite
  float16 values in disabled/default ORT. Known267 and fresh2x5000 are
  raw-bitwise-equivalent and correct in all four ORT/thread configurations,
  with error/nonfinite0.
- Both authority and candidate retain the same pre-existing AffineGrid
  data-propagation cloak; ordinary full/strict checks pass and the rewrite adds
  no mismatch. Staged as `others/71407/task245.onnx`; cumulative projected
  gain is `+0.388111033097`. Evidence:
  `root_task245_regolf_196/REPORT.md` and `audit.json`.
- An independent audit repeated known267 and new fresh2x3000 in all four ORT
  configurations, compared the final output plus all four code and Log tensors
  bitwise, and exhaustively checked the nonnegative float16 operator domain.
  It found zero differences/errors/nonfinite values and independently approved
  cost384. Evidence: `agent_review_task245_197/REPORT.md`.

### task008 current-authority exact regolf 194

- Current authority cost431 passes266 known cases but only1958/2000 fresh
  cases. A generator-reachable isolated-red witness proves that the missed move
  is semantic, and runtime output shape `[1,10,30,30]` contradicts the declared
  `[1,1,1,1]`; the lineage is therefore private-zero/shape-cloaked.
- Three exact scalar-sharing derivations each save one parameter byte but add
  one memory byte, tying431. A truthful index construction has a cost floor of
  at least2020. No strict-lower guaranteed pass-through candidate exists;
  adoptees0 and gain0. Evidence: `agent_task008_regolf_194/REPORT.md`.

### task319 Log-scale Selu screen 199

- Tested whether the single-use `inv_ln2` scalar could be absorbed into Selu
  after `Log(max_abs)`. On10,000 generator cases, `max_abs_f16` reaches0.5 and
  `log_abs_f32` reaches `-0.693359375`; 113 sampled cases enter the negative
  Selu branch. Therefore Selu is not a linear replacement on valid support and
  cannot guarantee authority pass-through. No candidate is staged; gain0.

### Active-stage exact fusion rescan 202

- Re-ran21 exact ONNX optimizer pass sets over all14 active71407 descendants,
  covering294 task/pass profiles. Although148 serialized graphs changed, none
  had a lower competition-profiled cost than its staged parent.
- This covers follow-on opportunities exposed only after the later exact
  task158/192/226/245/328 and related rewrites. New adoptees0; gain0.
  Evidence: `root_stage_fusion_202/REPORT.md` and `scan.json`.

### task270 current-authority truthful regolf 195

- The cost587 authority is known/fresh-correct but has two explicit runtime
  shape mismatches (four including dependent CastLike tensors) and a19-input
  Einsum. Current-derived truthful rebuilds bottom out at cost592.
- The cost592 shared-scale control passes known266 in four configs, fresh
  2x1000 in four configs, complete79-state renderer and256-mask checks, with
  error/nonfinite/mismatch0 and raw authority equality. A cost588 probe is only
  250/266, and four rank5 searches totaling50,000 steps retain15 sign errors.
- No strict-lower truthful candidate exists in the audited families; adoptees0
  and gain0. Evidence: `agent_task270_regolf_195/REPORT.md` and
  `audit/result.json`.

### task349 exact affine/support shave 203--204

- Superseded the staged cost3556 parent with cost3548 SHA `f7531b66a539...`.
  All11 table rows satisfy `top=1-2r` and `hstart=1-3r`, removing the11-element
  hstart table without adding a parameter. Generator AST proof fixes side to
  `{10,15,20,25,30}` and halo_end is clipped at side, so on complete valid
  support `Equal(x,30)==Greater(x,29)`; reusing max29 removes max30.
- Profile is3233 memory +315 parameters =3548, total immutable-authority gain
  `ln(3564/3548)=+0.004499445161`; incremental stage gain is
  `ln(3556/3548)=+0.002252253204`.
- Root known267x4, fresh2x2500x4 and separate5000x2 ORT are raw-bitwise equal.
  Independent review repeats known267x4 and fresh2x2500x4 (20,000/20,000 raw),
  proves generator support mechanically, and finds error/nonfinite/mismatch0,
  full/strict/data_prop/truthful/UB0 pass. Staged in71407; cumulative projected
  gain is now `+0.390363286301`. Evidence: `root_task349_affine_203/REPORT.md`
  and `agent_review_task349_affine_204/REPORT.md`.

### task066 complete-support uint8 carrier shave 200/206

- Replaced the single-use `Div(selLog, ln2)` with
  `Selu(alpha=1,gamma=1.4432698488)` and removed `ln2`. The authority and
  candidate profiles are346 memory plus216/215 parameters, cost562->561,
  projected gain `+0.001780944371`.
- The two float16 paths differ on51 isolated values, so admission is not based
  on a float identity. Independent constant contraction and exhaustive
  enumeration prove every1,861,056 S/U/flip/xpose generator geometry selects a
  positive mask in `[1,2^20-1]`; extra cyan can only add bits. Exhausting all
  1,048,577 uint32 values from0 through2^20 in four ORT configurations finds
  zero differences after the only consumer, `Cast(... -> uint8)`.
- Independent known266x4 and two fresh seeds x2000x4 are final-raw and
  `ti`-raw identical to authority, with runtime errors0. Full/strict/data_prop,
  all79 runtime shapes truthful, standard-domain and UB0 pass. Staged as
  `others/71407/task066.onnx`; cumulative projected gain is now
  `+0.392144230672`. Evidence: `agent_task066_selu_200/REPORT.md` and
  `agent_review_task066_206/REPORT.md`.

### task319 inherited-cloak exact pass-through shave 201/207

- Five local rewrites reduce the immutable task319 profile from863 memory plus
  140 parameters (cost1003) to840 plus138 (cost978), projected gain
  `+0.025241117927`: transpose-invariant square correlation, exact Boolean
  alternate index, unsaturated factor-2 predicate absorption, singleton-rank
  preservation, and terminal Scatter weight construction.
- Complete generator bounds give binary correlation `S<=25` and selected color
  count `C<=100`, so `8S<=200` and `2C<=200`; neither QLinearConv saturation
  nor uint8 shift overflow occurs and `8S>=2C` iff `4S>=C`. Exhaustive Boolean,
  2,626 `(S,C)` and100 terminal-index pairs independently pass.
- Candidate retains exactly the authority's26 declared/runtime shape
  mismatches and introduces0. Independent known267x4 and two fresh seeds
  totaling3000x4 are raw-bitwise equal, with runtime errors/nonfinite0 and
  UB0. Staged as `others/71407/task319.onnx`; cumulative projected gain is now
  `+0.417385348599`. Evidence: `agent_task319_exact_201/REPORT.md` and
  `agent_review_task319_207/REPORT.md`.

### task349 complete-support residual shave 205/209

- Superseded the staged task349 cost3548 parent with SHA `8ab46bc1217c...`,
  profile3239 memory plus293 parameters = cost3532. Incremental stage gain is
  `ln(3548/3532)=+0.004519781706`; immutable-authority gain is
  `ln(3564/3532)=+0.009019226867`.
- Five residual rewrites are closed over complete support: generate valid-cols
  by negative Gather from the existing affine powers; derive all11 legal
  shifts with nonsaturating uint8 BitShift; narrow exact sides10/15/20/25/30
  and coordinates0..29 to int8; produce the beam bound directly as rank4 Min;
  and split the duplicate special h-patch with an all-input exact condition.
- Independent review rederived every table row and all supported sides, found
  99 common nodes and22 initializers byte-identical, and passed known267x4 plus
  fresh2x2500x4 raw-bitwise equality. Full/strict/data_prop, all123 runtime
  shapes truthful, error/nonfinite0 and UB0 pass. Cumulative staged gain is now
  `+0.421905130305`. Evidence: `agent_task349_residual_205/REPORT.md` and
  `agent_review_task349_residual_209/REPORT.md`.

### task319 all-input residual shave 210/213

- Superseded task319 cost978 with SHA `a4e0531b0a3d...`, profile848 memory plus
  127 parameters = cost975. Incremental gain is
  `ln(978/975)=+0.003072199037`; immutable-authority gain is
  `ln(1003/975)=+0.028313316964`.
- Three additional transformations are all-input local identities: compare the
  ten-channel int64 ArgMax directly to an int64 ramp; reduce the fixed
  `[1,1,2]` Boolean tensor directly to a scalar; and transpose the existing
  one-hot background mask into terminal weights with Where. Independent proof
  exhausts all10 ArgMax indices, four Boolean assignments and100 background /
  target index pairs.
- Independent known267x4 and fresh2x1500x4 are raw-bitwise equal with
  error/nonfinite0. The same26 inherited mismatch signatures remain under both
  ORT modes, with new/removed mismatches0. Cumulative staged gain is now
  `+0.424977329342`. Evidence: `agent_task319_residual_210/REPORT.md` and
  `agent_review_task319_residual_213/REPORT.md`.

### task066 exact selector-factor residual shave 208/212

- Superseded task066 cost561 with SHA `622b3b282718...`, profile346 memory plus
  205 parameters = cost551. Incremental gain is
  `ln(561/551)=+0.017986096370`; immutable-authority gain is
  `ln(562/551)=+0.019767040741`.
- Removed the ten-element `greenhalf10=e3` initializer by reconstructing it
  inside the existing Gv/Gh Einsums from live Uchan/Vchan/Trow/Tcol/z1
  factors. Independent expansion finds exactly one nonzero product, at color3,
  with coefficient+1; directed804x4 checks also find no signed-zero difference.
- Because Gv/Gh and all13 downstream traces remain raw identical, the earlier
  exhaustive1,861,056 geometry and `[0,2^20]` uint8 carrier proofs carry
  through unchanged. Independent known266x4 and fresh2x2000x4 plus directed
  cases have error0, final nonfinite0, truthful79/79 shapes, standard ops and
  UB0. Cumulative staged gain is now `+0.442963425711`. Evidence:
  `agent_task066_residual_208/REPORT.md` and
  `agent_review_task066_residual_212/REPORT.md`.

### task192 center-basis pass-through shave 211/214

- Superseded the active POLICY90 cost1138 candidate with SHA
  `1200fe8473c0...`, profile200 memory plus934 parameters = cost1134.
  HardSigmoid, selected vector, adjacency, threshold and output equation remain
  unchanged; only the exact center/hist basis is refactored. Incremental gain
  is `ln(1138/1134)=+0.003521130399`, and immutable-authority gain is
  `ln(1609/1134)=+0.349861662705`.
- Independent proof checks all1,024 Boolean selected vectors including
  multi-hot cases across11 legal cell/color coefficients, totaling33,792
  comparisons with0 differences. Known265x4 and fresh2x1500x4 are raw-bitwise
  equal to the parent; one fresh miss is the disclosed parent POLICY90 behavior,
  not a new candidate regression. Errors/nonfinite0, truthful shapes,
  Hardmax0 and UB0 pass.
- The all-support exact ArgMax fallback independently improves1149->1143 SHA
  `5c5eaefa81ac...`, passes fresh3000/3000 and is stored separately as
  `FALLBACK_EXACT_DO_NOT_AUTO_MERGE/task192_exact1143.onnx.fallback`.
  Cumulative active staged gain is now `+0.446484556110`. Evidence:
  `agent_task192_local_211/REPORT.md` and
  `agent_review_task192_policy_214/REPORT.md`.

### Latest active-stage fusion rescan 216

- Re-ran the21 exact optimizer pass sets plus four DCE/initializer/no-op
  cleanup sets over all16 current descendants, covering400 profiles. There
  were201 protobuf changes and169 full/strict/data_prop-valid changes, but zero
  profiles changed competition cost and no strict-lower candidate survived.
- No active model has an unused initializer or byte-identical initializer alias
  group. New adoptees0; gain0. Evidence:
  `agent_stage_fusion_latest_216/REPORT.md` and `scan.json`.

### task158 complete-support residual shave 215/219

- Superseded the staged task158 cost7525 descendant with SHA
  `e7101699bfc022fa794e15d7f374a8febe3e2680b8388c67b9a81cdc9962ced0`,
  profile6638 memory plus860 parameters = cost7498. Incremental staged gain is
  `ln(7525/7498)=+0.003594492321`; immutable-authority gain is
  `ln(7578/7498)=+0.010612994283`.
- The rewrite is closed over all48 local generator configurations. Independent
  proof covers score support `{0,2,4,8,10,16,20,24,26,48,52,72,106,144,212}`
  and at least97 exact-zero TopK windows, and verifies the `0b1010` anchor
  bitmask and phase threshold exactly.
- Known266x4 and two independent fresh streams totaling3000x4 are raw-bitwise
  equal to the parent with runtime errors/nonfinite/shape mismatch0. It is
  staged as `others/71407/task158.onnx`; cumulative staged gain is now
  `+0.450079048431`. Evidence: `agent_task158_residual_215/REPORT.md` and
  `agent_review_task158_residual_219/REPORT.md`.

### Low-cost exact audit waves 218/220--222

- task209 had an exact2085->2083 local simplification, but inherited16 runtime
  shape mismatches and two CenterCropPad cloaks; its truthful rebuild costs2650.
  task366's7985->7984 diagnostic repaired OOB errors but still had98 runtime
  shape mismatches; its truthful repair costs9465. Both were rejected.
- task222 had no safe reduction. Removing its planted-rank mechanism failed
  known checks, and the generator admits identical inputs paired with different
  planted outputs, proving deterministic all-support reconstruction impossible.
- tasks218/394/397, 206/212/247/273, 159/199/259/301 and task341 produced no
  safe strict-lower candidate. In particular task301's240->236 proposal failed
  its first known case, while task341's apparent no-op removal exposed a
  truthful cost127397 instead of authority260.
- New adoptees0. Evidence: `agent_task209_residual_218/REPORT.md`,
  `agent_task366_residual_217/REPORT.md`, `agent_task222_exact_221/REPORT.md`,
  `agent_tasks218_394_397_222/REPORT.md`,
  `agent_tasks206_212_247_273/REPORT.md`,
  `agent_tasks159_199_259_301/REPORT.md`, and `root_new_exact_220/REPORT.md`.

### Low-cost exact audit waves 223--225

- tasks225/228/388/400 passed their stored gold and two fresh1000-case streams
  in four ORT configurations (8000 executions per task), with raw equality and
  error/nonfinite0. Ninety-six optimizer profiles yielded strict-lower0.
  task388's two exact manual rewrites cost9303 and1599 versus authority305.
- tasks153/200/316 passed complete known plus two fresh3000-case streams in four
  configurations, with truthful shapes and error/nonfinite/UB0, but44 exact
  optimizer profiles yielded strict-lower0. task161 passed known266/266 but only
  5966/6000 legal fresh cases, so the current heuristic rule was not extended.
- New adoptees0. Evidence: `agent_tasks225_228_388_400/REPORT.md` and
  `agent_tasks153_161_200_316/REPORT.md`.

### task175 accepted-support latent shave 226/independent review

- Added SHA
  `40a9405880836a60f100e0072b476e4383c12c7ee053eb12ada1f049ee2e8d7c`,
  profile0 memory plus145 parameters = cost145 versus authority166. Projected
  gain is `ln(166/145)=+0.135254045936`.
- The candidate removes one redundant latent slice from the existing single
  Einsum. It is truthful, full/strict/data_prop-valid, standard-domain, and has
  no lookup/cloak op or Conv-family UB. Four fixed validate examples score
  262/266 because they erase both cells of30 off-diagonal symmetric pairs.
- Random `generate()` explicitly rejects every such double erasure. Primary
  fresh two seeds x3000 and independent disjoint fresh two seeds x2000 pass in
  all four ORT configurations, totaling40000 candidate executions with
  runtime error/nonfinite/shape/determinism mismatch0. It is not private-zero
  and is staged transparently under the user's POLICY90 rule. Cumulative staged
  gain is now `+0.585333094367`.
- Evidence: `agent_tasks175_224_240_376/REPORT.md` and
  `agent_review_task175_policy90/REPORT.md`.

### Low-cost exact audit waves 227--229

- tasks189/263/304/383, 190/195/243/358 and109/184/368/374 produced no safe
  exact or POLICY90 winner. The nominal task184 421->420 candidate scored only
  7/169 in its runnable mode, failed all default sessions, and carried shape
  and margin violations. Other lower histories were wrong, unscorable, or
  runtime-shape cloaks.
- New adoptees0. Evidence: `agent_tasks189_263_304_383/REPORT.md`,
  `agent_tasks190_195_243_358/REPORT.md`, and
  `agent_tasks109_184_368_374/REPORT.md`.

### Broad residual audit waves 230--234

- tasks025/117/131/330 yielded72 residual variants:37 structural rejects,
  35 valid but non-lower, and0 strict-lower. tasks080/165/268/308,
  182/204/208/284, and008/062/239/363 likewise produced no safe exact or
  POLICY90 finalist. Apparent lower histories were runtime-shape cloaks,
  default-session failures, or far below90%.
- task251's cost582 history candidate could be made checker/strict/default-safe
  only by exposing64 truthful tensors; cost became295949 versus authority755.
  The first truthful `[1,30,30,30]` float tensor alone costs108000 bytes, so a
  safe repair below authority is impossible.
- task379's current1947 LB-white model is already at its exact contraction
  floor. The only remaining precontraction changes20 shared elements into32
  dense elements. tasks070/254/267 are one-node output-only exact-rank floors
  at costs66/42/30.
- New adoptees0. Evidence: `agent_tasks025_117_131_330/REPORT.md`,
  `agent_tasks080_165_268_308/REPORT.md`,
  `agent_tasks182_204_208_284/REPORT.md`,
  `agent_tasks008_062_239_363/REPORT.md`,
  `agent_task251_runtime_repair/REPORT.md`,
  `agent_task379_residual_exact/REPORT.md`, and
  `agent_tasks070_254_267_residual/REPORT.md`.

### task344 normal-POLICY90 compact-G promotion 235

- Added SHA
  `c5272a42bee419008a15d14bea734a6fb15956a863ad8e702deac0f02fcea5f6`,
  profile0 memory plus132 parameters = cost132 versus authority137. Projected
  gain is `ln(137/132)=+0.037179003242`.
- It is a standard-domain single Einsum with truthful canonical output and no
  intermediate tensor, lookup, shape cloak, Conv UB, runtime error or
  nonfinite value. task344 is not private-zero lineage.
- Known266/266 passes in four ORT configurations. Prior fresh seeds scored
  9984/10000 and9986/10000; independent disjoint seeds each scored9981/10000
  identically across four configurations. Candidate-authority sign differences
  were0 throughout the new20000 cases. A separate generator-reachable margin
  witness remains disclosed, so this is POLICY90 rather than all-support exact.
  Cumulative staged gain is now `+0.622512097609`.
- Evidence: `agent_task344_rebase_171/REPORT.md`,
  `agent_task344_margin_184/REPORT.md`, and
  `agent_task344_policy90_review/REPORT.md`.

### task310 exact parity-factor residual shave 236

- Added SHA
  `6ccf625a0dca41d5c9cb39ddb41c3756313f2a01ac95f38d70c880c677ccf858`,
  profile194 memory plus297 parameters = cost491 versus authority501. Projected
  gain is `ln(501/491)=+0.020161973290`.
- Replaced the16-element even-parity tensor used twice by the exact shared
  factorization `0.5*sum_k H[k,d]H[k,r]H[k,j]H[k,c]`, where
  `H=[[1,1],[1,-1]]`. Reconstructed float32 tensor entries are bit-identical.
- Known266x4 and two fresh seeds x5000x4 are raw-bitwise identical to authority,
  with max delta0 and runtime error/nonfinite/near-positive/shape mismatch/UB0.
  The authority's one known fresh truth miss is identically inherited; the
  candidate introduces0 new failures. Cumulative staged gain is now
  `+0.642674070899`. Evidence: `agent_task310_residual/REPORT.md`.

### task023 POLICY90 upgrade rejection 237

- The cost1541 candidate is known266/266 but scored44022/50000=88.044% on a
  larger holdout. Its best coordinate variant reached44512/50000=89.024%,
  below the user's90% gate. A two-layer morphology design had theoretical
  cost1611--1614 but was not implemented or validated, so no candidate was
  admitted. Evidence: `agent_task023_policy90_upgrade/REPORT.md`.
