# High48 target expansion — eight-task audit

## Outcome

The eight requested high-cost files were independently audited against the
immutable `submission_base_8005.16.zip`. **No safe, strictly cheaper candidate
exists in the complete retained-history frontier, the focused SOUND controls,
or the exact dead/duplicate-factor families.** This lane contributes `+0.0`,
emits no candidate, and does not build or modify a submission ZIP.

- baseline SHA-256: `73073208ec8b5b296f1fc13300a7e4df871135f0a9edd4a682ac7b48077aba00`;
- completed: **8/8**;
- history/SOUND models screened: **49**, of which **16** have lower truthful
  actual cost;
- safe pre-fresh finalists: **0**;
- accepted: **0**;
- all eight latest members are byte-identical to the 8004.50 base;
- the protected ZIPs, score files, and CSVs were not modified.

The existing 8005.16 members are treated only as fixed LB lineage, not as
permission for a new unsafe candidate. Several of them rely on shape cloaks,
lookup-like operators, or giant contractions. Every replacement was therefore
required to pass the stricter no-cloak/no-lookup/no-giant policy independently.

## Per-task decision

| task | latest actual cost | known disabled/default | strongest lower lead | terminal decision |
|---:|---:|---:|---|---|
| 008 | 431 | 266/266, 266/266 | static-427 history reprices to 22021; static-430 reprices to 454 | no lower candidate |
| 275 | 428 | 266/266, 266/266 | cost 317 is 128/266 and a 41-input Einsum | reject known/giant |
| 134 | 423 | 266/266, 266/266 | cost 320/322 known100 lookup models fail fresh-5000 | reject private guarantee |
| 112 | 422 | 266/266, **0/266** | no retained lower graph; truthful control costs 19891 | reject cost/runtime |
| 168 | 416 | 265/265, 265/265 | cost 166/285 known100 models are lookup + giant contraction | reject structure |
| 109 | 273 | 266/266, 266/266 | static 177..193 family reprices to >=273 or is unscorable | no lower candidate |
| 160 | 404 | 265/265, 265/265 | costs 384/384/402 are all 0/265 | reject known |
| 170 | 357 | 266/266, 266/266 | no retained lower graph; truthful control costs 24855 | no lower candidate |

The task109 and task170 costs above are official-like runtime-profiled costs;
they are lower than their nominal 405/387 static bookkeeping values because
the incumbent declarations hide larger runtime tensors. The hidden shape is
not accepted as a property of a new candidate.

## True-rule and SOUND findings

The Sakana `p` solver was decoded for every task and reproduced every stored
train/test/arc-gen pair: 266/266 for tasks 008, 275, 134, 112, 109, and 170;
265/265 for tasks 168 and 160.

- **task008:** bounding-box translation to the 2x2 color-8 anchor is global
  geometry. The ordinary truthful control costs 23288, far above 431.
- **task275:** generator-exact Kronecker substitution has only the size-3 and
  size-4 routes. The current 428 parameter-sharing model has prior 5000/5000
  generator evidence, but still uses a 41-input giant Einsum and has no strict
  lower safe successor. The historical cost-317 polynomial is wrong.
- **task134:** this is the private-risk Type-D color-role, scale, and location
  inference task. The two truthful-shape cheap archive leads are not exact:
  r04 cost 320 scores 4840/5000 disabled and 4823/5000 default; r06 cost 322
  scores 4803/5000 and 4825/5000. Both also contain forbidden lookup/scatter
  machinery. The ordinary rule implementation costs 12380.
- **task112:** the data-dependent four-way pivot reflection is Type D. The
  latest compact model cannot create a correct default-optimization session;
  the ordinary truthful implementation costs 19891.
- **task168:** the decoded bounded diagonal-ray rule is exact. The fixed 416
  member has prior 5000/5000 fresh evidence but uses a 52-input giant Einsum.
  Its ordinary no-giant rule control costs 20403. Cheaper 166/285 artifacts
  merely reproduce stored cases with lookup + giant contraction.
- **task109:** removing the pivot cross and reflecting the colored quadrant is
  global geometry. Every apparent lower history member collapses after
  truthful runtime profiling; the ordinary implementation costs 28110.
- **task160:** this is the only local Type-A rule in the lane: recolor exact
  3x3 plus sprites. The truthful convolutional control costs 2978. All three
  sub-404 renderers destroy every known output.
- **task170:** decoding a large scaled binary mosaic and masking the small
  color matrix is private-risk Type D. No graph below the truthful actual 357
  incumbent frontier exists, while the ordinary implementation costs 24855.

## History and exact-rewrite coverage

The all-400 source inventory covers **1,196 ZIPs, 448,568 ZIP members,
233,751 loose observations, and 13,591 unique non-baseline graphs**. Its
retained lower frontier was unioned with the top-200, loose, ZIP-sweep, and
task-specific SOUND artifacts, deduplicated by SHA-256, then reprofiled against
the latest costs. Sixteen artifacts were genuinely lower; none passed cost,
known100, and safe structure together.

The exact-factor audit found no unused initializer, duplicate initializer,
duplicate expression, or wholly dead node in any of the eight latest members.
Task134 has an unused `poolv` tensor, but it is the required first output of a
live two-output `MaxPool`; the consumed `pooli` index output prevents removing
the node or omitting its required value output.

## Fresh-gate disposition

Fresh testing is fail-closed after the cost, structure, and complete-known
gates. Because **zero** proposal reached the pre-fresh stage, no new 2000x2 run
could make a candidate adoptable. Existing stronger evidence was retained for
the only close private-risk family: task134 r04/r06 each have independent
5000-case runs in both ORT modes and fail exactness as reported above. No
candidate was accepted from a private-zero lineage.

Authoritative evidence:

- `baseline_audit.json` — latest hashes, actual costs, full checker, strict
  inference, domains, Conv-bias audit, and runtime shape traces;
- `known_baseline_dual.json` — all stored examples under both ORT modes;
- `true_rule_audit.json` — decoded-rule reproduction on the complete corpus;
- `history_audit.json` — 49-model union, 16 lower actual-cost rows, known and
  structural decisions;
- `exact_factor_audit.json` — exact dead/duplicate opportunity proof;
- `result.json` — eight final decisions;
- `winner_manifest.json` — empty authoritative promotion list.
