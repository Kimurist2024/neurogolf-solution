# 8018.91 lane A — cost 100–249

## Strict result

One new candidate clears every mandatory gate:

| task | authority | candidate | score gain |
|---:|---:|---:|---:|
| 175 | 140 | **131** | **+0.066445099408** |

Used alone on the 8018.91 archive, this projects to **8018.976445099408**.
This is an arithmetic projection, not an LB measurement.

Candidate:

`candidates/task175_gauge_s_factor_reuse.onnx`

Single-task probe ZIP (400 members; only task175 changed):

`submission_PROBE_task175_cost131.zip`

SHA-256:

`22fe38f6428dbc2f98b7135825325044f1898a7da23e2bea9b7584d97bfe4265`

## Exact rewrite

The pinned 8018.91 task175 member is byte-identical to the previously audited
LB-white cost-140 model.  Two algebraic reductions are combined:

1. An exact tensor-network gauge absorbs the 2x2 `W` and two-element `V`
   initializers into `TA`/`TB`, reducing 140 to 134.
2. `S = T @ Msel` exactly, where `T` is a 3x3 row-swap permutation.  Every
   `S` occurrence in the same output Einsum is replaced by the already-live
   `Msel` plus one shared `T`.  This removes 12 parameters, adds 9, and reduces
   134 to 131 without creating a node output or counted activation.

The serialized float32 factor reconstruction has maximum absolute error zero.

## Mandatory gates

- authority archive SHA-256:
  `e43865760ec8807fbb217fba718226ca6b86d9128b911479214e3252b9f9e091`;
- task175 authority-member SHA-256:
  `b6404486ccc1a74c36bab6031f11c54c7326f787a743f02dff77e63c782af343`;
- full ONNX checker and strict data-propagating shape inference: pass;
- canonical static input/output `[1,10,30,30]`: pass;
- local and official train/test/arc-gen gold: **266/266 exact**;
- official cost profile: memory 0, parameters 131, cost 131;
- raw margin: minimum positive 0.25, maximum non-positive 0.0, no value in
  `(0,0.25)`;
- seed 801891177: 2000/2000 exact under ORT_DISABLE_ALL threads 1 and 4;
- seed 801891178: 2000/2000 exact under ORT_DISABLE_ALL threads 1 and 4;
- authority-vs-candidate sign differences on the 4,000 fresh cases: 0;
- runtime, nonfinite, output-shape, and small-positive errors: 0.

Machine evidence is in `task175_cost131_evidence.json`.

## Other search coverage

- Rebased both local and model-wide exact 2-D Einsum factor-reuse scans against
  all 400 members of `submission_base_8018.91.zip`.  In cost 100–249, task175
  was the sole exact parameter-saving lead.
- Exhausted 4,096 no-new-parameter selector-alias gauge variants.  None passed
  even the two-case known-data pre-screen (`task175_full_alias_gauge_search.json`).
- A nominal cost-122 direct `Msel` alias is **rejected** because it fails gold.
  It must not be merged.
- The cost-134 model is strict-safe but superseded by cost 131.
- Output-reachability scanning found an unused `Min` in task183, but deleting
  it leaves the inherited output declaration/inference at `[1,1,1,1]` rather
  than canonical `[1,10,30,30]`.  It was rejected by the strict static-shape
  gate and no task183 candidate was emitted.

This lane wrote only under `scripts/golf/restart8018_91_lane_low/`.  It did not
write the root submission, CSV, or score pointer; those shared files may be
updated independently by the parent campaign.
