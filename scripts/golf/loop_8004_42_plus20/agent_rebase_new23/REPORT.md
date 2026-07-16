# 8005.16 updated-member audit: tasks 133/145/182/187/201/204/216/233

## Result

**Safe adoptees: 0; projected gain: +0.000000.**  All eight assigned members
were extracted directly from `submission_base_8005.16.zip` (SHA-256
`73073208ec8b5b296f1fc13300a7e4df871135f0a9edd4a682ac7b48077aba00`).
No submission ZIP or protected root score file was modified.

The current actual costs and hashes are:

| task | cost | member SHA-256 | complete stored result | runtime-shape mismatches |
|---:|---:|---|---|---:|
| 133 | 4393 | `6c5dc3a593b0900e16966b9d4c40af509a34c1dd1f0264c31cd30eaf9b4570e5` | 267/267 both modes | 30 |
| 145 | 5129 | `35cf952052882ff0198d01b64b75e7d36b2ba054b758089c6b54310559544d19` | disable-all 267/267; default session error | 65 |
| 182 | 951 | `605285fc5c047465614c311cdd9c9511db48413a72126cbffde755c074dfb581` | disable-all 267/267; default session error | 47 |
| 187 | 1798 | `bb40138844229a6ede66203b2a99e3a474e43ac385e08f7fb0c079bed0231126` | disable-all 266/266; default TopK error | 13 |
| 201 | 789 | `fb28f6065fbac760fd5a9e40d00af44eb5128c3d76676e852e62baddb574beda` | 266/266 both modes | 6 |
| 204 | 2222 | `312fa4435c543c24301b15718602d148faa5c6510348a71f6482528b3092547b` | disable-all 268/268; default session error | 53 |
| 216 | 1025 | `9a5f4f10d6e014b3f053ce1dabeb39cbeaf95964ae685aa71514fd695caf0756` | 266/266 both modes | 53 |
| 233 | 7308 | `4263f04fd9a4fd783182ea66cd4164ab6a51f4109eca71e7f591057f3018a89f` | disable-all 266/266; default 49/266, 2 errors | 28 |

Every incumbent passes the full ONNX checker and strict static shape inference,
but all eight contradict their declared intermediate shapes at runtime.  That
distinction is decisive: strict inference alone does not make a shape-cloaked
member an admissible source for another micro-shave.

## Terminal decisions

- **task133:** the generator requires component/template recovery and scaled
  stamping.  The exact compact family is generator-risky; the previous stream
  was 94/100 with three runtime failures.  The independent truthful model is
  dual-known/fresh exact but costs 5570, above 4393.
- **task145:** the rule is global minimum/maximum leaf-area selection over a
  recursive red guillotine partition, including ties.  The truthful
  dual-known/fresh3000 model costs 10175.  The 5129 incumbent declares
  `[1,10,1,1]`, emits `[1,10,30,30]`, and cannot create a default ORT session.
- **task182:** the rule is boxed-sprite template matching and recoloring.  The
  current graph has 47 shape contradictions, default-ORT failure, and a
  dynamic QLinearConv bias whose channel length cannot be statically proven.
  The truthful rule control costs 7099.  The old cost-990/993 exact-reuse
  attempts are now both more expensive than 951 and structurally invalid.
- **task187:** the mandated Identity removal was independently rechecked and
  remains rejected: its sanitized sessions fail the TopK contract.  The only
  executable cheaper historical lead is shape-cloaked and 4695/5000 (93.9%)
  fresh in each mode.  It clears the user's numerical 90% bar but fails the
  unconditional runtime/truthful-shape gates, so it is not adoptable.
- **task201:** the latest 789 member uses a 51,241-entry
  `TfIdfVectorizer` signature table and `ScatterElements`.  Further table
  pruning would remain forbidden lookup/cloak lineage.  The direct
  generator-derived graph is dual-known/fresh5000 exact but costs 7898; the
  three cheaper archive leads score 0%, 0%, and 86.2% fresh.
- **task204:** the parity-fill rule is simple, but the compact 2222 graph is
  not a safe template: output metadata is `[1,10,1,1]`, runtime is
  `[1,10,30,30]`, and default ORT fails.  Retained histories cost at least
  2544 after actual profiling and keep structural defects.  The truthful
  rewrite experiment costs 6200 and is known-wrong.
- **task216:** the rule selects the blue rectangle with the uniquely most red
  cells and crops it.  Identity removal is raw-equal on the stored corpus but
  fails strict/data-propagating inference and runtime-shape truth.  The compact
  family has a retained fresh Gather out-of-bounds witness; the rule-derived
  rebuild costs 9135.
- **task233:** the rule matches rotated 3x3 sprite codes to cutouts in the red
  box and renders the matched colors.  The exact initializer alias saves one
  cost unit but inherits default ORT's 49/266 result and two errors.  The clean
  specification-derived model costs 17007, so neither that model nor any
  compact descendant beats 7308 safely.

No candidate survived the prerequisite cheaper + dual-ORT + truthful-shape +
no-lookup/no-UB gate, so fresh multi-seed validation was correctly not started.
This lane contributes no model to the integration ZIP.

## Evidence

- `baseline_audit.json`: exact current member hashes, actual runtime cost,
  full/strict checks, op inventory, Conv-family audit, and complete dual-ORT
  stored results.
- `baseline_runtime_shapes.json`: all-node declared/runtime shape traces.
- `result.json`: one terminal disposition for each of the eight tasks and the
  retained evidence paths.
- `audit_current.py`, `trace_baseline_shapes.py`: deterministic reproduction.

