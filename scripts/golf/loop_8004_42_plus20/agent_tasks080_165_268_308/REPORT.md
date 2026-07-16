# tasks080/165/268/308 high-memory residual / POLICY90 audit

## Outcome

**No candidate is admissible.** The winner set is empty and projected gain is
`+0.000000`. Large fresh was not run because no model cleared the mandatory
strict-lower and structural pre-gates.

The immutable authority was root `submission.zip` at LB **8009.46**, SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`.
The root ZIP, `all_scores.csv`, and `others/71407` were read only and were not
changed.

| task | member SHA (prefix) | reprofiled memory + params = cost | candidate | decision |
|---:|:---|---:|:---:|:---|
| 080 | `761ed9c5c4d9` | 2837 + 213 = **3050** | none | no strict-lower exact or POLICY90 graph |
| 165 | `d6d40c11204c` | 517 + 70 = **587** | none | all lower descendants rely on malformed runtime shapes |
| 268 | `4c8ec91a517e` | 373 + 47 = **420** | none | authority itself is lookup/cloak; no clean lower graph |
| 308 | `fc845e9edee0` | 376 + 57 = **433** | none | authority itself is shape-cloaked; repair is far above 433 |

`authority_profile.json` is a fresh competition-profile remeasurement of the
exact root members. All four pass full checker and strict shape inference with
`data_prop=True`, have finite outputs on the scoring witness, and have Conv-bias
UB count zero. Those syntactic passes do not override direct runtime-shape
witnesses.

## Generator semantics and POLICY90 disposition

- **task080** compiles the line-grid motif completion rule. The scorer-valid
  generated size families are `5,6,7,9,10`; size 8 produces 31x31 and is skipped
  by the scorer contract. Conditional on scorer-valid generation, dropping any
  whole size branch leaves only about 80% support, below POLICY90. The current
  graph is the only one of the four with truthful runtime shapes and no lookup.
- **task165** detects the kite, identifies the random-pixel color, and extends
  that color down qualifying columns. The generator is fixed 20x20, but every
  historical sub-587 construction reaches the low cost by underdeclaring the
  spatial `CenterCropPad` chain. A newly adopted candidate must be truthful;
  the authority exception is SHA-specific and cannot be inherited.
- **task268** detects the colored open box, fills it with yellow, and draws the
  two fountain rays under flip/transpose. The current 1,476,925-byte payload
  encodes inputs through two huge `TfIdfVectorizer` tables and 33 shape-cloaked
  `CenterCropPad` nodes. It is not a standard no-lookup candidate lineage.
- **task308** reconstructs a centered 3x3, 5x5, or 7x7 symmetric target from
  broken-border witnesses against the modal background. The current graph
  declares `[1,1,1,1]` while returning `[1,10,30,30]`; its TopK path also fails
  default-ORT session construction. The finite parameter set is not small
  enough for a complete-support exception: positions, colors, grid dimensions,
  and non-clobbering witness layouts all vary.

None of the four tasks is in the 51-task private-zero catalog. That permits a
normal >=90% POLICY90 candidate, but only after strict-lower, full/strict,
truthful-shape, standard/no-lookup, margin/nonfinite, and UB0 gates. No model
reached the fresh gate.

## History and safe optimizer scans

The SHA-deduplicated historical inventories and the current-only exact scans
were inspected rather than trusting filenames or static estimates:

- task080: 77 models / 929 source aliases were previously scanned; the nearest
  non-authority known model costs 3053. Current-only cleanup, CSE, no-op,
  optional-output, initializer-dedupe, constant-fold, and combined scans yield
  no strict decrease.
- task165: 93 unique non-authority graphs were inventoried; nine static-lower
  leads were actual-audited, with zero admissible result. Every retained lower
  family uses the same `CenterCropPad` cloak.
- task268: a 54-SHA broad inventory found no safe graph below the then-cost-446
  authority. The one complete-known cost-327 rebuild is lookup-derived and
  scores only 2219/5000 (44.38%) fresh. It cannot qualify for POLICY90.
- task308: the broad scan covered 434 unique models across its wave; the nearest
  alternative known-correct graph was cost 470 against a cost-446 authority.
  Later scans found no strict-lower lead against cost 434, hence none against
  the current lower cost 433.

The apparent task080 dead outputs `max_all_val` and `max4_val` are the required
first outputs of MaxPool nodes whose index outputs are consumed; they cannot be
omitted while retaining the second output.

## Manual algebraic and memory-shrink attempts

- task080 constant-folding `Cast(wcol)` saves 10 runtime bytes but adds 10
  initializer elements: `3050 -> 3050`. It is exact but not strict-lower.
- task165 CSE of duplicate `CastLike(__sp_hid,seventeen_u8)` gives nominal
  `587 -> 547`, SHA prefix `455c05527d91`, but causes all 265 competition
  executions to error after allocator behavior changes. Default ORT also
  rejects the malformed shape chain.
- task268 replacing the dtype-only bool `CastLike` witness with `Cast(BOOL)` is
  algebraically exact, but exposes the 30x30 runtime tensor: `420 -> 1318`.
  The truthful generator-rule control costs 18665.
- task308 dtype-anchor reuse is an exact cost tie. Reusing the earlier Shape or
  bypassing the fixed shape copy makes TopK unloadable/unscorable. Direct shape
  tracing previously measured 63,546 bytes of actual intermediates versus 376
  nominal bytes, so a truthful repair cannot be strict-lower.

## Evidence

- `authority_profile.json`: fresh SHA/cost/checker/strict/profile evidence.
- `report.json`: machine-readable per-task history, attempt, gain, and verdict.
- `audit_authority.py`: non-promoting reproducer; all generated files stay in
  this lane.

Final verdict: **do not merge or stage any model from this lane**.
