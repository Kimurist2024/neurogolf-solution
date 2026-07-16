# Tasks 189 / 263 / 304 / 383 strict regolf audit

## Decision

**NO_STRICT_LOWER_SUPPORT_SAFE_WINNER.** Nothing is eligible for merge or
promotion. Exact accepted cost gain is **0** and exact accepted score gain is
**+0.0**.

The authority is the corresponding member extracted read-only from root
`submission.zip` (archive SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`).
The before/after manifests for `submission.zip`, `all_scores.csv`, and
`others/71407` are identical. No ZIP or score ledger was edited.

| task | authority SHA-256 | memory + params = cost | truthful runtime shapes | decision |
|---:|---|---:|---|---|
| 189 | `9bedf9f8ff6e402c2111409893d90ff5c2a1c2069326c3b07c3a8794364bb6a9` | 149 + 34 = **183** | no: 13 mismatches, including output | reject shape/runtime cloak |
| 263 | `7c20efba425b4d413b725c0065dda7039617ca93ad8786dbcaa8793d5e416050` | 136 + 45 = **181** | no: 3 full-grid intermediates | reject shape cloak |
| 304 | `e395301e8b11cc06ce90b68e7ddfefd87ec003437431b484aa0c6b4f2f3b3f51` | 0 + 180 = **180** | yes | no correct lower simplification; giant 46-input Einsum/color Basis is not a clean replacement lineage |
| 383 | `d0dde772dc57a600f8757bae491e6540c88ec7d16b97f8aafecb766b202c656d` | 105 + 67 = **172** | no: 6 mismatches, including output | reject CenterCropPad/runtime cloak |

All four Sakana rules independently reproduce every complete known pair:
task189 `266/266`, task263 `267/267`, task304 `266/266`, and task383
`266/266`, with zero rule errors or shape mismatches. The compiled semantics
are:

- task189: orient the corner 2x2 palette and 6x6 green mask around the cyan
  divider, then recolor each mask cell by its palette quadrant;
- task263: split the 3-wide Conway sprites, select the unique sprite whose
  occupied-cell count differs, and preserve the input transpose;
- task304: find the modal input color and stamp the 3x3 input at every modal
  cell in the 9x9 output;
- task383: infer the two box colors and the barnacle rows/columns, remove the
  protrusions, and extend their full horizontal/vertical stripes outside the
  box.

## Runtime evidence

The four configurations are ORT `DISABLE_ALL` / default optimization crossed
with 1 / 4 intra-op threads.

- task189 authority: known `266/266` and each disjoint fresh seed
  `1000/1000` under both `DISABLE_ALL` settings; both default settings produce
  a runtime error on every case. Nonfinite, near-positive, and decoded output
  shape errors are zero in successful runs. Static output is declared
  `[1,1,1,1]` but runtime output is `[1,10,30,30]`.
- task263 authority: known `267/267` and fresh `1000/1000` for seeds
  `189263304` and `383304263` in every configuration. Errors, nonfinite values,
  near-positive values, and decoded output shape mismatches are zero. However
  `gn`, `q`, and `qf16` are statically `[1,1,1,1]` and actually full-grid
  `[1,10,30,30]`.
- task304 authority: complete known `266/266` in every configuration, with zero
  errors/nonfinite/shape mismatches and minimum positive raw value
  `17.502702713012695`. The long authority-only fresh extension was stopped
  after optimizer scanning found no strict-lower candidate; no candidate was
  waived through without fresh evidence.
- task383 authority: known `266/266` and each fresh seed `1000/1000` under both
  `DISABLE_ALL` settings. Default optimization cannot create the session
  (`TopK`: requested k exceeds the statically cloaked axis). Successful runs
  have zero runtime errors, nonfinite values, near-positive values, or decoded
  output shape mismatches; minimum positive is `0.372314453125`. Static output
  is `[1,1,1,1]`, runtime output is `[1,10,30,30]`.

Fresh inputs for all executed streams were generated from the official task
generators, and the independent Sakana rules matched `1000/1000` on each seed.

## Optimization and manual simplification search

Forty-four fixed-point `onnxoptimizer` profiles were tested (dead-end/CSE,
initializer aliases, idempotent/no-op cleanup, safe combined cleanup,
Conv/Pad/BN fusions, shape folds, Einsum-to-MatMul, `rewrite_where`, and
`adjust_add`). **Strict-lower profiles: 0.** Shape-folding on tasks189/383 is
itself rejected by strict inference because it exposes the inherited cloak.

For task304, exact symbolic precontraction of all shared `H` + `SF/SG`
selector pairs is truthful and raw-identical on the four-configuration witness,
but changes 180 to **198** parameters (SHA-256
`13819dcd0053e424a37f9d1b66de3a7d72743f5ed9ce19fe7b566be01c4c758d`),
so it is rejected before fresh gating.

Six lower task304 latent-axis deletions are truthful-shape models but fail the
first known case in all four configurations, with zero runtime errors and zero
output-shape errors:

| candidate family | cost | SHA-256 | differing threshold cells on known[0] | verdict |
|---|---:|---|---:|---|
| color factor 0 | 168 | `7f80e0de67183a8ce254e3177081f2650ab72a974be9757bd0d335b8dfad4ecb` | 21 | reject |
| color factor 1 | 168 | `1516a6c79a45fcfb6456af5f8d238837a42dfdf8d44b7043d2ae772192afd54e` | 42 | reject |
| color factor 2 | 168 | `418bc5d4466afa7de4e0ceabdd91bea3174c10d8ed9681197d6adf23e23a5dad` | 48 | reject |
| color factor 3 | 168 | `f4debd8d7b4a7e233c842b5b39781220e5ede00db3b0a8a95daa44466fe8e0ae` | 420 | reject |
| state factor 0 | 167 | `e9413890db201ffe46af48c9fac03c092eafa0e96cabfdf482ca4a975e7a3375` | 54 | reject |
| state factor 1 | 167 | `923a6e4c5fac0c9451d0eaae72b7a30c29f678af00627c73a4f9662a337fcaec` | 648 | reject |

Their hypothetical gains (`+0.068992871487` or `+0.074963038473`) are not
counted because each is already wrong on known data. Fresh testing is therefore
inapplicable, not treated as a partial pass. A task263 direct truthful-Cast
repair was also rejected: strict inference leaves a dynamic Slice extent and
the first known case is wrong, so it is neither scorable nor authority-equivalent.

## Evidence

- `evidence/audit.json`: machine-readable authority profile, hashes, complete
  known/fresh counts, all four runtime configurations, independent truth,
  runtime shape traces, optimizer profiles, manual probes, and immutable-file
  guards.
- `evidence/authority_costs.json`: independently profiled authority costs.
- `audit_lane.py` and `quick_complete.py`: reproducible, non-promoting audit.
- `candidates/`: rejected probes only; **none is a winner**.

No lookup-table candidate, private-zero assumption, CenterCropPad/runtime-shape
cloak, hidden type/shape omission, nonfinite candidate, or error-based candidate
was accepted.
