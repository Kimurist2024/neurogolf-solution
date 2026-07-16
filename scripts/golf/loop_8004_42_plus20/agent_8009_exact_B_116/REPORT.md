# 8009.46 exact golf B — final report

## Outcome

**NO_SAFE_EXACT_CANDIDATE**. The immutable 8009.46 payload was not modified.
No candidate was promoted, and no fresh run was admitted because no
strict-lower candidate passed the earlier dual-ORT and truthful-shape gates.

- Authority: `submission_base_8009.46.zip`
- Authority SHA-256:
  `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
- Tasks inventoried: 12/12
- Unique transformed graphs: 15
- Strict-lower graphs: 2, both task264
- Winners: 0

## Fixed current inventory

| task | immutable member SHA-256 | memory | params | cost | result |
|---:|---|---:|---:|---:|---|
| 178 | `8e63512bb0f2f742fe1e9b1efc8751a82db1808c98c2636f6a37f98eada57823` | 206 | 56 | 262 | no lower candidate |
| 228 | `13946c8d5a52886212f495b13fa6c128a091e77b84e10e27b442ed87f9694a45` | 241 | 50 | 291 | no lower candidate |
| 234 | `c66bb8c3e92a3a2ef5844872cb36d54076ba697f8ee6e777ceef6c2dd0997c8a` | 229 | 132 | 361 | no lower candidate |
| 264 | `44163ea87aca5993f076612fd1f8d1c21a232521964666c3644e7dc3817b9160` | 321 | 23 | 344 | two lower candidates rejected |
| 325 | `5e8e919a23dd9f6e8ba0f82ff08ebef634b1b1975e466ce80bf03ed5288ef3b1` | 165 | 57 | 222 | no lower candidate |
| 344 | `05bedf3ca834aadfc973c00fc91cafdb4d0ae1aaab374115d924e2e33fb1bf6c` | 0 | 137 | 137 | no lower candidate |
| 357 | `3c2e8d92b0c893c6836c64119074a2e38e7b975cf7bf8ef09654953b054e6d6c` | 196 | 61 | 257 | no lower candidate |
| 387 | `21a0c07d0101a7c538210dc44553f807d1bddcf755746ceaabad20c69561c160` | 234 | 95 | 329 | no lower candidate |
| 388 | `f27fa5f4f0bcade23d02fed2a74e3c2b826b11140bd03d29f47e0c59c382a8e1` | 63 | 22 | 85 | no lower candidate |
| 392 | `96ed4658476f09cb77f09848081bb5830b96b6ea65c8a5a3278f28f9036e32f9` | 272 | 57 | 329 | no lower candidate |
| 397 | `2361956a9b6d1391aff9d8bc4af26d5112877e0e2946b365d04e461899f4d7e1` | 249 | 89 | 338 | no lower candidate |
| 398 | `339e0b25b3f45862f51b98c239755e597aca040b951acec5065b380a753d2513` | 144 | 202 | 346 | no lower candidate |

Each cost was independently reprofiled from the immutable ZIP member and
matched the inventory pass.

## Search coverage

The scan exercised:

- output-unreachable nodes and unused initializers;
- value-identical initializer aliases;
- Identity and exact no-op Cast, CastLike, Reshape, Transpose, one-input
  Concat, and neutral Add/Sub/Mul/Div removal;
- deterministic common-subexpression elimination;
- unused optional outputs;
- uniform elementwise constants reduced to scalar broadcasting;
- 29 conservative `onnxoptimizer` passes individually and as a fixed-point
  set;
- `onnxsim` constant folding and simplification.

Most passes reproduced the current graph byte-for-byte. Fifteen unique graphs
remained; five were structurally invalid, eight retained the same official
cost, and two were strict-lower.

## task264 rejection

Both strict-lower files perform the same exact rewrite: the identical INT64
initializers `s1=[1]` and `axes1=[1]` are aliased. The independent proof
confirmed value, dtype, and shape equality; `axes1` is removed and all uses are
routed to `s1`.

| candidate | SHA-256 | cost | disable-all known/raw | default ORT | truthful trace |
|---|---|---:|---:|---|---|
| manual alias | `52802ff9cf366a2528eb282e26722e9b53a8aab22d86dbd29095a1add21aaf8e` | 343 | 265/265 in threads 1 and 4 | session construction fails | 44/87 mismatches |
| optimizer duplicate initializer | `42b1438c91f40309e69ea6e44c9c1a8aaeb7f3ffba29d7ea30e21475d2b7d5a3` | 343 | 265/265 in threads 1 and 4 | session construction fails | 44/87 mismatches |

The default-session failure is at `CenterCropPad`: its one-element shape input
does not match two axes. Runtime tracing also shows declared output
`[1,10,1,1]` versus actual `[1,10,30,30]`, plus 43 intermediate mismatches.
The standalone Conv-bias checker reported zero findings, but that is
insufficient to override dual-ORT and truthful-shape failures.

Fresh verification was deliberately skipped after these mandatory gates
failed. This avoids spending fresh evidence on an ineligible candidate and,
more importantly, prevents replacing the LB-white payload with a shape-cloaked
variant.

## Artifacts

- `inventory.json` — all task SHAs, official costs, graphs, and opportunities
- `candidate_scan.json` — all unique transformed graphs and structural/cost decisions
- `audit.json` — independent cost, known four-config, raw equivalence, shape trace, and rejection evidence
- `manifest.json` / `winner_manifest.json` — empty winner sets and non-promotion declaration
- `baseline/` — immutable extracted target members
- `candidates/` — rejected task264 evidence only

The `neurogolf-onnx-golf` workflow affected the decision directly: a nominal
one-cost exact alias was rejected because admission requires default and
disable-all ORT plus truthful runtime shapes, not merely disable-all known
accuracy.

