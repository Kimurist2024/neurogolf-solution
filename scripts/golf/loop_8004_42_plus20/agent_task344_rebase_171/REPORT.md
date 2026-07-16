# task344 rebase171 — cost137 authority re-audit

## Result

No auto-adopt winner. The best new model is a **probe-only** candidate:

- `candidates/task344_compact_g_cost132.onnx`
- SHA-256 `c5272a42bee419008a15d14bea734a6fb15956a863ad8e702deac0f02fcea5f6`
- competition actual: memory 0 + params 132 = **132**, strictly below 137
- potential score gain: **+0.037179**

The candidate passes every mechanical and sampled gate, but the finite-support
margin proof required for private-safe auto-admission is incomplete. It was
not merged into the root submission.

## Fixed authorities

- LB authority: `submission_base_8009.46.zip` / root `submission.zip`, SHA-256
  `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`.
- task344 member: SHA-256
  `05bedf3ca834aadfc973c00fc91cafdb4d0ae1aaab374115d924e2e33fb1bf6c`.
- generator/spec: `inputs/arc-gen-repo/tasks/task_d90796e8.py`.
- authority cost: memory 0 + params 137 = 137.
- root `all_scores.csv`: SHA-256
  `8c99379ce1c77e6894d53b593ccb5ae10f01c3c70e5666f832b1b2454c709b78`.

`d90796e8` is the generator hash, while `05bedf3c...` is the immutable current
ONNX payload hash.

## True rule and support

All updates read the original grid simultaneously:

```text
3 with an orthogonally adjacent 2 -> 8
each such adjacent 2             -> 0
everything else                  -> copied
```

The generator uses dimensions 3..10 and colors `{0,2,3,5}`. The authority's
spatial initializer `B[2,30]` is exactly zero after coordinate 9. However,
`common.random_pixels(..., 0.04)` makes every gray-cell subset have nonzero
support. Thus fresh sampling is not a complete enumeration: at 10x10, the gray
subset alone has up to `2^100` reachable states.

## Initializer/factor audit

The authority matrices have full numerical rank: H=3, V=4, B=2, S=3, M=4.
Consequently, direct lower-rank SVD replacements are not exact. The effective
neighbor-mode tensor has rank 3; its singular values are approximately
`12649.52, 4479.51, 2653.80`, so local rank 2 cannot reproduce it exactly.

Explored reductions:

| family | cost | decisive result |
|---|---:|---|
| local-rank-2 split-full | 132 | best known had 2 wrong cells; independent validation had 17 wrong cells |
| diagonal S | 131 | best known had 476 wrong cells |
| shared-V Gram absorption | 132 | known4 perfect; float32 identity residual `2.05e-7` |
| compact `G = H.T @ S @ H` absorption | **132** | known4 perfect; residual improved to `3.30e-8` |
| sparse B zero-tail | nominally lower | rejected: full checker/strict Einsum shape inference cannot infer sparse rank |

The remaining reduced-rank/diagonal variants are exact subfamilies of the
screened full variants or necessarily change a full-rank factor. P and M are
independent full-rank 4x4 matrices, so tying or rank-factorizing them is not an
exact element reduction.

## Candidate verification

`task344_compact_g_cost132.onnx` passed:

- official `score_and_verify`: correct, actual cost 132;
- independent team validator: valid, 266/266, actual cost 132;
- full ONNX checker and strict data-propagating inference;
- truthful runtime output `[1,10,30,30]`, mismatch 0;
- standard domain, no functions/subgraphs/sparse initializer/lookup/scatter;
- banned ops 0, Conv UB findings 0, runtime errors 0, nonfinite values 0;
- known 266/266 in ORT disable-all, basic, extended, and enable-all;
- minimum nonzero known ORT magnitude `3.415962`.

Fresh serialized-weight evaluation:

| seed | candidate / 10000 | authority / 10000 | candidate-authority sign differences |
|---:|---:|---:|---:|
| 344171501 | 9984 | 9984 | 0 cells |
| 344171777 | 9986 | 9986 | 0 cells |

The 30 rule failures are shared by the accepted authority; they are not caused
by the new absorption. Maximum sampled raw delta was `5.32e-5`.

## Why probe-only

The serialized float32 `G` differs from the real-valued target by at most
`3.2977e-8`. A coefficient analysis over supported colors gives:

- maximum logit-coefficient error: `2.80013e-5`;
- maximum total spatial weight for dimensions 3..10: `4.456994`;
- global real-logit error upper bound: **`1.24802e-4`**.

This is small, but the matching minimum absolute authority margin over the
entire reachable generator support was not proved. Therefore the candidate is
not described as exact for every float input, is not private-safe by fresh
testing alone, and is not auto-admitted. An isolated LB probe is required.

Machine-readable evidence is in `audit/final_audit.json`, `result.json`,
`winner_manifest.json`, and `audit/exact_build.json`.
