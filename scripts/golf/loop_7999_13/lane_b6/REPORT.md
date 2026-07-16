# B6 exact-factor wave report

## Outcome

No candidate was accepted. The exact `submission_base_7999.13.zip` baseline remains unchanged, with projected score **7999.13** and projected gain **+0.000000**.

This lane inspected tasks 075, 159, 200, 218, 225, 228, and 388. It scanned 185 byte-distinct historical models and audited initializer sharing, local coefficient reconstruction, simple operator fusion, and attribute substitution against the exact archive members.

## Immutable baseline

- Archive: `submission_base_7999.13.zip`
- SHA-256: `a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`
- Root ZIP/CSV/ledger files modified by this lane: **no**

| Task | ZIP member | SHA-256 | Actual cost | Memory | Params |
|---:|---:|---|---:|---:|---:|
| 075 | 383 | `ea2280d32f09e571182c0dbae57155a7e2b8a23a88d0a027ae9add3c9770ceb8` | 345 | 311 | 34 |
| 159 | 145 | `806b248eaeadd82509800843ee9b5327ea7fdd294bfb3c82f94d700937a67634` | 293 | 146 | 147 |
| 200 | 184 | `8c91d1a61ac9bdd4cb5e812b5f0db57e0bf49c1cf2fd256724960118430df8f3` | 346 | 200 | 146 |
| 218 | 398 | `6740bffa1e0434430c998a6a7b1b05251258071f2b741b362d00c53d86934113` | 329 | 260 | 69 |
| 225 | 205 | `c55b5673a1e36b07114e82a629d23b01cefbd7b56289ad314b272d7180ef8a4a` | 333 | 233 | 100 |
| 228 | 208 | `13263c210602e01b0c1940efc8704eea208066bf87448b7a20d132c68c53f51a` | 302 | 252 | 50 |
| 388 | 355 | `f4450fa21dfce9e893c6b70646d43590ff60fb02b3d6a21856c97438061945b3` | 311 | 291 | 20 |

## Historical scan

| Task | Distinct models | Runtime OK | All-known correct | Cheaper | Cheaper and correct |
|---:|---:|---:|---:|---:|---:|
| 075 | 28 | 28 | 19 | 0 | 0 |
| 159 | 35 | 35 | 26 | 3 | 0 |
| 200 | 20 | 20 | 19 | 3 | 2 |
| 218 | 29 | 22 | 21 | 1 | 0 |
| 225 | 26 | 24 | 23 | 1 | 0 |
| 228 | 26 | 25 | 24 | 1 | 0 |
| 388 | 21 | 21 | 19 | 1 | 0 |

The two apparently viable task200 models both cost 345 and pass all known examples, but both are structurally unsafe:

- `artifacts/handcrafted/task200.onnx` (`b9036595...`)
- `others/2/7501/task200_improved_cost345.onnx` (`a54e5330...`)

Each replaces the valid two-element Conv bias with a one-element initializer while the Conv has two output channels. The host audit reports `('Conv', 'conv_bias', 1, 2)`. This depends on out-of-bounds or uninitialized memory and can vary by runtime, allocation, or execution order. Both candidates were therefore rejected before fresh testing; a nominal +0.002894 score gain cannot justify an error-prone model.

## Per-task result

- **075:** no historical model below cost 345 exists. The remaining Slice constants are proportional but cannot be shared without adding more output memory than the saved parameters.
- **159:** three cheaper models all fail known cases. The exact baseline already uses the lower-memory Div formulation. The structured Eq matrices can be related only through a transform whose retained parameter count removes the possible saving.
- **200:** the only two cheaper known-perfect models are the unsafe one-element Conv-bias variants above. The safe cost-344 sharing experiment fails known cases because it changes basis semantics.
- **218:** the sole cheaper model fails known cases. Sharing row/column rank vectors requires a runtime transpose or reshape path whose memory cost exceeds the initializer saving.
- **225:** the sole cheaper model fails known cases; no valid initializer or operator sharing was found.
- **228:** the sole cheaper model fails known cases. Known exact direct reconstructions are more expensive than cost 302.
- **388:** sharing `[1,2,3]` between ReduceL2 and Slice would nominally save three parameters, but ONNX full checking rejects it: ReduceL2 uses int64 axes while this Slice's starts/ends/axes index family is int32. Converting all Slice indices to int64 costs more memory than it saves.

## Gate disposition

Strict admission requires all-known complete, fresh/domain 5000/5000, zero runtime errors, and zero structural/UB findings. No candidate passed the prerequisite known-plus-structural gates, so fresh 5000 evaluation was intentionally not run on rejected models. `winner_manifest.json` is therefore empty.
