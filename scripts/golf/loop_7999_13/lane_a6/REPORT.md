# Lane A6 — model-specific exact re-evaluation

## Result

No winner is retained; projected lane gain is **0.0**. The exact
`submission_base_7999.13.zip` was the only baseline used. No root ZIP, score
CSV, score pointer, or shared model was modified.

The strict gate order was structure/runtime, known gold, strict cost, then
fresh/domain 5000. No candidate reached the last gate: every probe either
failed structure/runtime, regressed cost, or tied the incumbent. This avoids
promoting an error task merely because its static parameter count is smaller.

## Pinned baseline recheck

Archive SHA-256:
`a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`.
All seven exact members passed the known verifier with zero runtime errors.

| task | member SHA-256 | memory | params | cost |
|---:|---|---:|---:|---:|
| 048 | `04455e6bef7fc9252df79102c3fbc914855199ae834adc356ecd2365a11d12a8` | 308 | 71 | 379 |
| 037 | `ed757e114a4471ab5851c6b2a31d0ad5a79b01b7ab6d9c160abd2746c5d873dc` | 368 | 6 | 374 |
| 264 | `d2be90cfe3692f3b10e6add8998eae5f4bb6ec2a66e306ac5f803199fe83046c` | 259 | 103 | 362 |
| 234 | `782e0b9f5b99509b834a78d41d5777e6e6a91107e16058d7a693ea090ccad7c6` | 236 | 132 | 368 |
| 397 | `55ff7a949d67d42993d1db2060666d344efa5b0d91a53dc45e779c6469006bfa` | 284 | 80 | 364 |
| 392 | `e68e69908e85933a9cc1367d8c8653560ed5b45188f0da3729a74e8d92491fb5` | 290 | 55 | 345 |
| 387 | `df39eba7bd5bb4a6fad8c97f0f5e05ae0392a4990e829cc4435ff05220324e28` | 242 | 95 | 337 |

## Candidate outcomes

- **task048 zero sharing:** `0-x` and `x>0` can remove the scalar zero by
  using `~(x-1)` and a bool cast. It is exact on known gold, but the new uint8
  intermediate raises memory 308 -> 309 while params fall 71 -> 70. Cost stays
  **379**, so it is rejected as a tie.
- **task037 true-rule fusion:** the four five-stage `Selu/PRelu` chains are
  exact binary ANDs and their terminal comparisons are OR/complement logic.
  The compact Min/Max graph passes full ONNX checking, but deleting the
  intermediates changes ORT's allocation schedule for the deliberately
  underspecified CenterCropPad shapes. It deterministically raises a Slice
  buffer mismatch, so it is an error candidate and is rejected.
- **task037 bool pipeline:** retaining node topology removes that runtime
  error and remains known-correct, but bool logical kernels expose the full
  spatial tensors to profiling. Cost becomes **437,668** versus 374.
- **task037 direct shape constant:** `[29,31]` is algebraically equal to the
  current PRelu result, but direct rank-1 inference conflicts with the rank-0
  shape cloak and twenty downstream CenterCropPad annotations. Full checking
  rejects the probe.
- **task392 Unsqueeze attribute:** an opset-12 axes attribute would remove
  three parameters, but opset 12 cannot type-check this graph's int8 `Sub`.
  Opset 13+ requires the axes input, so the incumbent is retained.

## Static/model-specific conclusions

- **task264:** the 81-element int32 `slot_map` is the charged part of the
  dynamic palette lookup. Expanding a 9-element row or generating it at
  runtime costs at least the same 81 runtime elements plus new intermediates;
  `color_vals` and `eq_vals` have incompatible dtype/shape roles.
- **task234:** `fg_pick_f` and `col_sel2_f` are row reversals, but both index
  the same packed tensor in one terminal Einsum, so sharing requires a charged
  axis-reversal tensor. Exhaustive contractions using up to four copies of the
  existing `term_U_order_f`/`term_V_order_f` did not exactly reproduce either
  selector. The apparent rank-1 moment matrix is only approximate in float32.
- **task387:** the 30 copies of uint8 254 duplicate a scalar value, but
  ScatterElements requires the full-rank data tensor. Runtime expansion costs
  the same 30 elements before its shape/index overhead. The coordinate arrays
  likewise require charged slicing/casting to share.
- **task397:** the known broadcast/runtime candidate was explicitly excluded.
  Archived alternatives re-score at 407, 409, 395, and 368 against incumbent
  cost 364; none is cheaper. No broadcast shortcut was rebuilt or adopted.

Machine-readable disposition is in `winner_manifest.json`; structural and UB
results are in `structural_audit.json`.
