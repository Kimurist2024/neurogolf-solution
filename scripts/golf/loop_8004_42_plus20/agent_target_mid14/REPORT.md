# target-mid14 report

## Outcome

No admissible improvement was found for task034, task374, task025, or task250.
The immutable baseline was `submission_base_8004.50.zip`, SHA-256
`63cb4c2abf794bb3cc0ceb531db907625c82638656e7d1ab29865d39b42a6cac`.
Projected gain is **0.0**. No ZIP, optimized/handcrafted model, score file, or
submission file was modified.

The exhaustive history source was the previously materialized union of 441
loose-sweep rows and 50 accepted-history rows (452 unique non-current
task/SHA pairs). For these four tasks it contains 0/1/2/2 non-current SHAs for
034/374/025/250 respectively. The full per-candidate ledger and all measured
checks are in `audit_results.json`.

## Generator truth

- **034 / `1f0c79e5`**: red corners of the 2x2 seed encode 1-3 diagonal
  directions; paint the seed and every in-grid sprout stripe with the sole
  non-red seed color.
- **374 / `ea32f347`**: color the three separated gray straight lines by their
  distinct lengths, shortest/middle/longest -> 2/4/1.
- **025 / `1a07d186`**: preserve full guide lines, erase sparse pixels, and
  project a pixel beside its matching-color guide on the same side, including
  the generator transpose branch.
- **250 / `a48eeaf7`**: locate the red 2x2 box and independently clamp every
  gray probe to its surrounding 4x4 frame.

None of the four is in the project private-zero catalog. The stronger
private-lineage 100%-fresh exception therefore does not apply.

## Decisions

| task | 8004.50 cost | baseline SHA-256 | policy-clean spec control | result |
|---:|---:|---|---:|---|
| 034 | 511 (250+261) | `1d13df745203958b5b477bf96b88ad73aefd1ad66971b132a569334d94da974c` | 3626 | baseline uses `ScatterElements`; no clean historical lower SHA and no competitive rebuild |
| 374 | 481 (451+30) | `93fb94260388ab83bc35043c0ee11ae08b1bf3e8fa962a3b47b08ba73794d24a` | 3361 | baseline declares 1x1 output for runtime 10x30x30 and uses cloak/lookup; honest graph is much larger |
| 025 | 474 (396+78) | `22a44063541ed6c696b535d4059abf5dd4844a0fafef2df6006b6f1e027ea595` | 370205 | baseline declares 1x1 output, uses `CenterCropPad`, and ends in a 25-input giant `Einsum` |
| 250 | 468 (279+189) | `bd479d52a359a4a1162387298b3f78691e7fa882a771afd5dda7313b92db6b0e` | 485 | baseline itself is clean; the only new cheaper graph fails known correctness |

Every listed control passes full checker, strict `data_prop` inference,
positive canonical static shapes, runtime shape truth, standard domains,
lookup/cloak/giant-Einsum exclusion, and Conv bias UB0. Each control also has
100% known correctness in both ORT modes with zero runtime errors:
034=267/267, 374=267/267, 025=266/266, 250=265/265. Their costs are nevertheless
above their respective immutable baselines.

## New task250 attempt

`task250_direct_roi_rc.onnx` (SHA-256
`4ec7005df2b9014c5178b730b43ef9eefeb71f3760e728c10454d690509b5d77`)
keeps the generator-derived ROI branch in fp32 instead of fp16->fp32. It is a
truthfully shaped, standard-domain graph with Conv bias UB0 and margin minimum
1.0. Measured cost falls **468 -> 464** (memory 279 -> 275, params unchanged
at 189; nominal score gain +0.008584).

It is rejected: both ORT modes produce only **33/265** known-correct examples
with runtime errors 0. The original integer carrier is semantically necessary
to quantize the power-coded box coordinate before `Resize`; deleting it moves
the ROI on 232 known cases.

Because the mandatory known-100 gate failed, running independent fresh
5000x2 cannot change the rejection and was intentionally skipped. Likewise,
the other three tasks have no candidate that is both policy-clean and strictly
cheaper, so no fresh confirmation was warranted. This is the requested early
no-prospect termination.

## Evidence files

- `audit_results.json`: exact costs, SHA-256 values, dual-ORT known counts,
  runtime-shape comparisons, structural checks, and all-history candidate rows.
- `result.json`: authoritative handoff summary.
- `build_task250_direct_roi.py`: deterministic rejected-candidate builder.
- `search_recode.py` / `synthesize_recode.py`: task034 separator searches; no
  <=3-op constant-free replacement using already-entitled scalar inputs exists.

No candidate is merge-authoritative.
