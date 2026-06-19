# NeuroGolf ONNX Golf Playbook

Distilled from dozens of worker FAILURE_LOGs. Read this BEFORE building any net —
it lets you skip the trial-and-error that wastes attempts. Target runtime is
**ONNX Runtime 1.24 CPU with `ORT_DISABLE_ALL`** (no graph optimizations).

## 0. Scoring (what you are minimizing)

- `cost = (intermediate tensor memory in bytes) + (parameter count)`. `score = max(1, 25 - ln(cost))`.
- **The `input` and `output` tensors are FREE** (not counted). Every OTHER node-output tensor IS counted (memory = elements × dtype bytes, using the LARGER of static-declared vs ORT-runtime shape).
- So the game is: do the work on the smallest tensors (1-byte dtypes, cropped spatial), and make the heavy 10-channel one-hot BE the graph output (see §2.1).
- `try_candidate` compares your candidate's cost against the current incumbent and AUTO-PROMOTES only if strictly cheaper. Cutting raw **params** below the incumbent can promote even at equal memory.

## 1. ORT 1.24 + DISABLE_ALL dtype/op support matrix (MEMORIZE THIS)

Picking a wrong dtype = a failed build/run. The recurring truth:

| Op | Works on | DOES NOT work on (ORT 1.24 / checker) |
|---|---|---|
| **Add / Sub / Mul** | int32, int64, float32, float16 | **int8, uint8** (no kernel; checker also needs opset≥14, still flaky → avoid) |
| **Where** | **uint8**, float, int32, bool-cond | **int8** (no kernel) |
| **CumSum** | **int32, float16** only | int8, uint8 |
| **MatMul** | float16 (cheap), float32; int8 via **MatMulInteger** | plain int8/uint8 MatMul |
| **ReduceSum** | int32, float (axes = INPUT since opset 13) | **int8, uint8** |
| **ReduceMax / ReduceMin** | float32 (and int32); axes = **ATTRIBUTE until opset 18**, INPUT from 18 | **bool**, int8 unreliable |
| **MaxPool** | float (use fp16) | int8/uint8/bool |
| **Equal** | int8, uint8, int32, float (opset ≥ 11) | — |
| **Min / Max / Or / And / Not** | uint8, int8, bool, int32, float | — |
| **Slice / Concat / Pad / Cast / Transpose / Gather / Reshape** | all dtypes (Gather/GatherElements **indices must be int32/int64**) | **Pad on bool** sometimes fails shape-inference → Pad uint8/int8 instead, or accept bool output Pad (works at runtime) |
| **OneHot** | — | **no integer OneHot kernel** → emit one-hot via `Equal` vs an arange (see §2.1) |
| **ScatterND** | updates **float32**, indices **int64** | f16/int32 updates/indices rejected |
| **Conv** | float I/O (fp32 or fp16) | int — a Conv color-decode output is `[1,1,30,30]` f32 = 3600 B (often the floor) |

Rules of thumb:
- **Masks → bool or uint8**. Arithmetic on masks → cast to **int32 or fp16** (1-byte arithmetic is unavailable, so 1-byte tensors only help as *storage*, not as Mul/Add operands).
- **Prefix sums → fp16 triangular MatMul** (exact for integers up to ~2048), NOT CumSum-in-int32 (int32 is 4×). Triangular ones matrix `U`: `prefix = x @ U`; suffix = `x @ L`.
- **Index/coordinate grids → int32** (Gather needs int32/64; int8 arithmetic unsupported).
- Use opset domain `''` (ai.onnx). Opset **18** if you need ReduceMax/Min `axes` as an input; **13** is fine otherwise. `Equal` needs ≥ 11.

## 2. Cost-reduction patterns (the wins, in priority order)

### 2.1 Make the one-hot output FREE (biggest, almost always applies)
Never materialize a counted `[1,10,30,30]` one-hot. Instead carry a **single-channel
label grid** `lab[1,1,30,30]` (uint8/int8) and emit:
```
output = Equal(lab, arange[1,10,1,1])   # this Equal IS the graph output -> 0 bytes counted
```
Out-of-grid cells → set `lab` to a **sentinel** (e.g. 255 or -1) that matches no
channel ⇒ zero-hot outside the grid (no separate mask op needed).

### 2.2 Paint onto the input when output ≈ input
If output is "input with a few cells changed":
```
output = Where(mask[1,1,30,30], color_onehot[1,10,1,1], input)
```
`input` and the small `[1,10,1,1]` color vector are free; no `[1,10,30,30]` temp.

### 2.3 Collapse 10 channels to one grid EARLY
A `1×1 Conv` with weights `[0,1,..,9]` turns the one-hot input into a single
color-id grid (`color k → k`, background → 0). Do ALL spatial work on `[1,1,H,W]`,
never `[1,10,H,W]`. (Conv output is f32 `[1,1,30,30]` = 3600 B — usually the floor.)
For presence/in-grid masks: `ReduceMax` over channels, or a width/height-collapsing
Conv.

### 2.4 Crop to the generator's MAX grid, Pad back at the very end
The hidden set obeys the generator's size bounds (e.g. 10×10, 18×18, 21×21, ≤30).
Crop to that max, work small, `Pad` to `[1,*,30,30]` last. BUT: cropping the FULL
float `input` to `[1,10,k,k]` is often WORSE (still 10-channel float) — channel-reduce
FIRST (§2.3), then crop the single-channel grid. The crop bound MUST come from the
generator's randint ranges, never from the visible examples.

### 2.5 Integral images for rectangle queries
`CumSum` (int32 or fp16) twice → corner `Gather` (A−B−C+D) for O(1) rectangle sum /
emptiness / bbox. Cheaper than per-cell scans for box/region logic.

### 2.6 Keep every working tensor 1-byte where it is only stored/compared
bool / uint8 `[1,1,H,W]` masks. Only widen to int32/fp16 at the exact op that needs it.

### 2.7 Do NOT merge masks if it adds node outputs
The scorer counts EVERY node-output tensor. Fusing two masks that each feed one
consumer into one tensor is net-zero or worse. Measure, don't assume.

## 3. Margin & correctness (gate requirements)

- Scorer thresholds **raw > 0**. On-cells: clearly positive (bool `True`=1, or ≥1). Off-cells: **exactly 0** or non-positive. **NOTHING in the open interval (0, 0.25)** (a `model_margin_stable` failure / platform sign-flip).
- A **bool output** gives a clean margin (on=1, off=0) for free.
- fp16 is **exact** for small integers (counts ≤ 9, indices ≤ ~120, sums ≤ 2048) — safe for label grids and prefix sums. Do not use fp16 for large/accumulating values.

## 4. Structural floors (when to STOP, not burn attempts)

You usually CANNOT reach cost ≤ 2000 because of hard floors:
- The f32 Conv color-decode output `[1,1,30,30]` = **3600 B** (Conv needs float I/O).
- For 10×10-output tasks: the bool one-hot region or input crop floors near 1000–4000 B.
Once you hit these, further trims save < 400 B (≈ +0.03 score) at rising correctness
risk. **Stop** — per the brief's stop rules. A wrong cell ⇒ 0 for the whole task.

## 5. Workflow (do this, in order)

1. `\.venv/bin/python scripts/golf/brief.py --task N` and READ the generator at
   `inputs/arc-gen-repo/tasks/task_<hash>.py` (+ `common.py`). **COMPILE the spec**, do not fit examples.
2. Write a numpy reference; verify it on the visible examples AND **≥ 2000–3000 FRESH
   `generate()` instances** (vary the seed). 0 mismatches required — an example-fit net
   scores 0 on the private benchmark (a real −15 pt incident happened).
3. Translate to the smallest static-shape ONNX (onnx.helper) using §1–§2.
4. `\.venv/bin/python scripts/golf/try_candidate.py --task N --onnx PATH` — it validates
   size / banned-ops / static-shapes / gold / margin and AUTO-PROMOTES if strictly cheaper.
5. Iterate, but obey stop rules (§4): stop at floor, stop after a promotion + 3 non-improvements, stop after ~8 serious attempts if nothing promotes.

Banned ops: **Loop, Scan, NonZero, Unique, Script, Function, Compress, any *Sequence***. No nested graphs (no GRAPH/GRAPHS attributes). All tensor shapes static (no `dim_param`). ≤ 1.44 MB. Input is `[1,10,30,30]` one-hot, content top-left-anchored (NO position invariance needed; hidden grids ≤ 30×30).

## 6. Proven transform building blocks

- **Flood fill**: `MaxPool` (3×3 = Chebyshev radius 1) on a fp16 mask, re-mask to the region each step; N steps reach distance N. Per-step re-mask prevents leaking across gaps.
- **Bounding box**: row/col presence via `ReduceMax`, then `CumSum` forward+reverse thresholded `>0`, or `ReduceMin/Max` over a coordinate index grid (sentinel the absent cells).
- **Connected-component size / neighbor count**: `Conv` with a plus/box kernel on the mask.
- **Symmetry / mirror**: `Gather` with a reversed index `2S-1-i`, OR'd with the original.
- **Translate / relocate a block**: build 0/1 selector matrices via `CumSum`, then `Rsel @ grid @ Csel^T` (fp16) — pure move, disjoint support = no collision.
- **Color of a region**: masked `ReduceSum` (int32/fp16) → argmax, or dot with an arange.
- **Frequency / histogram**: `ReduceSum` over H,W per channel; rank via `count'[k']>count[k]` comparisons.
