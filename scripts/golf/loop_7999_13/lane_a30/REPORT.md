# A30 strict optimization report — task068 / task383

## Outcome

No candidate is adoptable under the Wave16/current strict gate. Both current
members have real cost 172, but both get that cost through deliberately false
shape metadata. Task383 additionally uses a 21-operand terminal Einsum. The
lane did not preserve or further mutate those mechanisms.

| task | current SHA-256 | real cost | strict-safe historical floor | result |
|---|---|---:|---:|---|
| 068 | `7d86339dd896fa0445ec73f9d3151d098d0a2691a12fe5b1de6e7b6f5665b687` | 172 | 623 | no winner |
| 383 | `d0dde772dc57a600f8757bae491e6540c88ec7d16b97f8aafecb766b202c656d` | 172 | 3621 | no winner |

No root ZIP, score CSV, shared handcrafted artifact, or aggregate was changed.

## True rules

### task068 (`31aa019c`)

The random-mode generator always makes a 10x10 grid with six to nine distinct
foreground colors. Exactly one foreground color occurs once; every other
foreground color occurs two to seven times. The singleton is interior. The
output is zero except for a red 3x3 square centered on the singleton, with the
singleton's original color restored at the center.

This is a global frequency selection followed by a bounded local render. The
spec-derived reference matched all 266 stored train/test/arc-gen pairs and
5,000 independently seeded generator instances.

### task383 (`f1cefba8`)

The input contains a two-color rectangular box. The inner rectangle's
"barnacles" mark one or more rows/columns. The output restores the regular
outer/inner box, replaces the marked lines inside the box with the outer color,
and extends those rows/columns outside the box with the inner color. Duplicate
barnacle coordinates are generator-valid, so frequency-only color heuristics
are not complete.

This is a data-dependent bounding-box and line-extraction transform (Type D).
The readable reference matched all 266 stored pairs and 2,000 independently
seeded generator instances.

## Current-member rejection

### task068

The current graph declares output `[1,10,1,1]` while ORT produces
`[1,10,30,30]`. It has two `CenterCropPad` nodes, intentionally contradictory
value-info, and a final four-factor coded Einsum. It therefore fails the lane's
truthful-shape/no-cloak gate even though local gold execution reports cost 172.

### task383

The current graph declares output `[1,1,1,1]` while ORT produces
`[1,10,30,30]`. It has two `CenterCropPad` nodes and a 21-operand terminal
Einsum (equation length 75), so it fails both the truthful-shape and no-new-
unsafe-giant gates. This member is not an acceptable base for a small shave.

## Historical and rebuild search

`scan_history.py` deduplicated every loose task-named ONNX found under
`scripts/golf`, `artifacts`, and `others`:

- task068: 576 files / 30 unique hashes;
- task383: 559 files / 29 unique hashes.

Each unique model was checked with the full checker, strict shape inference,
static dimensions, declared and inferred output `[1,10,30,30]`, no
`CenterCropPad`, and no giant Einsum (12+ operands or equation length 60+).
The complete evidence is in `history_manifest.json`.

The best truthful/no-cloak/no-giant static cost was 623 for task068 and 3621
for task383. Static cost is already a lower bound before ORT runtime can expose
larger tensors. Neither can beat the cost-172 incumbents. The repository-wide
ZIP archive inventory independently retained eight static-under-172 task068
models and two task383 models; every one has false output metadata, and the two
task383 models are also giant-Einsum graphs.

For task068, the smallest honest graph's cost is dominated by materializing
the frequency selector and spatial/channel conjunction. A standard direct
rebuild cannot avoid a counted intermediate for combining dynamic channel,
row, and column state. For task383, even the smallest honest graph must derive
the box support, colors, and marked row/column profiles; the best found honest
factorization remains far over 172. Scatter/TopK pruning and initializer/Einsum
factor sharing cannot close these 451- and 3449-cost gaps without returning to
the rejected metadata cloak or a new unsafe giant contraction.

## Reproduction

```bash
.venv/bin/python scripts/golf/scratch_codex/task068/agent_spec/reference_check.py --fresh 5000
.venv/bin/python scripts/golf/scratch_codex/task383/reference.py
.venv/bin/python scripts/golf/loop_7999_13/lane_a30/scan_history.py
```

Expected scan summary:

```text
68  ... truthful_no_cloak_no_giant_static_below_172: 0 ... best: 623
383 ... truthful_no_cloak_no_giant_static_below_172: 0 ... best: 3621
```

Decision: `NO_WINNER` for both tasks; do not adopt any lane artifact.
