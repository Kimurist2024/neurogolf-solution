# C5 SOUND wave final report

No strict winner was found for tasks 102, 109, 112, 134, 170, 222, or 245.
The accepted manifest is empty, projected gain is `+0.0`, and no root
submission or score artifact was changed.

The exact `submission_base_7999.13.zip` members were used as the only cost
baseline. Each raw generator was read directly and reduced to the following
rule before any graph work:

| task | exact cost | true rule / type | independent fresh | result |
|---:|---:|---|---:|---|
| 102 | 493 | fill interiors of square gray frames red; bounded component geometry | 2721/3000 | incumbent unsound; sound rebuild above floor |
| 109 | 406 | mirror the NW sprite four ways around the central cross; Type D | 5000/5000 | sound floor retained |
| 112 | 422 | complete four red reflections around the green 2x2 pivot; Type D | 5000/5000 | sound floor retained |
| 134 | 423 | find/downsample the scaled 3x3 megasprite and recolor it; Type D | 2977/3000 | incumbent unsound; sound rebuild above floor |
| 170 | 387 | downsample the megasprite and apply the separate palette box; Type D | 5000/5000 | sound floor retained |
| 222 | 380 | isolate the solid same-color rectangle from 16x16 noise; Type C | 2813/3000 | incumbent unsound; sound rebuild above floor |
| 245 | 387 | translate the red sprite into the green-corner 7x7 frame; Type D | 5000/5000 | sound floor retained |

All fresh runs above had zero runtime errors. The four sound incumbents were
rerun with independent seeds distinct from the earlier 3000-case screening.

## Algebraic probes

### task109

The incumbent's `ReduceL1` reduces H,W with p=1. Replacing it by the exactly
equivalent parameter-free `GlobalLpPool(p=1)` removes the two-element axes
initializer and passes library gold, official gold, margin (`11.0`), and a
fresh smoke test. Its profiled memory rises enough to change cost **406 ->
422**, so it is rejected at the terminal cost gate. The prior repository
metadata-only 405 model was deliberately not adopted; C5 requires a real
sound-rule/algebraic improvement rather than a harvested value-info shave.

### task112

`sign4_ch_i8` contains three identical `[-1,+1]` rows, but the first dimension
is not redundant: it replicates updates to match ScatterND indices
`[3,2,4,4,4]`. Shrinking it to one row produces updates `[1,2,4,4]` and a
runtime shape error on every known case. Both the cloaked and truthful probes
are rejected. Folding the constant shape scaffold also makes previously
hidden shapes statically visible and conflicts with downstream declarations,
so it fails the strict structure gate before scoring.

### task170 and task245

Both models contain fixed-input Shape scaffolding, but replacing it with a
literal initializer exposes the compact graph's hidden runtime tensor shapes.
The resulting graphs cannot pass full checker plus strict shape inference
without making declarations truthful, which raises them above their 387
baselines. task245's 2x2 sign matrix is rank one, but forming that outer
product introduces at least four bytes of counted runtime memory to save only
two parameters.

## Why the three unsound incumbents were not patched

- **task102:** correctness requires associating each interior cell with a
  complete 4x4, 5x5, or 6x6 square frame while rejecting nonsquare and
  nested/junk frames. The current learned two-QLinearConv detector misses
  279/3000. Honest channel extraction plus the multi-size frame detectors and
  rendering exceed cost 493 before all cases are covered.
- **task134:** the scale (2-6), location, megacolor, and output color are all
  input-dependent global quantities. The existing TfIdf/TopK/Resize shortcut
  misses 23/3000. This is the Type-D failure mode explicitly called out by the
  SOUND rebuild guide; no example-fit correction is admissible.
- **task222:** the output rectangle has width/height 2-8 and area 9-16, amid
  dense random colored noise. Exact selection needs color-specific row and
  column runs plus boundary/component consistency. The single rank-8 Einsum
  matcher misses 187/3000; increasing or retuning an analog matcher is not a
  proof of the global rectangle rule and cannot be accepted.

Evidence is in `baseline_fresh_audit.json`, the rejected probe models/builders,
and the empty `winner_manifest.json`.
