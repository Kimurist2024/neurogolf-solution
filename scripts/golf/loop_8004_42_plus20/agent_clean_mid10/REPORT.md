# clean-mid10 terminal report — tasks 370/182/201/251

## Result

**Admitted candidates: 0.  Projected gain: +0.000000.**

All four lanes terminate at the truthful-cost gate.  The immutable baseline is
`submission_base_8004.50.zip` (SHA-256
`63cb4c2abf794bb3cc0ceb531db907625c82638656e7d1ab29865d39b42a6cac`).
No ZIP, CSV, score file, or `artifacts/handcrafted` member was changed.

| task | incumbent cost / SHA-256 | exhaustive cheaper-history result | honest sound control cost / SHA-256 | score delta if substituted |
|---:|---|---|---|---:|
| 370 | 968 / `104eb571e60f99f9bc60ac45af60bf84e01cab014e1b804c312322d791230c9e` | 8 SHAs: six nonstatic; the two truthful actual profiles are 1965 and 2018, both dearer | 35645 / `3fbe03ad7bbe499ecf7545e790e63b75e9a9b27b18223260edf1f73d3471a00d` | -3.606132 |
| 182 | 957 / `7d3f800f43ba8d2bd224e70406e967ea005df8049301fa4fb03d2ba99564f828` | 6 SHAs: all nonstatic; four also have Conv-family bias UB | 169429 / `4c80813c6e5c0a047ff0debb35e4443766a5f58dc85294d0da80d46f75c83322` | -5.176386 |
| 201 | 793 / `bf3fee21c1539a1084b64fffb236b7ec79915d9316d83ff50704bd7a3e3b9560` | 3 SHAs: every one is TfIdf/Scatter lookup; one is additionally nonstatic | 7898 / `e256952ce10c0cec494e92117d9f593e8605752d28336ca80a98d949b1cd3488` | -2.298542 |
| 251 | 755 / `57f557717f6c9b582b0051519e722721bc5c904fd310252bcd4030f2df8d5c63` | 4 SHAs: actual 760/763 are not cheaper; actual 709/582 cannot create a default-ORT session because their CenterCropPad shape arity is invalid | 24708 / `708781b4d013035bff093dd8ea1d504def66989c0b29545d9097e3e0913e541a` | -3.488165 |

The four honest controls pass full ONNX checking, strict shape inference with
data propagation, both ORT modes, bias-UB=0, and complete known data:
task370 `266/266`, task182 `267/267`, task201 `266/266`, and task251
`266/266`, with zero runtime errors in each mode.  They are all 10–177 times
too expensive, so the staged policy correctly stops before two-seed fresh
testing.  The already-established task201 sound-control evidence additionally
records fresh `5000/5000` in
`scripts/golf/scratch_codex_plus10/wave2_sound/REPORT.md`.

## Decoded generator rules

- **task370 / `e8dc4411`**: recover the black origin sprite and the single
  colored direction hint, then repeat that exact sprite along the hint vector;
  generator flip/transpose and the true grid boundary must be respected.
- **task182 / `776ffc46`**: the colored sprite inside the gray 7x7 frame is a
  template; every unframed color-1 sprite with the same shape is recolored to
  the template color.
- **task201 / `846bdb03`**: recover two Conway sprites plus the framed yellow
  corners/side colors, undo the optional horizontal flip, and assemble the
  data-dependent cropped frame.  This task has historical private-zero lineage;
  the cheap TfIdf/Scatter family therefore cannot receive the required pass
  guarantee.
- **task251 / `a5313dff`**: for each fully in-grid red rectangle, paint its
  black interior ring blue while preserving the red border and red inner core;
  clipped rectangles remain unchanged.

task370 and task251 are outside the private-zero catalog.  task182 is not in
that catalog but is explicitly in the downstream-contamination ledger.  task201
is named by `SOUND_REBUILD_PROMPT.md` as historical black/private-zero lineage.
None of these statuses weakens the rejection: every nominal gain already fails
the earlier truthful-cost, structural, lookup, bias, or dual-runtime gate.

Full per-SHA evidence is in `result.json`; the reproducible no-promotion audit
is `audit_lane.py`.
