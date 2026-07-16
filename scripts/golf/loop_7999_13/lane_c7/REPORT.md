# C7 sound rebuild report

Reference: exact `submission_base_7999.13.zip` only  
Reference SHA-256: `a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`

## Result

No candidate is eligible for promotion. Projected gain is **0.0**.

Every retained model had to be strictly cheaper than the exact ZIP member,
exact on the complete known set and dual gold, independently fresh-exact with
zero errors, valid under default ORT and `ORT_DISABLE_ALL`, and free of unsafe
structural/profile behavior. No explored or harvested model met all gates.

| task | raw generator rule | exact cost (memory + params) | independent baseline audit | disposition |
|---:|---|---:|---|---|
| 029 | crop the unchanged interior of the unique complete monochrome rectangle | 270 (239 + 31) | disable-all 3000/3000, errors 0; default ORT session creation fails | no safe cheaper candidate; exact member itself is default-ORT unsafe |
| 091 | crop the two glowstick boundary columns and the cyan contents between them | 265 (249 + 16) | disable-all 3000/3000; default 100/100; errors 0 | frozen at verified compact ROI floor |
| 301 | sort horizontal bars by length, place ascending at bottom and right-align | 240 (118 + 122) | disable-all 3000/3000; default 100/100; errors 0 | no cheaper structurally safe exact implementation |
| 316 | sort colored pixels by input column and emit them in a 3x3 snake | 246 (131 + 115) | disable-all 3000/3000; default 100/100; errors 0 | prior exhaustive projection/zero-point reductions remain dominated or incorrect |
| 341 | fill the cyan bridge between the two horizontal colored blocks | 260 (231 + 29) | disable-all 3000/3000; default 100/100; errors 0 | current redesign already beats archived models; further carrier/axis shaves fail |
| 355 | return the base color of the partition block containing the most common sparse marker color | 250 (228 + 22) | disable-all 2966/3000, wrong 34; default 99/100, wrong 1; errors 0 | exact member is unsound; one-byte shaves are also known-inexact, so none accepted |
| 357 | paint the width-dependent cyan field and bouncing blue path | 258 (198 + 60) | disable-all 3000/3000; default 100/100; errors 0 | no removable guard column or safe lower-state formulation found |

Full fixed-seed records and first failures are in
`baseline_fresh_audit.json`. Seeds are `799913000 + task` for disable-all and
`800913000 + task` for default ORT.

## Existing-candidate reevaluation

The repository-wide harvest inventory was filtered to these seven task IDs.
It found no strictly cheaper eligible model:

- task029 historical models have static cost floors 375--5288, above 270.
- task091 historical candidates complete-profile at cost 266 and 270, above 265.
- task301's lower-parameter correct implementation costs 1141; another compact
  artifact is rejected for a 51-label giant `Einsum` structural hazard.
- task316's historical model has static floor 406, above 246.
- task341 historical models have static floors 280 and 1427, above 260.
- task355's harvested best only ties cost 250; the known cost-249 probes pass
  264/267 and make three distinct generator color errors.
- task357 historical models have static floors 270 and 279, above 258.

The task091 opset-axes audit identified a theoretical one-parameter axis
initializer saving, but downgrading the graph to opset 17 is invalid because
the model uses `GroupNormalization`, which is unavailable at that opset.

## New task357 search

The final model uses a 16-entry column code followed by `QLinearConv`. Seven
strictly smaller candidates removed one through seven right guard entries and
increased convolution padding by the same amount. All models pass full ONNX
checking and strict shape inference, but all fail the generator rule:

- removing one guard entry passes only 14/100 fixed fresh cases and creates a
  color-8 ghost column for widths 3--10;
- removing two through seven entries passes 0/100.

Varying the retained tail code cannot repair the conflict. The terminal three
tap classifier needs the four existing right-side guard codes. Alternative
dynamic output-zero-point formulations were also ruled out: the present
`5 * width - 72` threshold is required to separate both the cyan base field
and blue path while keeping convolution padding black.

Rejected probes remain isolated as
`task357_drop_guard_1.onnx` through `task357_drop_guard_7.onnx`; none is listed
in `winner_manifest.json`.

## Safety conclusion

There is deliberately no winner rather than promoting a high-rate but unsound
model. task029 must be rebuilt to remove its optimizer-dependent
`CenterCropPad` shape cloak, and task355 needs a true partition/count rebuild;
minor shaving of either model cannot satisfy the zero-error acceptance gate.
