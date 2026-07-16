# task275 gold-exact diagonal-reuse result

- Authority: cost 428
- Candidate: cost 419
- Projected gain: +0.021252275660
- Candidate SHA-256: `c7ddaab77f6da011a99d233775ab02964f1a5e714f4dbb02045d1ecdda57c8e2`
- Official gold: exact pass
- Strict checker/static shape: pass
- Minimum positive raw margin: 38415.54296875
- Fresh seed 275419001: 2000/2000, errors 0
- Fresh seed 275419002: 2000/2000, errors 0
- Small positives in `(0, 0.25)`: 0
- Non-finite cases / runtime shape mismatches: 0 / 0
- Root submission/CSV/score-pointer writes by this lane: none

The final Einsum reuses the same learned 3x3 color map in both former T/W
roles and reads its diagonal with repeated subscript `aa` as the W-row scale.
This preserves the required sign-rank three while removing W's nine
parameters.  The spatial router is unchanged.

The earlier `task275_diag_color_cost413_351d0b2a8557.onnx` experiment is
rejected: it incorrectly collapsed quotient/remainder distractor colors and
failed official gold at train[1].  It is not an admission.
