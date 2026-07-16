# A20 — task191 / task216 strict audit

## Result

No candidate is safe to adopt. Projected gain is **+0.000000**. No root ZIP,
CSV, score ledger, or handcrafted artifact was changed.

The exact baseline is `submission_base_7999.13.zip`, SHA-256
`a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`.
Its actual costs are task191 = 3444 and task216 = 1511.

## Audit coverage

- Extracted both baselines directly from the exact archive.
- Audited all 8 retained SHA-distinct history models for each task.
- Also audited the existing generator-rule-derived task216 rebuild.
- Ran full ONNX checker, strict shape inference with data propagation,
  standard-domain/function/sparse/external checks, Conv-bias checks, giant
  Einsum and lookup checks, and declared-versus-runtime shape tracing.
- Ran every known example in both ORT `ORT_DISABLE_ALL` and default modes.
- Required zero session/runtime errors before any fresh validation.

## task191

Only two history models are actually cheaper:

| candidate | actual cost | possible gain | result |
|---|---:|---:|---|
| `task191_r07` | 3426 | +0.005240186664 | reject |
| `task191_r08` | 3430 | +0.004073325388 | reject |

Both pass all 267 known cases under `ORT_DISABLE_ALL`, but default ORT cannot
create a session after sanitizer normalization: `CenterCropPad` receives a
one-element shape for two axes. This violates the required both-ORT,
zero-session-error gate. Runtime shape tracing also rejects both: r07 cannot
be instrumented because it contains duplicate node names, while r08 has 27
declared/runtime mismatches. The other six variants cost 3819–3997 and are
not improvements.

The earlier 96.26% task191 result therefore was not promoted. It predates the
current both-ORT and no-shape-cloak requirements and cannot override these
structural/runtime failures.

## task216

None of the 8 history models is actually cheaper. Their actual costs are
1511, 1511, 1511, 1525, 1543, 1544, 1618, and 1534. The apparent static costs
769–864 are not official-like actual costs and come with extensive declared
versus runtime shape mismatches.

The existing rule-derived rebuild correctly implements the generator's
"select the rectangle with the most red pixels" rule and passes known
266/266 in both ORT modes. However, it costs 9135 and declares its dynamic
crop as `[1,2,18,18]` while the audited runtime crop is `[1,2,5,6]`. It is
both more expensive and a forbidden shape cloak.

## Fresh decision

Fresh 5000 was not run because no candidate passed the mandatory prerequisite
gates. This preserves the required order: known-all in both ORTs first, then
fresh validation only for candidates that remain structurally safe and
strictly cheaper.

Machine-readable evidence is in `model_manifest.json`, `history_audit.json`,
`extra_audit.json`, `audit.json`, and `winner_manifest.json`.
