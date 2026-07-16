# Lane A9 — exact 7999.13 initializer/factor wave

## Result

No safe strict winner was found for tasks 008, 025, 062, 250, 268, 275,
and 308. Projected gain is `0.0`. No root ZIP, CSV, ledger, score pointer, or
shared handcrafted artifact was modified.

The authority is the exact `submission_base_7999.13.zip` archive with SHA-256
`a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`.

## New experiments

### task025 causal-mask rewrite

Four uses of the two-element finite mask `[0,-40]` were replaced by the
standard `Attention` optional-mask omission plus `is_causal=1`, removing two
parameters. The candidate passes full checking and strict shape inference and
nominally scores `474 -> 472` (`396+78 -> 396+76`). It is semantically wrong:
both ORT-disabled and default ORT produce **0/266** known-correct, 266 wrong,
and zero runtime errors. A finite directional mask here is not the same
operation as the causal flag, so the candidate is rejected.

The other task025 initializer relation is also unsafe. `sigV=[0,1]` and
`negsigK=[0,-240]` are proportional, but compensation needs a negative
`Attention` scale, which produces NaNs. Folding the repeated final `outscale`
into `vsgn` also cannot delete `outscale`: the same initializer is used by an
earlier `Attention` node.

### task268 boolean-anchor rewrite

Replacing `CastLike(...,_bool_like)` with standard `Cast(to=BOOL)` removes one
parameter and remains 266/266 known-correct with zero errors under both ORT
modes. It is not cheaper. The change exposes the full 30x30 boolean carrier to
the profiler: memory rises `399 -> 1298`, and actual cost rises **446 -> 1344**.
It is rejected before fresh validation.

The only historical task268 model that was both cheaper and complete-known
cost 327, but its existing mandatory fresh audit is only 2219/5000 (44.38%).
It remains generator-unsound and was not reconsidered for merge.

## Exact-hash review of the remaining targets

- task008: removing the int32 anchor is known-correct but costs 454 versus 431;
  the real cast tensors are more expensive than the parameter saved.
- task062: equal-valued constants have schema-required incompatible ranks.
  Reshaping/recreating them adds at least as much charged state as it removes.
- task250: the 4x4 absent/present code tables are full-rank arbitrary codes.
  Previous exact-hash harvest places the closest alternate known-correct model
  at cost 473, above the incumbent 468.
- task275: `MA != BA` and `MB != BB`; joint family ranks are full. Their common
  zero rows 16..29 cannot be trimmed without a charged reconstruction of the
  30-output axis. Historical cost-317 polynomial models fail known gold.
- task308: the incumbent already contains the newer 4x4-index plus reflect-pad
  reduction. The exact-hash historical audit found no lower known-correct
  member.

Fresh-5000 was not started for the new candidates because neither passed both
the strict-cheaper and complete-known gates. Machine evidence is in
`baseline_manifest.json`, `rejected_candidate_audit.json`,
`failure_manifest.json`, and `winner_manifest.json`.
