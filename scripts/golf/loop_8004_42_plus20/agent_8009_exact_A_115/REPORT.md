# 8009.46 exact golf A — final report

Authority: `submission_base_8009.46.zip` SHA256 `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`.
No root submission, score file, or `others/` artifact was modified.

## Outcome

**No admissible strict-lower winner was found.**

## Inventory conclusion

task153 was the only authority member with zero inferred/runtime shape mismatch. It contained no dead node, initializer alias, no-op, CSE, optional-output, safe constant-fold, identity-Einsum, permuted-alias, rank-1, or dictionary-factor opportunity.
For task153, exhaustive int8 coefficient search also proved that deleting the two shifted-square Add nodes cannot preserve the existing single-QLinearConv color decoder (colors 1, 2, and 8 become non-separable).
The other eleven authority members either showed inferred/runtime shape mismatches or could not complete an all-intermediate trace. All 18 emitted rewrite/normalization controls failed structural or official-runtime gates before fresh admission. They were rejected; the LB-white member was never replaced by an older payload.
No candidate reached `FRESH_PENDING`, so no fresh run was warranted.

Per-task hashes, official costs, opportunities, candidate rejection stages, and shape evidence are in `audit/result.json`.
