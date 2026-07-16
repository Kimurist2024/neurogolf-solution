# A22 task101/task158 strict rebuild audit

## Outcome

The exact immutable `submission_base_7999.13.zip` payload (SHA-256
`a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`)
has one local strict-gate winner:

| task | exact cost | candidate cost | known, both ORT modes | independent fresh, both modes | gain |
|---:|---:|---:|---:|---:|---:|
| 158 | 7815 | 7657 | 266/266, errors 0 | 5000/5000, errors 0 | +0.020424701742453515 |

This projects `7999.13 -> 7999.150424701743`. The retained model is
`sound/task158_spec_anchor13.onnx`, SHA-256
`24e58dbce2a09e42e4d5929d9970f77451e396ce7fea71ea0952dbb4165ab7c0`.

Task101 has no admissible winner. Its three cheaper archive histories have
actual costs 5672, 5688, and 5703, but all contain input-signature fixture
branches. The 5703 history also comes from an explicitly quarantined
`private0` lineage. Two generic-rule references pass known data but cost 7264
and 8284 and retain the same wide-fixture branch, so neither is admissible.

## task158 validation

The candidate is derived from the spec-exact task158 reconstruction, not from
the old private-zero model. Its only semantic-preserving reduction crops the
stride-2 anchor lattice from 14x14 to 13x13. The generator limits width to 25
and height to 26. Endpoint coordinates are therefore at most 24/25, and the
phase decoder's tile index `floor(coordinate / 2)` is at most 12; tile 13 is
unreachable.

Strict structural and runtime gates all pass:

- actual cost `7657 = 6736 memory + 921 params`, against exact base 7815;
- full ONNX checker and strict shape inference with data propagation;
- all inferred dimensions positive and static;
- runtime intermediate shapes match every declaration;
- standard opset-18 domain only, no functions, sparse/external data, banned
  operators, giant Einsum, TfIdf/Hardmax lookup, or giant initializer;
- exact Conv and QLinearConv bias lengths;
- known corpus 266/266 under `ORT_DISABLE_ALL` and default optimization,
  runtime/session errors 0 in both;
- independent seed 92215158 fresh generator test 5000/5000 under both modes,
  runtime/session errors 0 in both;
- all 33 generator-reachable `(height,width)` shapes from 14x15 through 26x25
  occurred in the independent fresh run.

The exact 7999.13 baseline scored only 4824/5000 on that same fresh set in each
ORT mode. Candidate admission is against generator gold, not baseline
equivalence.

## Archive history and residual risk

All 11 retained loose-history models were reprofiled by the current ORT
scorer. The apparent task158 static-cost leaders actually cost 7838--10024 or
more at runtime, so none beats the exact 7815 base. Several also fail every
known case under default optimization. This is why inventory static costs were
not used for adoption.

Task158 has an older official `Error processing ONNX networks` record whose
exact hash was not preserved. The documented private-zero task158 SHA is
`ac4075e4b0bfa900953817909714792607d0c660939e3a3b23790c41be113dee`,
which is not this candidate. The candidate's exact SHA has no known official
result, so it is locally sound but must remain an isolated submission until an
official run completes; it should not be silently batched or described as
leaderboard-white.

No root ZIP, score CSV, ledger, or handcrafted model was modified. Reproduction
evidence is in `model_manifest.json`, `audit_rows.json`,
`fresh_dual_5000.json`, and `winner_manifest.json`.
