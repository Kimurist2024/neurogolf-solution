# task233 exact-alias audit

## Verdict

**REJECT.** The isolated rewrite is mathematically exact and reduces measured
cost from 7432 to 7431, but task233's current graph is not stable in the required
default-ORT complete-known gate.

## Exact rewrite

- `one_i8` and `audit_one_i16` are identical scalar TensorProto values after
  erasing only their names: same INT8 dtype, scalar shape, and value `1`.
- Both consumers of `audit_one_i16` were repointed to `one_i8`, then the duplicate
  initializer was removed.
- Candidate: `task233_exact_alias.onnx`
- Candidate SHA-256:
  `94cb98f706b7117336640949ca34ba92dc49c58a3bebd5708d90799ed160b123`
- Full checker and strict shape inference with data propagation pass.
- `verify_fix.py` smoke: fresh 100/100, runtime cost 7431.

## Rejection evidence

The mandatory dual-ORT complete-known run in `verification.json` found:

- disable-all: 266/266 correct, 266/266 raw-bitwise equal to the current model,
  runtime errors 0;
- default ORT: only 49/266 gold-correct at the checkpoint, 2 runtime errors;
  candidate/current raw outputs were equal on 264/266 cases.

The exact equality confirms the rewrite itself does not introduce a semantic
regression. It also confirms that it inherits the current model's default-ORT
instability. Because runtime0 and complete-known100 are unconditional promotion
requirements, the two-seed fresh run was stopped after 250 generated cases and
the candidate is not counted or integrated.
