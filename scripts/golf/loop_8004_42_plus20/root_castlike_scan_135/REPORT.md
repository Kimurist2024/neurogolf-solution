# Root CastLike witness scan 135

## Result

No candidate was admitted. The immutable 8009.46 authority, root submission,
score ledger, and `others/71407` candidates were not modified by this lane.

All400 authority models were scanned for an initializer whose only uses are the
second, type-witness input of `CastLike`. Forty-seven tasks matched. Replacing
each such node with `Cast(to=witness.dtype)` is an ONNX semantic identity and
allows the witness initializer to be removed.

Forty-six candidates lost deliberate inferred-shape cloaks and reprofiled above
their authority despite having fewer parameters. They were rejected before
runtime auditing. The sole nominal strict-lower survivor was:

| task | official cost | candidate SHA-256 | result |
|---:|---:|---|---|
| 071 | 188 -> 187 | `1abcca8e1b56070a40e8f2c86335b2af7b782148491a6d2c2f97c9991d7d2e6c` | rejected |

task071 passes full checker, strict shape inference/data propagation, and the
Conv-bias UB check. It is raw-equivalent and100% correct on known265 plus two
independent1500-case streams under default ORT. Under ORT_DISABLE_ALL, however,
every known and fresh case fails at the replacement Cast with a buffer reuse
shape mismatch (`{1} != {30}`). The authority has three existing runtime versus
declaration shape contradictions that the type-witness form happens to mask.
This violates the campaign's no-new-error and dual-runtime gates, so the file
was not staged.

Evidence: `build.json`, `audit_task071.json`, `build_candidates.py`, and
`audit_task071.py`.
