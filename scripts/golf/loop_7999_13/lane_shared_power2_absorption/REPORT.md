# Shared signed-power-of-two absorption audit

## Accepted task051 rewrite

- Exact source: `submission_base_7999.13.zip`
- Candidate: `task051_r01.onnx`
- SHA-256: `67222f7b552145fd354ebc0e39de2bdf333bde18f25e7f397b3dbb70b139e6c3`
- Actual cost: `283 -> 279` (`memory 80`, parameters `203 -> 199`)
- Projected gain: `+0.014235115821872`
- Known: `265/265` under both disabled and default ORT
- Exact differential: candidate and exact baseline are raw-bitwise identical on
  5,000 fresh cases in each ORT mode, with zero runtime errors
- External validator: raw and threshold equal `500/500`, `ACCEPT_STRICT`

The unique factor `ca=[1,-16,-1024,-2097152]` is absorbed into the shared
initializer `J1`.  The other `J1` use is compensated by dividing `cb` by the
same signed powers of two, leaving `[1,1,1,-1]`.  Binary exponent shifts are
exact for these float32 values, and the two contractions are algebraically
unchanged.  This removes four parameters and one operand from the incumbent's
existing final Einsum.  The candidate therefore introduces no new operation,
runtime shape, or execution path.

The exact baseline itself is not generator-perfect on the independent stream.
The candidate preserves every raw baseline output bit-for-bit and exceeds the
user-authorized 95% threshold, so it is admitted only through the same
exact-baseline-equivalence exception used for tasks 070 and 379.

## Whole-archive result

The metadata-preserving Wave 12 archive is
`../submission_7999.13_wave12_candidate_meta.zip`, SHA-256
`80563b1f8c59cf1c4b05561bcb9402298277a168ad555be4aa0dbe0c318e8ad7`.
It has 400 unique tasks, no missing/duplicate/oversized member, preserves the
Wave 11 metadata, and has zero Conv-family bias-length findings.  Root ZIP,
CSV, score ledger, and handcrafted artifacts were not modified.

## Other pass results

The first pass also rediscovered task333's exact `449 -> 447` sign absorption.
It remains separate from Wave 12 while the longer independent dual-runtime
audit finishes.  A signed-permutation change-of-basis pass produced zero
strictly cheaper candidates.
