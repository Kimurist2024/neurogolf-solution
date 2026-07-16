# task046 residual audit — no safe strict-lower candidate

## Outcome

`NO_SAFE_STRICT_LOWER`. The exact task046 authority in
`submission_base_8009.46.zip` remains unchanged at cost **627**
(memory 397 + params 230), member SHA-256 `fb649383229d5cdcb562b8c1ce52256ff344193810888b795c20ac0aa0660d77`.
No candidate ONNX or ZIP was emitted, and nothing was staged.

## Authority and runtime evidence

- Full ONNX checker and strict data-propagating inference pass. All inferred
  node-output shapes are static, positive, and match the cost calculation.
- Standard domains only; lookup ops, subgraphs, functions, sparse
  initializers, unused/duplicate initializers, dead outputs, and Conv-family
  UB findings are all absent.
- Known 267 cases pass in all four configurations
  (disable-all/default × threads 1/4), with runtime errors 0 and output
  nonfinite values 0. The raw uint8 output digest is identical in all four:
  `0119a4d354f83652d739bd2ccbcdefe7aa20b09ab5cc06007c9a75b469c89532`.
- Fresh 512 cases from `task_234bbc79` pass in all four configurations with
  runtime errors 0 and output nonfinite values 0. Their raw output digest is
  likewise configuration-stable:
  `e29582cefa15abd19e88c14da02e23ad2025f53c1f291ae83c7304d00b47c35b`.

The inherited nonfinite trace is precisely localized. For three-segment
inputs, optional fourth-segment value `c4_rep` is zero, so `c4_log` contains
one `-Inf`; `QuantizeLinear` produces finite uint8 `c4=0`. This occurred in
140/267 known cases and 265/512 fresh cases, identically in all four runtime
configurations. There were no NaNs, positive infinities, other nonfinite
tensors, or nonfinite outputs. The audit introduced no new behavior.

## Residual search

The current SHA is the earlier 631→627 probe. Its four-byte reduction removed
three left-row masks and replaced `and2_sm0` by the broadcastable `two_u8`
initializer. A direct comparison against the former SHA `71aae814...` found
raw output equality on all 267 known cases. That support-dependent rewrite is
already the LB-white authority; it was not generalized into another
empirical-only edit.

No further exact cleanup remains in the requested classes:

- Dead/unused nodes or initializers: 0; duplicate initializers/nodes and CSE
  groups: 0; unconditional identities: 0.
- Default-attribute sites: 0; the full fusion/cleanup pass suite changes
  nothing. `Add`→`Sum` is invalid because this graph uses uint8 arithmetic.
- All scalar initializers already cost one element. Dense `code_w`, `flat_w`,
  and `flat_b` shapes are fixed by their Conv contracts. Sparse reconstruction
  of `flat_w` costs 760, above 627.
- UINT8/BOOL intermediates are already one byte. INT32 tensors are required
  GatherElements indices. FLOAT is tied to the float32 input and
  ConvTranspose; narrowing it would require a 9000-element cast and would not
  be all-input raw-equivalent.
- No scalar node output was constant across known+fresh traces. In particular,
  every remaining `and2_sm1`…`and2_sm15` took both 0 and 2, so the exact
  `and2_sm0` replacement cannot be repeated.
- The historical task046 inventory contains 65 distinct non-authority SHAs;
  its minimum static cost is 648, with zero models below 627. Earlier exact
  scanners also emitted no task046 record.

Detailed reproducible evidence is in `authority_audit.json`; the compact
decision record is `result.json`.
