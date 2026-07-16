# Lane B29 — task366 truthful-shape repair audit

## Outcome

No winner. The historical `cost=7646` candidate was repaired without changing its computation, but its truthful cost is `9465`, which is `1478` above the baseline task366 cost of `7987`.

The root submission was not modified.

## What was repaired

Source candidate:

- `others/2/1201/7120/task366_further_improved.onnx`
- SHA-256: `45540ecc35f21f681861dedee1b84a337a9316e8e291c131c87c5b721ee6242a`
- Declared score cost: `7281 memory + 365 params = 7646`
- Runtime audit: `107` declared/runtime shape mismatches and `8` undeclared intermediates

Truthful metadata repair:

- `task366_cost7646_truthful_metadata.onnx`
- SHA-256: `34a9ded3a87b87c41d4d73247cb070948f3d55cabbeeb4830319f4bc87c915c8`
- Runtime audit: `0` declared/runtime mismatches and `0` undeclared intermediates
- Truthful cost: `9100 memory + 365 params = 9465`

Five diverse executable probes traced all `763` runtime tensors. No tensor changed shape across those probes. The repair replaces only shape/type metadata; graph nodes, initializers, opsets, input names, and output names are byte-identical to the source candidate. Their computational fingerprint is the same: `8e05fd7a52e6beaf50d0fd7c47fe2d232b303b94cad1f4f42de4efebdebfa91c`.

## Validation

- Full ONNX checker: pass
- Strict shape inference with data propagation: pass
- All static dimensions positive: pass
- Standard-domain / banned-op / nested-graph audit: pass
- Conv bias undefined-behavior findings: `0`
- Lookup red flags: no `TfIdfVectorizer`, `Hardmax`, or giant `Einsum`; maximum node inputs `6`
- ORT optimizations disabled: raw output exactly equal on `5/5`, errors `0`, max absolute difference `0.0`
- ORT default: raw output exactly equal on `5/5`, errors `0`, max absolute difference `0.0`

Historical fresh evidence for the source computation was `4685/4757 = 98.4864%` correct among executable generated cases (`72` wrong). This exceeds the user-permitted 95% threshold, but it does not overcome the required cost gate.

## Decision

`REJECT_TRUTHFUL_COST_ABOVE_BASELINE`

The apparent `341`-cost improvement (`7987 -> 7646`) came from false or missing intermediate tensor shape declarations. Once all runtime shapes are represented truthfully, the candidate becomes `1478` cost worse than the baseline (`7987 -> 9465`). Per the lane assignment, fresh5000 dual-mode validation and external-validator500 were therefore skipped. No lookup, fixture correction, execution change, or remaining shape cloak was accepted.

Detailed machine-readable evidence is in `build_manifest.json`, `audit.json`, and `winner_manifest.json`.
