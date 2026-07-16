# Lane B30 — task345 exact optimization

## Outcome

No strict score winner was found. The exact 8000.46 baseline member remains `cost=389` (`248 memory + 141 params`). The lane produced a fully standard, shape-truthful model at the same cost, but an equal-cost control is not a score improvement and was not adopted.

The root submission, ZIPs, CSVs, and shared score ledgers were not modified.

## Safe same-cost control

The authority model uses six negative-padding Conv nodes to extract rows. Starting from the exact baseline bytes, the lane swapped each Conv's data and weight operands: the shared packed kernel becomes the small Conv data tensor and the real input becomes a legal dynamic 30x30 Conv weight. Positive padding selects the same row. Keeping the baseline's pre-scaled gray coefficients removes the five `Mul` nodes used by prior legal cost-410 models.

- File: `task345_legal_swapped_prescaled_cost389.onnx`
- SHA-256: `dace8512d2882a34289713fa530cb2991c17164a6fbfb465f7e7f3051e28bbcb`
- Cost: `248 + 141 = 389`
- Negative Conv padding: `0`
- Declared/runtime shape mismatches: `0`
- Undeclared intermediates: `0`
- Full checker and strict data-propagating inference: pass
- Conv-bias UB findings: `0`
- Banned ops, nested graphs, sparse initializers, nonstandard domains, lookup ops: none
- Known ORT_DISABLE_ALL: `264/264`, errors `0`
- Known default ORT: `264/264`, errors `0`
- Official decoder gold: pass
- Baseline raw differential: exact `264/264` in each ORT mode, errors `0`, maximum absolute difference `0.0`

This safely improves the prior legal floor from 410 to 389, but does not beat the scoring baseline.

## Strict-cheaper search

Repository history contained 26 byte-distinct actual-scored models: 24 were known-correct, and the only model below 389 was the prior row-2 omission at cost 365. It is only `153/264` on known cases and is rejected before fresh validation.

The remaining exact reductions were exhausted as follows:

- Conv sharing: all six Conv nodes already share one 100-element packed kernel. The kernel is exact rank 1, but replacing it with 20 parameters requires a counted 400-byte runtime materialization; projected cost is 709.
- Sparse storage: sparse `Wpack`, `cfac`, `wfac`, and combined `cfac+wfac` probes all fail ONNX full shape checking for their Conv/Einsum consumers.
- Duplicate constants and CSE: no exact duplicate initializer and no duplicate full node signature exists. Repeated `q0`, `r9`, and zero rows are already reused directly by `Concat`.
- Initializer slice/diagonal reuse: deriving the 30-element `wfac` from the reversed `Wpack` channel needs 200 counted bytes to save 30 parameters, a net cost increase of 170. Slicing a zero saves one parameter but adds a four-byte tensor, a net increase of three.
- Decoder gauge: all 26 int32 scalar intermediates were constrained as possible outside-grid blank rows over all 16,540 legal generator cases. No exact decoder gauge exists.
- Bitwise/Cast carriers: every signed and unsigned 16-bit gray multiplier was screened for both Add and Xor carriers. Add had 10,606 information-preserving multipliers per signedness and Xor had 7,850, but exact modular interval solving found zero output-free rank-1 decoders. The theoretical cost-380 path is therefore unrealizable in this decoder family.

No standard, shape-honest model with actual cost below 389 survived the cost/structure gate. Consequently known→fresh5000→external500 candidate validation was not started; those expensive gates are reserved for strict-cheaper candidates.

## Decision

`NO_STRICT_CHEAPER_ADMISSIBLE_CANDIDATE`

Winner count: `0`; verified score gain: `0.0`.

Machine-readable evidence is in `audit.json`, `build_manifest.json`, `blank_reuse_search.json`, the 16-bit search JSON files, and `winner_manifest.json`.
