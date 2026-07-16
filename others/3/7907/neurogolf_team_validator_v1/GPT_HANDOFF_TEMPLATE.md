# NeuroGolf GPT Session Handoff

Update the values below at the start of every new session.

## Current best

- Baseline ZIP: `UPDATE_ME.zip`
- Leaderboard score: `UPDATE_ME`
- SHA-256: run `python ngolf_validator.py audit-zip --zip UPDATE_ME.zip`
- Never optimize from an older ZIP unless performing an isolated A/B comparison.

## Non-negotiable rules

1. Validate every train, test, and arc-gen example.
2. Compare outputs after thresholding at `> 0`.
3. Use ONNX Runtime with graph optimization disabled for validation and profiling.
4. Sanitize all internal names before profiling.
5. Accept only a strict decrease in `runtime memory + parameter elements`.
6. Preserve one input, one output, static positive shapes, standard ONNX domains, and no subgraphs/functions/sequences.
7. Reject LOOP, SCAN, NONZERO, UNIQUE, SCRIPT, FUNCTION, and COMPRESS.
8. Keep every ONNX file below 1.44 MiB and preserve exactly 400 archive entries.
9. Run randomized differential checks against the current baseline.
10. Package risky rewrites alone. Do not batch until the leaderboard confirms them.

## Competition metric

```text
cost = parameter element count
     + sum of each non-input/output tensor's maximum observed runtime bytes
score = max(1, 25 - ln(cost))
```

Node count, MACs, FLOPs, execution time, and compressed ZIP size are not direct score terms.

## Proven optimization hierarchy

1. Whole-program direct-output synthesis.
2. Exact factorization inside the same output-writing Einsum.
3. Procedural generation of constants when cheaper than stored parameters.
4. Packed state and direct selection instead of Concat/Gather tables.
5. Optional-default input removal and exact scalar constant deduplication.
6. Sanitized ORT-basic cleanup followed by strict rescoring.

## Required artifacts for the next session

- Current best ZIP and leaderboard score.
- Candidate ONNX or candidate ZIP.
- Validator `comparison.json` and `comparison.csv`.
- Latest signal ledger and a list of leaderboard-positive/negative isolated tasks.
