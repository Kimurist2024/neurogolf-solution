# NeuroGolf GPT Session Handoff

## Current baseline
- ZIP: `submission_base_8000.46.zip`
- SHA-256: `74cb9c4a96dcf9d1d1ca4ad1e0afdb67f5218c0a06ad8e896ad1654b6548f534`
- Leaderboard score: `8000.46`
- Treat this ZIP as immutable. Build every candidate by replacing only intended task files.

## Hard acceptance gate
1. All train, test, and arc-gen examples must pass after output thresholding at `> 0`.
2. Standard ONNX domains only; no functions, subgraphs, sequences, or excluded operators.
3. Strict shape inference and ONNX Runtime execution must succeed.
4. Official-like `runtime memory + parameter elements` must strictly decrease.
5. Preserve archive order, all 400 tasks, and the 1.44 MiB per-file limit.
6. Run randomized differential testing against the baseline. Isolate any non-exact candidate.
7. Never batch a candidate that has previously hurt the leaderboard unless it is isolated again.

## Metric rules
- `cost = max-observed runtime bytes for all non-input/output tensors + initializer/Constant element count`
- `score = max(1, 25 - ln(cost))`; zero cost scores 25.
- Node count, MACs, FLOPs, and serialized ZIP size are not direct score terms.
- A longer graph with tiny tensors can beat a short graph with large tensors.

## Recommended workflow
```bash
python ngolf_validator.py compare-zips \
  --baseline baseline.zip --candidate candidate.zip \
  --data-dir neurogolf_2026_data --random-cases 500 \
  --out-json comparison.json --out-csv comparison.csv
```

## Latest local comparison
- Accepted tasks: `[13, 105]`
- Total projected gain: `0.025703390`

## What to give the next GPT session
- The current best ZIP and its leaderboard score.
- The candidate ONNX file or candidate ZIP.
- `comparison.json`, `comparison.csv`, and the latest signal ledger.
- A clear instruction to optimize only from the current best baseline.

## Proven strategy hierarchy
1. Direct-output synthesis that avoids scored intermediates.
2. Exact in-Einsum factorization with no new activation tensors.
3. Procedural generation of constants when activation memory is cheaper than parameters.
4. Packed integer/bitwise state and direct selection instead of materialized tables.
5. Optional-default input removal and exact scalar initializer deduplication.
6. Sanitized ORT-basic optimization, followed by official profiling and exact validation.
