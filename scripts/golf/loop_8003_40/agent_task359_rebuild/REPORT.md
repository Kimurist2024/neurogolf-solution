# task359 sound rebuild report

## Outcome

No ONNX candidate was emitted. The current 8003.40 member costs only **24**
(24 parameters, no counted intermediate) by using a 20-input giant `Einsum`.
That construction class is prohibited in this sound-rebuild lane. A truthful
standard-operator implementation has a structural floor far above 24.

## Decoded semantics

The exact expansion of `inputs/sakana-gcg-2025/raw/task359.py` is:

1. For output cell `(r,c)`, concatenate input row `r` and input column `c`.
2. Count each color in that concatenated sequence.
3. Return a maximum-count color, breaking ties by its first occurrence in the
   row-then-column sequence.

The readable Python rule and NumPy reference agree on all **266/266 known**
cases. The NumPy reference also agrees with the exact Python rule on
**5000/5000 fresh** inputs.

However, the raw rule agrees with fresh generator outputs on only
**4996/5000**. This is not a tie-break bug: one mismatch has raw color 3 with
count 10 versus generator color 4 with count 9. The other three expose the
documented row-first tie behavior. The raw solver is therefore a close
approximation to the generator's actual stripe-restoration transform.

The generator-exact rule is:

1. Fill every row with that row's mode to obtain a horizontal reconstruction.
2. Fill every column with that column's mode to obtain a vertical reconstruction.
3. Compare each reconstruction's total agreement with the noisy input.
4. Select the higher-agreement reconstruction.

This rule matches **266/266 known** and **5000/5000 fresh**, with zero generator
errors. Full mismatch diagnostics and the fresh shape distribution are in
`REFERENCE_AUDIT.json`.

## Why cost below 24 is not soundly reachable

A standard implementation requires per-color row and column histograms,
per-line argmax, orientation scoring/selection, and one-hot output assembly.
One `[1,10,30]` histogram alone contains 300 values: 1200 bytes as float32 or
600 bytes as float16. It is a counted intermediate and already exceeds the
entire incumbent cost by at least 25 times. The second orientation and later
selection stages increase the floor further.

No allowed single ONNX node can directly perform global color counting,
data-dependent row/column orientation selection, deterministic mode selection,
and full one-hot emission. Achieving zero intermediate memory here requires the
forbidden polynomial/lookup/shape-cloak classes.

## Decision

- Verdict: `NO_CANDIDATE_STRUCTURAL_FLOOR`
- Candidate ONNX: none
- Dual-ORT fresh validation: not run because no model passed the lower-cost
  structural gate
- Conv bias UB in baseline: 0
- ZIP merge: none
- Root ZIP, score JSON/CSV, and `LOOP_STATUS.md`: unchanged by this lane
