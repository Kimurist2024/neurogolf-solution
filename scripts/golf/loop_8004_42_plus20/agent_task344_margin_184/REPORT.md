# task344 compact-G cost132 full-support margin audit

## Verdict

`PROBE_ONLY`.  Do not promote the cost-132 compact-G candidate from
`agent_task344_rebase_171`.

The requested full-support sign-margin certificate does not exist for this
candidate.  A generator-reachable 10x10 input makes the accepted cost-137
authority positive and the compact-G candidate negative at output channel 8,
row 5, column 5.

## Reproduced ONNX Runtime disagreement

The same sign disagreement is reproduced with ONNX Runtime graph optimization
disabled, basic, extended, and fully enabled:

- authority: `+0.00024549689260311425`
- compact-G cost132: `-0.0001138478473876603`

The float64 contraction of the serialized float32 initializers independently
also has opposite signs (`+1.8326161921322637e-7` versus
`-2.3958608181828822e-6`).

## Why the witness is in full generator support

The audit records an explicit latent outcome of `task_d90796e8.generate`, not
an unconstrained four-color grid or a fresh random sample:

- dimensions are 10x10, within 3..10;
- the initial gray set is one exact Bernoulli subset;
- the 18 selected centers form one exact padded-grid Bernoulli subset;
- their minimum pairwise Manhattan distance is 3, so the generator's greedy
  spacing filter accepts every one;
- every accepted center has either the legal no-green outcome or one specified
  cardinal green direction;
- the outside center `(10,3)` legally draws its green cell inward to `(9,3)`.

All finite choices have strictly positive probability.  The proof script also
reconstructs the grid from those latent choices and verifies it with the task's
explicit `generate(width, height, rows, cols, colors)` path.

## Artifacts

- `prove_or_refute.py`: deterministic support and ONNX Runtime proof
- `audit/margin_counterexample.json`: complete machine-readable evidence
- `result.json`: fail-closed lane result

No root submission, score table, or other active stage was modified.
