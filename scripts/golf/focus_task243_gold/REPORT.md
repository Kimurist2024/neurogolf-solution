# task243 truthful-repair result

## Outcome

No task243 improvement is admissible over the current cost-145 authority.
The apparent cost-110 Shape/Reshape deletion is not a truthful ONNX model: it
retains 1-element declarations for tensors whose runtime shapes are 30x30 and
10x10.  Regenerating truthful metadata necessarily exposes those tensors to
the scorer.

The root submission, `all_scores.csv`, and `best_score.json` were not modified.

## Authority defect behind the apparent 110 cost

Current task243 declares:

- `L`: 1x1, but runtime is 30x30;
- `CB`: 1x1, but runtime is 10x10;
- `output`: 1x1x1x1, but runtime is 1x10x30x30.

`Shape(r0) -> Reshape(r0, Shape(r0))` is semantically an identity, but deleting
it while preserving those declarations merely reveals the stale shape cloak;
it does not create a valid cost-110 graph.

## Truthful constant repair

`candidates/task243_truthful_constant.onnx` materializes the actual fixed
relation matrices and retains the terminal Einsum.

- full checker: pass;
- strict shape inference with data propagation: pass;
- truthful input/output: 1x10x30x30;
- official gold exact: pass;
- visible margin: pass, minimum nonzero absolute value 1.0;
- profile: memory 0 + params 1010 = cost 1010;
- deterministic fresh seed 24320260715: **1988/2000**, 12 failures.

Therefore even the exact truthful counterpart of the current authority fails
the required fresh gate.  The inherited giant-Einsum rule is a finite-walk
approximation, not a complete flood-fill implementation.

## Requested-gate reference candidate

`candidates/task243_truthful_safe.onnx` was built as a truthful reference that
does pass the requested gates.  It splits the contraction, adds 16 propagation
steps, strengthens the blue seed, scales every fixed adjacency factor by the
exact positive value 7/8 to avoid float32 overflow, and stabilizes the final
threshold with `Sign`.

- full checker and strict static shape inference: pass;
- no banned ops, functions, nested graphs, dynamic dimensions, or shape cloak;
- official gold exact: pass;
- visible margin: pass, minimum nonzero absolute value 1.0;
- fresh: **2000/2000**, errors 0, nonfinite 0, shape mismatches 0, near-boundary
  values 0;
- profile: memory 72000 + params 1010 = **cost 73010**.

This is a validation/reference model only.  It is not an optimization against
cost 145 and was not promoted.  Fresh 2000/2000 is the requested empirical
gate, not a proof for every possible randomly generated grid.

## Evidence

- `build.json`: authority shapes and truthful constant construction.
- `base_failure.json`: the truthful cost-1010 candidate's failed fresh audit.
- `safe_build.json`: final reference construction.
- `evidence.json`: full official/static/margin/fresh audit and root hashes.
