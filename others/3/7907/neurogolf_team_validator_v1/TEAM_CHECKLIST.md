# Candidate acceptance checklist

## Baseline discipline

- [ ] Baseline ZIP filename, score, and SHA-256 recorded.
- [ ] Candidate was built from that exact ZIP.
- [ ] Only intended task entries changed.

## Model validity

- [ ] File is below 1.44 MiB.
- [ ] One input and one output.
- [ ] Standard ONNX domains only.
- [ ] No functions, subgraphs, sequences, or excluded operators.
- [ ] Strict shape inference passes with positive static dimensions.
- [ ] ONNX Runtime executes with graph optimization disabled.

## Correctness

- [ ] All train examples pass.
- [ ] All test examples pass.
- [ ] All arc-gen examples pass.
- [ ] Candidate output is thresholded at `> 0` for task correctness.
- [ ] Random differential threshold mismatches are zero for a strict candidate.
- [ ] Differential executability matches the baseline.

## Metric

- [ ] Runtime profile was collected over all known examples.
- [ ] Candidate memory is measured from maximum observed allocation per tensor.
- [ ] Parameter count includes scalar initializers and Constant attributes.
- [ ] Candidate total cost is strictly lower.
- [ ] Projected score gain is recorded.

## Submission

- [ ] Isolated ZIP built for risky or task-specific changes.
- [ ] Combined ZIP contains only previously proven or strict candidates.
- [ ] ZIP has exactly 400 ONNX entries.
- [ ] Archive order and all non-target files match the baseline.
- [ ] Kaggle result is added to the signal ledger before the next batch.
