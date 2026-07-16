# Lane C34 — task009 sound rebuild audit at 8000.46

No candidate is promotable. Projected gain is **+0.0** and no aggregate or
root score file was changed.

## Exact baseline

- Archive: `submission_base_8000.46.zip`, SHA-256
  `74cb9c4a96dcf9d1d1ca4ad1e0afdb67f5218c0a06ad8e896ad1654b6548f534`.
- This archive is byte-identical to the former Wave17 aggregate.
- task009 SHA-256:
  `372fef762ffbc873f8c6ef0f3e2f59478773e17702f4129d5e7e9ce8c783bfaa`.
- Actual external-validator cost: **2619** = memory 2567 + params 52.

## Rule and correctness

The authoritative generator is `task_06df4c85.py`. It creates a spacing-2
line-grid and connects same-colored endpoint pairs along logical rows and
columns. The readable reference implements the exact raw step-3 set rule. It
passes all repository pairs: train 3/3, test 1/1, arc-gen 261/261, total
**265/265**.

The baseline independently passes fresh seed 93401:

- `ORT_DISABLE_ALL`: **5000/5000**, wrong 0, errors 0;
- default ORT: **5000/5000**, wrong 0, errors 0.

The external validator passes known **265/265**, reports zero warnings, and
confirms cost 2619.

## Structural gate and search conclusion

The incumbent is already a scalarized rule compiler. Its counted memory is
dominated by the unavoidable 900-byte final label tensor, 450 bytes of exact
30x30 row assembly, 505 bytes of scalar float decode plus uint8 casts, and the
row/column carry conditions. Boundary carries already omit the algebraically
redundant current-position nodes at both ends. Further local pruning changes a
valid border or extra-line case.

Strict audit passes full ONNX checker, strict inference with data propagation,
complete static typing for every node output, truthful declared/runtime
`[1,10,30,30]` shapes in both ORT modes, standard domain only, no functions or
sparse initializers, no nested graphs, no banned/lookup/Einsum ops, and **101
Conv nodes with zero biases**.

The only known strict-cheaper structural lead costs 2457, but its propagation
pruning scores only **4327/5000 (86.54%)** on independent fresh cases in both
ORT modes. The cost-2072 lead has 101 `TfIdfVectorizer` lookup nodes and is
forbidden. Neither is eligible. No new candidate was emitted.

## Evidence

- `reference_verification.json`
- `fresh_task009_base_{disabled,default}_5000.json`
- `external_validator_task009.json`
- `candidate_audit.json`
- `rejected_manifest.json`
- `winner_manifest.json`
