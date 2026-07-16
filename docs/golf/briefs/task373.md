# Task 373 Golf Brief

## Current Net
- path: `artifacts/optimized/task373.onnx`
- file size: 509 bytes
- cost: 513
- score: 18.759724
- memory: 480
- params: 33
- nodes: 2
- value_info tensors after shape inference: 1
- local gold-correct: True

## Op Histogram

- GridSample: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta -0.562119
- cost 314: score 19.250607, delta +0.490883

## Examples
- train: 2 shown
- test: 1 shown
- arc-gen: 3 shown, 69 remaining

### train[1]
input 2x6 -> output 2x6

input:
```text
333333
999999
```

output:
```text
393939
939393
```

### train[2]
input 2x6 -> output 2x6

input:
```text
444444
888888
```

output:
```text
484848
848484
```

### test[1]
input 2x6 -> output 2x6

input:
```text
666666
222222
```

output:
```text
626262
262626
```

### arc-gen[1]
input 2x6 -> output 2x6

input:
```text
999999
222222
```

output:
```text
929292
292929
```

### arc-gen[2]
input 2x6 -> output 2x6

input:
```text
222222
666666
```

output:
```text
262626
626262
```

### arc-gen[3]
input 2x6 -> output 2x6

input:
```text
222222
888888
```

output:
```text
282828
828282
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 373 --onnx path/to/candidate.onnx
```
