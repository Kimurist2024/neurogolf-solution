# Task 262 Golf Brief

## Current Net
- path: `artifacts/optimized/task262.onnx`
- file size: 903 bytes
- cost: 3770
- score: 16.765170
- memory: 3744
- params: 26
- nodes: 8
- value_info tensors after shape inference: 7
- local gold-correct: True

## Op Histogram

- Slice: 1
- ArgMax: 1
- Squeeze: 1
- Gather: 1
- Unsqueeze: 1
- Expand: 1
- Pad: 1
- OneHot: 1

## Targets

- cost 900: score 18.197605, delta +1.432436
- cost 314: score 19.250607, delta +2.485437

## Examples
- train: 4 shown
- test: 1 shown
- arc-gen: 3 shown, 3 remaining

### train[1]
input 3x3 -> output 3x3

input:
```text
005
050
500
```

output:
```text
333
444
222
```

### train[2]
input 3x3 -> output 3x3

input:
```text
005
005
005
```

output:
```text
333
333
333
```

### train[3]
input 3x3 -> output 3x3

input:
```text
500
050
500
```

output:
```text
222
444
222
```

### train[4]
input 3x3 -> output 3x3

input:
```text
050
005
050
```

output:
```text
444
333
444
```

### test[1]
input 3x3 -> output 3x3

input:
```text
005
500
050
```

output:
```text
333
222
444
```

### arc-gen[1]
input 3x3 -> output 3x3

input:
```text
005
500
050
```

output:
```text
333
222
444
```

### arc-gen[2]
input 3x3 -> output 3x3

input:
```text
500
050
005
```

output:
```text
222
444
333
```

### arc-gen[3]
input 3x3 -> output 3x3

input:
```text
050
500
005
```

output:
```text
444
222
333
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 262 --onnx path/to/candidate.onnx
```
