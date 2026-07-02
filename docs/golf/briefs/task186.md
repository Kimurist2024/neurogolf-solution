# Task 186 Golf Brief

## Current Net
- path: `artifacts/optimized/task186.onnx`
- file size: 688 bytes
- cost: 346
- score: 19.153561
- memory: 300
- params: 46
- nodes: 11
- value_info tensors after shape inference: 10
- local gold-correct: True

## Op Histogram

- Sub: 3
- Cast: 2
- Slice: 1
- ReduceSum: 1
- Gather: 1
- Unsqueeze: 1
- Concat: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta -0.955956
- cost 314: score 19.250607, delta +0.097046

## Examples
- train: 10 shown
- test: 2 shown
- arc-gen: 3 shown, 252 remaining

### train[1]
input 3x3 -> output 3x3

input:
```text
000
100
000
```

output:
```text
200
000
000
```

### train[2]
input 3x3 -> output 3x3

input:
```text
010
100
000
```

output:
```text
220
000
000
```

### train[3]
input 3x3 -> output 3x3

input:
```text
001
000
100
```

output:
```text
220
000
000
```

### train[4]
input 3x3 -> output 3x3

input:
```text
010
001
000
```

output:
```text
220
000
000
```

### train[5]
input 3x3 -> output 3x3

input:
```text
001
000
000
```

output:
```text
200
000
000
```

### train[6]
input 3x3 -> output 3x3

input:
```text
110
000
100
```

output:
```text
222
000
000
```

### train[7]
input 3x3 -> output 3x3

input:
```text
010
110
000
```

output:
```text
222
000
000
```

### train[8]
input 3x3 -> output 3x3

input:
```text
110
000
101
```

output:
```text
222
020
000
```

### train[9]
input 3x3 -> output 3x3

input:
```text
010
110
100
```

output:
```text
222
020
000
```

### train[10]
input 3x3 -> output 3x3

input:
```text
100
001
011
```

output:
```text
222
020
000
```

### test[1]
input 3x3 -> output 3x3

input:
```text
010
000
010
```

output:
```text
220
000
000
```

### test[2]
input 3x3 -> output 3x3

input:
```text
010
011
100
```

output:
```text
222
020
000
```

### arc-gen[1]
input 3x3 -> output 3x3

input:
```text
000
000
010
```

output:
```text
200
000
000
```

### arc-gen[2]
input 3x3 -> output 3x3

input:
```text
000
001
000
```

output:
```text
200
000
000
```

### arc-gen[3]
input 3x3 -> output 3x3

input:
```text
101
000
001
```

output:
```text
222
000
000
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 186 --onnx path/to/candidate.onnx
```
