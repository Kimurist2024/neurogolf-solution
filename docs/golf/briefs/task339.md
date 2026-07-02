# Task 339 Golf Brief

## Current Net
- path: `artifacts/optimized/task339.onnx`
- file size: 1021 bytes
- cost: 762
- score: 18.364053
- memory: 654
- params: 108
- nodes: 14
- value_info tensors after shape inference: 13
- local gold-correct: True

## Op Histogram

- Cast: 3
- ReduceSum: 2
- Slice: 1
- Sub: 1
- ArgMax: 1
- Add: 1
- OneHot: 1
- Unsqueeze: 1
- Gather: 1
- Mul: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta -0.166448
- cost 314: score 19.250607, delta +0.886554

## Examples
- train: 4 shown
- test: 1 shown
- arc-gen: 3 shown, 258 remaining

### train[1]
input 3x3 -> output 1x2

input:
```text
000
100
010
```

output:
```text
11
```

### train[2]
input 3x3 -> output 1x3

input:
```text
020
200
020
```

output:
```text
222
```

### train[3]
input 3x3 -> output 1x1

input:
```text
070
000
000
```

output:
```text
7
```

### train[4]
input 3x3 -> output 1x4

input:
```text
080
880
800
```

output:
```text
8888
```

### test[1]
input 3x3 -> output 1x5

input:
```text
440
404
004
```

output:
```text
44444
```

### arc-gen[1]
input 3x3 -> output 1x9

input:
```text
777
777
777
```

output:
```text
777777777
```

### arc-gen[2]
input 3x3 -> output 1x2

input:
```text
040
004
000
```

output:
```text
44
```

### arc-gen[3]
input 3x3 -> output 1x2

input:
```text
010
000
010
```

output:
```text
11
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 339 --onnx path/to/candidate.onnx
```
