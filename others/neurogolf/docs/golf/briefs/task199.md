# Task 199 Golf Brief

## Current Net
- path: `artifacts/optimized/task199.onnx`
- file size: 2948 bytes
- cost: 16157
- score: 15.309891
- memory: 16103
- params: 54
- nodes: 22
- value_info tensors after shape inference: 21
- local gold-correct: True

## Op Histogram

- Cast: 3
- ReduceSum: 3
- Slice: 2
- ReduceMax: 2
- Equal: 2
- And: 2
- Pad: 2
- Where: 2
- Conv: 1
- Greater: 1
- CumSum: 1
- Mul: 1

## Targets

- cost 900: score 18.197605, delta +2.887714
- cost 314: score 19.250607, delta +3.940716

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 3x3 -> output 3x3

input:
```text
020
000
000
```

output:
```text
040
020
000
```

### train[2]
input 5x5 -> output 5x5

input:
```text
00000
00000
00600
00000
00000
```

output:
```text
40404
40404
40404
00600
00000
```

### train[3]
input 9x9 -> output 9x9

input:
```text
000000000
000000000
000000000
000000000
009000000
000000000
000000000
000000000
000000000
```

output:
```text
404040404
404040404
404040404
404040404
404040404
009000000
000000000
000000000
000000000
```

### test[1]
input 12x12 -> output 12x12

input:
```text
000000000000
000000000000
000000000000
000003000000
000000000000
000000000000
000000000000
000000000000
000000000000
000000000000
000000000000
000000000000
```

output:
```text
040404040404
040404040404
040404040404
040404040404
000003000000
000000000000
000000000000
000000000000
000000000000
000000000000
000000000000
000000000000
```

### arc-gen[1]
input 11x11 -> output 11x11

input:
```text
00000000000
00000000009
00000000000
00000000000
00000000000
00000000000
00000000000
00000000000
00000000000
00000000000
00000000000
```

output:
```text
40404040404
40404040404
00000000009
00000000000
00000000000
00000000000
00000000000
00000000000
00000000000
00000000000
00000000000
```

### arc-gen[2]
input 4x4 -> output 4x4

input:
```text
0000
5000
0000
0000
```

output:
```text
4040
4040
5000
0000
```

### arc-gen[3]
input 4x4 -> output 4x4

input:
```text
0000
1000
0000
0000
```

output:
```text
4040
4040
1000
0000
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 199 --onnx path/to/candidate.onnx
```
