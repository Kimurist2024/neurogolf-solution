# Task 039 Golf Brief

## Current Net
- path: `artifacts/optimized/task039.onnx`
- file size: 1419 bytes
- cost: 1039
- score: 18.053986
- memory: 1020
- params: 19
- nodes: 17
- value_info tensors after shape inference: 16
- local gold-correct: True

## Op Histogram

- Slice: 2
- ReduceSum: 2
- Less: 2
- Cast: 2
- ArgMax: 2
- Reshape: 2
- Add: 2
- Concat: 2
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +0.143619
- cost 314: score 19.250607, delta +1.196621

## Examples
- train: 2 shown
- test: 1 shown
- arc-gen: 3 shown, 258 remaining

### train[1]
input 10x10 -> output 3x3

input:
```text
0000000000
0000000000
0000770000
0006886000
0078448700
0078448700
0006886000
0000770000
0000000000
0000000000
```

output:
```text
007
068
784
```

### train[2]
input 10x10 -> output 3x3

input:
```text
0000000000
0100001000
0036530000
0052260000
0062250000
0035630000
0100001000
0000000000
0000000000
0000000000
```

output:
```text
100
036
052
```

### test[1]
input 10x10 -> output 3x3

input:
```text
0000000000
0000000000
0000080000
0004484000
0088334000
0004338800
0004844000
0000800000
0000000000
0000000000
```

output:
```text
000
044
883
```

### arc-gen[1]
input 10x10 -> output 3x3

input:
```text
0000000000
0000000000
0000000000
0400804000
0087780000
0877770000
0077778000
0087780000
0408004000
0000000000
```

output:
```text
400
087
877
```

### arc-gen[2]
input 10x10 -> output 3x3

input:
```text
0000000000
0090000900
0005655000
0005776000
0006775000
0005565000
0090000900
0000000000
0000000000
0000000000
```

output:
```text
900
056
057
```

### arc-gen[3]
input 10x10 -> output 3x3

input:
```text
0000000000
0090000900
0007947000
0004779000
0009774000
0007497000
0090000900
0000000000
0000000000
0000000000
```

output:
```text
900
079
047
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 39 --onnx path/to/candidate.onnx
```
