# Task 178 Golf Brief

## Current Net
- path: `artifacts/optimized/task178.onnx`
- file size: 2275 bytes
- cost: 13056
- score: 15.522997
- memory: 12969
- params: 87
- nodes: 45
- value_info tensors after shape inference: 44
- local gold-correct: True

## Op Histogram

- Reshape: 6
- Pad: 5
- Greater: 5
- Slice: 4
- ReduceSum: 4
- Where: 3
- Sub: 2
- Abs: 2
- ReduceMax: 2
- And: 2
- Cast: 2
- CumSum: 2
- Transpose: 2
- Equal: 2
- MatMul: 2

## Targets

- cost 900: score 18.197605, delta +2.674608
- cost 314: score 19.250607, delta +3.727610

## Examples
- train: 5 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 3x3 -> output 3x1

input:
```text
111
222
111
```

output:
```text
1
2
1
```

### train[2]
input 3x3 -> output 1x3

input:
```text
346
346
346
```

output:
```text
346
```

### train[3]
input 3x5 -> output 1x4

input:
```text
23381
23381
23381
```

output:
```text
2381
```

### train[4]
input 4x2 -> output 3x1

input:
```text
22
66
88
88
```

output:
```text
2
6
8
```

### train[5]
input 6x4 -> output 4x1

input:
```text
4444
4444
2222
2222
8888
3333
```

output:
```text
4
2
8
3
```

### test[1]
input 4x9 -> output 1x5

input:
```text
112333884
112333884
112333884
112333884
```

output:
```text
12384
```

### arc-gen[1]
input 5x5 -> output 1x3

input:
```text
88818
88818
88818
88818
88818
```

output:
```text
818
```

### arc-gen[2]
input 7x1 -> output 4x1

input:
```text
9
9
9
2
2
8
9
```

output:
```text
9
2
8
9
```

### arc-gen[3]
input 1x8 -> output 1x4

input:
```text
22777344
```

output:
```text
2734
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 178 --onnx path/to/candidate.onnx
```
