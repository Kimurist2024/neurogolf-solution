# Task 188 Golf Brief

## Current Net
- path: `artifacts/optimized/task188.onnx`
- file size: 1765 bytes
- cost: 8103
- score: 16.000010
- memory: 8056
- params: 47
- nodes: 43
- value_info tensors after shape inference: 42
- local gold-correct: True

## Op Histogram

- Where: 7
- Slice: 5
- Cast: 4
- ReduceSum: 4
- Less: 4
- Sum: 4
- ReduceMax: 3
- Div: 2
- Greater: 2
- Sub: 2
- Abs: 2
- Not: 2
- Mul: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +2.197595
- cost 314: score 19.250607, delta +3.250597

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 4x8 -> output 4x4

input:
```text
11321132
11331133
33113311
23112311
```

output:
```text
1132
1133
3311
2311
```

### train[2]
input 3x6 -> output 3x3

input:
```text
444444
648648
668668
```

output:
```text
444
648
668
```

### train[3]
input 6x2 -> output 3x2

input:
```text
23
32
44
23
32
44
```

output:
```text
23
32
44
```

### test[1]
input 8x3 -> output 4x3

input:
```text
545
454
664
262
545
454
664
262
```

output:
```text
545
454
664
262
```

### arc-gen[1]
input 4x4 -> output 2x4

input:
```text
1344
3331
1344
3331
```

output:
```text
1344
3331
```

### arc-gen[2]
input 3x4 -> output 3x2

input:
```text
8888
8787
8787
```

output:
```text
88
87
87
```

### arc-gen[3]
input 3x4 -> output 3x2

input:
```text
6767
1616
1616
```

output:
```text
67
16
16
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 188 --onnx path/to/candidate.onnx
```
