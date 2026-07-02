# Task 032 Golf Brief

## Current Net
- path: `artifacts/optimized/task032.onnx`
- file size: 2347 bytes
- cost: 3405
- score: 16.867000
- memory: 3336
- params: 69
- nodes: 19
- value_info tensors after shape inference: 18
- local gold-correct: True

## Op Histogram

- Sub: 3
- ReduceSum: 2
- Cast: 2
- Slice: 1
- Conv: 1
- Sign: 1
- Sqrt: 1
- Greater: 1
- CumSum: 1
- Sum: 1
- Mul: 1
- Where: 1
- ScatterElements: 1
- Equal: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +1.330605
- cost 314: score 19.250607, delta +2.383607

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 4x4 -> output 4x4

input:
```text
0409
0000
0460
1000
```

output:
```text
0000
0000
0400
1469
```

### train[2]
input 6x6 -> output 6x6

input:
```text
000009
000800
000000
400000
407800
407000
```

output:
```text
000000
000000
000000
400000
407800
407809
```

### train[3]
input 5x5 -> output 5x5

input:
```text
00010
03000
03012
60000
03000
```

output:
```text
00000
00000
03000
03010
63012
```

### test[1]
input 5x5 -> output 5x5

input:
```text
02043
50000
00600
52040
50000
```

output:
```text
00000
00000
50000
52040
52643
```

### arc-gen[1]
input 6x6 -> output 6x6

input:
```text
280503
000000
000000
200000
080003
200003
```

output:
```text
000000
000000
000000
200003
280003
280503
```

### arc-gen[2]
input 4x4 -> output 4x4

input:
```text
0005
0080
0005
6200
```

output:
```text
0000
0000
0005
6285
```

### arc-gen[3]
input 4x4 -> output 4x4

input:
```text
0010
0210
0206
0000
```

output:
```text
0000
0000
0210
0216
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 32 --onnx path/to/candidate.onnx
```
