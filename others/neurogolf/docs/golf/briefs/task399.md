# Task 399 Golf Brief

## Current Net
- path: `artifacts/optimized/task399.onnx`
- file size: 1744 bytes
- cost: 1184
- score: 17.923346
- memory: 912
- params: 272
- nodes: 8
- value_info tensors after shape inference: 7
- local gold-correct: True

## Op Histogram

- ReduceSum: 2
- Sub: 1
- Abs: 1
- Neg: 1
- ArgMax: 1
- Gather: 1
- Conv: 1

## Targets

- cost 900: score 18.197605, delta +0.274259
- cost 314: score 19.250607, delta +1.327261

## Examples
- train: 8 shown
- test: 3 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 5x5 -> output 3x3

input:
```text
22000
22000
00000
00000
00000
```

output:
```text
100
000
000
```

### train[2]
input 5x5 -> output 3x3

input:
```text
00000
02200
02200
00022
00022
```

output:
```text
101
000
000
```

### train[3]
input 7x7 -> output 3x3

input:
```text
0000000
0220000
0220220
0000220
0022000
0022000
0000000
```

output:
```text
101
010
000
```

### train[4]
input 6x6 -> output 3x3

input:
```text
000000
022000
022000
000000
002200
002200
```

output:
```text
101
000
000
```

### train[5]
input 3x3 -> output 3x3

input:
```text
000
022
022
```

output:
```text
100
000
000
```

### train[6]
input 7x7 -> output 3x3

input:
```text
0000220
0000220
0220000
0220220
0000220
0220000
0220000
```

output:
```text
101
010
100
```

### train[7]
input 7x7 -> output 3x3

input:
```text
0000220
0220220
0220000
0000022
2200022
2202200
0002200
```

output:
```text
101
010
101
```

### train[8]
input 7x7 -> output 3x3

input:
```text
0022022
0022022
2200000
2202200
0002200
0000000
0000000
```

output:
```text
101
010
100
```

### test[1]
input 6x6 -> output 3x3

input:
```text
000220
220220
220000
002200
002200
000000
```

output:
```text
101
010
000
```

### test[2]
input 7x7 -> output 3x3

input:
```text
0000000
2202200
2202200
0000022
0022022
0022000
0000000
```

output:
```text
101
010
100
```

### test[3]
input 7x7 -> output 3x3

input:
```text
2202200
2202200
0000022
0220022
0220000
0000220
0000220
```

output:
```text
101
010
101
```

### arc-gen[1]
input 7x7 -> output 3x3

input:
```text
0000000
0022000
0022000
2200000
2200000
0000220
0000220
```

output:
```text
101
010
000
```

### arc-gen[2]
input 3x3 -> output 3x3

input:
```text
000
220
220
```

output:
```text
100
000
000
```

### arc-gen[3]
input 5x5 -> output 3x3

input:
```text
00220
00220
00000
00000
00000
```

output:
```text
100
000
000
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 399 --onnx path/to/candidate.onnx
```
