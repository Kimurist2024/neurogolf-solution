# Task 272 Golf Brief

## Current Net
- path: `artifacts/optimized/task272.onnx`
- file size: 2039 bytes
- cost: 460
- score: 18.868774
- memory: 0
- params: 460
- nodes: 1
- value_info tensors after shape inference: 0
- local gold-correct: True

## Op Histogram

- Conv: 1

## Targets

- cost 900: score 18.197605, delta -0.671168
- cost 314: score 19.250607, delta +0.381834

## Examples
- train: 4 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 3x3 -> output 3x3

input:
```text
022
022
200
```

output:
```text
022
022
100
```

### train[2]
input 4x4 -> output 4x4

input:
```text
2220
0200
0002
0200
```

output:
```text
2220
0200
0001
0100
```

### train[3]
input 5x4 -> output 5x4

input:
```text
2200
0200
2202
0000
0222
```

output:
```text
2200
0200
2201
0000
0222
```

### train[4]
input 3x3 -> output 3x3

input:
```text
220
202
020
```

output:
```text
220
201
010
```

### test[1]
input 5x4 -> output 5x4

input:
```text
2202
0200
0020
2000
0022
```

output:
```text
2201
0200
0010
1000
0022
```

### arc-gen[1]
input 4x3 -> output 4x3

input:
```text
200
002
222
002
```

output:
```text
100
002
222
002
```

### arc-gen[2]
input 3x4 -> output 3x4

input:
```text
0222
2200
0002
```

output:
```text
0222
2200
0001
```

### arc-gen[3]
input 4x5 -> output 4x5

input:
```text
20000
00200
00222
00222
```

output:
```text
10000
00200
00222
00222
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 272 --onnx path/to/candidate.onnx
```
