# Task 114 Golf Brief

## Current Net
- path: `artifacts/optimized/task114.onnx`
- file size: 4671 bytes
- cost: 1249
- score: 17.869901
- memory: 1091
- params: 158
- nodes: 26
- value_info tensors after shape inference: 25
- local gold-correct: True

## Op Histogram

- Cast: 5
- Reshape: 4
- Slice: 3
- Equal: 3
- Where: 3
- Not: 2
- ReduceMax: 2
- ArgMax: 1
- Concat: 1
- GatherElements: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +0.327704
- cost 314: score 19.250607, delta +1.380706

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 2x2 -> output 4x4

input:
```text
12
38
```

output:
```text
0120
1122
3388
0380
```

### train[2]
input 2x3 -> output 4x5

input:
```text
184
838
```

output:
```text
01840
11844
88388
08380
```

### train[3]
input 3x3 -> output 5x5

input:
```text
214
802
328
```

output:
```text
02140
22144
88022
33288
03280
```

### test[1]
input 3x2 -> output 5x4

input:
```text
28
14
34
```

output:
```text
0280
2288
1144
3344
0340
```

### arc-gen[1]
input 3x2 -> output 5x4

input:
```text
28
13
48
```

output:
```text
0280
2288
1133
4488
0480
```

### arc-gen[2]
input 3x2 -> output 5x4

input:
```text
88
12
88
```

output:
```text
0880
8888
1122
8888
0880
```

### arc-gen[3]
input 3x2 -> output 5x4

input:
```text
11
34
24
```

output:
```text
0110
1111
3344
2244
0240
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 114 --onnx path/to/candidate.onnx
```
