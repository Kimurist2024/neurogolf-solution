# Task 006 Golf Brief

## Current Net
- path: `artifacts/optimized/task006.onnx`
- file size: 724 bytes
- cost: 395
- score: 19.021114
- memory: 360
- params: 35
- nodes: 11
- value_info tensors after shape inference: 10
- local gold-correct: True

## Op Histogram

- Cast: 4
- Slice: 2
- And: 1
- Not: 1
- Concat: 1
- Conv: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta -0.823509
- cost 314: score 19.250607, delta +0.229493

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 3x7 -> output 3x3

input:
```text
1005010
0105111
1005000
```

output:
```text
000
020
000
```

### train[2]
input 3x7 -> output 3x3

input:
```text
1105010
0015111
1105010
```

output:
```text
020
002
020
```

### train[3]
input 3x7 -> output 3x3

input:
```text
0015000
1105101
0115101
```

output:
```text
000
200
002
```

### test[1]
input 3x7 -> output 3x3

input:
```text
1015101
0105101
1015010
```

output:
```text
202
000
000
```

### arc-gen[1]
input 3x7 -> output 3x3

input:
```text
0015100
1115011
1005100
```

output:
```text
000
022
200
```

### arc-gen[2]
input 3x7 -> output 3x3

input:
```text
1005110
0005000
1015100
```

output:
```text
200
000
200
```

### arc-gen[3]
input 3x7 -> output 3x3

input:
```text
1115101
1015101
1005110
```

output:
```text
202
202
200
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 6 --onnx path/to/candidate.onnx
```
