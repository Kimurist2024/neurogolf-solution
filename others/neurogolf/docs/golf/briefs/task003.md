# Task 003 Golf Brief

## Current Net
- path: `artifacts/optimized/task003.onnx`
- file size: 1424 bytes
- cost: 1114
- score: 17.984288
- memory: 1070
- params: 44
- nodes: 37
- value_info tensors after shape inference: 36
- local gold-correct: True

## Op Histogram

- Mul: 9
- Slice: 5
- Sub: 5
- Concat: 4
- Cast: 3
- Sum: 3
- ReduceSum: 2
- Less: 2
- Not: 2
- Where: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +0.213318
- cost 314: score 19.250607, delta +1.266319

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 258 remaining

### train[1]
input 6x3 -> output 9x3

input:
```text
010
110
010
011
010
110
```

output:
```text
020
220
020
022
020
220
020
022
020
```

### train[2]
input 6x3 -> output 9x3

input:
```text
010
101
010
101
010
101
```

output:
```text
020
202
020
202
020
202
020
202
020
```

### train[3]
input 6x3 -> output 9x3

input:
```text
010
110
010
010
110
010
```

output:
```text
020
220
020
020
220
020
020
220
020
```

### test[1]
input 6x3 -> output 9x3

input:
```text
111
010
010
111
010
010
```

output:
```text
222
020
020
222
020
020
222
020
020
```

### arc-gen[1]
input 6x3 -> output 9x3

input:
```text
110
010
011
010
110
010
```

output:
```text
220
020
022
020
220
020
022
020
220
```

### arc-gen[2]
input 6x3 -> output 9x3

input:
```text
100
011
100
011
100
011
```

output:
```text
200
022
200
022
200
022
200
022
200
```

### arc-gen[3]
input 6x3 -> output 9x3

input:
```text
101
001
101
100
101
001
```

output:
```text
202
002
202
200
202
002
202
200
202
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 3 --onnx path/to/candidate.onnx
```
