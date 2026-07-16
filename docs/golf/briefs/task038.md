# Task 038 Golf Brief

## Current Net
- path: `artifacts/optimized/task038.onnx`
- file size: 1065 bytes
- cost: 2158
- score: 17.323063
- memory: 2116
- params: 42
- nodes: 16
- value_info tensors after shape inference: 15
- local gold-correct: True

## Op Histogram

- Slice: 4
- Mul: 3
- Sub: 2
- Relu: 2
- ReduceSum: 1
- Resize: 1
- Sum: 1
- Concat: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +0.874542
- cost 314: score 19.250607, delta +1.927544

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 9x9 -> output 1x5

input:
```text
000022001
011022000
011000022
000000022
102200000
002201100
000001100
000000000
010000001
```

output:
```text
11000
```

### train[2]
input 9x9 -> output 1x5

input:
```text
110200002
110001100
000201100
000000001
011022000
011022002
000000000
000220110
010220110
```

output:
```text
11110
```

### train[3]
input 9x9 -> output 1x5

input:
```text
220110000
220110011
100000011
022000000
022011010
000011000
000020000
011000022
011001022
```

output:
```text
11110
```

### test[1]
input 9x9 -> output 1x5

input:
```text
000002201
110102200
110000000
000001100
022001100
022000000
100000220
220110220
220110000
```

output:
```text
11100
```

### arc-gen[1]
input 9x9 -> output 1x5

input:
```text
220101100
220201100
000000000
200011211
000011011
000020002
220100000
220022011
001022011
```

output:
```text
11110
```

### arc-gen[2]
input 9x9 -> output 1x5

input:
```text
000020022
011011022
011211010
200000000
110200020
110002100
000100000
100022020
020022100
```

output:
```text
11100
```

### arc-gen[3]
input 9x9 -> output 1x5

input:
```text
010000022
002210022
002200100
100000022
000101022
000000000
002200000
002200011
200000011
```

output:
```text
10000
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 38 --onnx path/to/candidate.onnx
```
