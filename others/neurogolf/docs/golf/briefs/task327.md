# Task 327 Golf Brief

## Current Net
- path: `artifacts/optimized/task327.onnx`
- file size: 2321 bytes
- cost: 7538
- score: 16.072288
- memory: 7516
- params: 22
- nodes: 18
- value_info tensors after shape inference: 17
- local gold-correct: True

## Op Histogram

- Pad: 8
- Slice: 3
- Sum: 3
- Mul: 2
- Cast: 1
- ReduceMax: 1

## Targets

- cost 900: score 18.197605, delta +2.125317
- cost 314: score 19.250607, delta +3.178319

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 258 remaining

### train[1]
input 3x3 -> output 6x6

input:
```text
610
300
000
```

output:
```text
610000
361000
036100
003610
000361
000036
```

### train[2]
input 3x3 -> output 6x6

input:
```text
040
080
200
```

output:
```text
040000
084000
208400
020840
002084
000208
```

### train[3]
input 3x3 -> output 6x6

input:
```text
006
130
000
```

output:
```text
006000
130600
013060
001306
000130
000013
```

### test[1]
input 3x3 -> output 6x6

input:
```text
003
000
049
```

output:
```text
003000
000300
049030
004903
000490
000049
```

### arc-gen[1]
input 3x3 -> output 6x6

input:
```text
070
400
002
```

output:
```text
070000
407000
042700
004270
000427
000042
```

### arc-gen[2]
input 3x3 -> output 6x6

input:
```text
080
040
100
```

output:
```text
080000
048000
104800
010480
001048
000104
```

### arc-gen[3]
input 3x3 -> output 6x6

input:
```text
530
000
060
```

output:
```text
530000
053000
065300
006530
000653
000065
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 327 --onnx path/to/candidate.onnx
```
