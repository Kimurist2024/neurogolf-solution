# Task 329 Golf Brief

## Current Net
- path: `artifacts/optimized/task329.onnx`
- file size: 1652 bytes
- cost: 10260
- score: 15.763992
- memory: 10214
- params: 46
- nodes: 21
- value_info tensors after shape inference: 20
- local gold-correct: True

## Op Histogram

- Sub: 4
- Mul: 3
- ReduceSum: 2
- Relu: 2
- Sum: 2
- Slice: 1
- Cast: 1
- ReduceMax: 1
- Div: 1
- Floor: 1
- Neg: 1
- Min: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +2.433613
- cost 314: score 19.250607, delta +3.486615

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 3x3 -> output 3x3

input:
```text
640
039
100
```

output:
```text
040
030
000
```

### train[2]
input 5x5 -> output 5x5

input:
```text
80300
86560
36300
00059
50900
```

output:
```text
00300
00500
00300
00000
00900
```

### train[3]
input 5x5 -> output 5x5

input:
```text
30400
30470
06007
00800
08022
```

output:
```text
00400
00400
00000
00800
00000
```

### test[1]
input 7x7 -> output 7x7

input:
```text
0030007
8108000
0030803
0701070
0000000
1086000
0806010
```

output:
```text
0000000
0008000
0000000
0001000
0000000
0006000
0006000
```

### arc-gen[1]
input 3x3 -> output 3x3

input:
```text
096
020
014
```

output:
```text
090
020
010
```

### arc-gen[2]
input 3x3 -> output 3x3

input:
```text
040
001
208
```

output:
```text
040
000
000
```

### arc-gen[3]
input 3x3 -> output 3x3

input:
```text
010
076
506
```

output:
```text
010
070
000
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 329 --onnx path/to/candidate.onnx
```
