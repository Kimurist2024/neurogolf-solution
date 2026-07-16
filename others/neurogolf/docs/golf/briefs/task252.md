# Task 252 Golf Brief

## Current Net
- path: `artifacts/optimized/task252.onnx`
- file size: 1812 bytes
- cost: 29062
- score: 14.722813
- memory: 28800
- params: 262
- nodes: 28
- value_info tensors after shape inference: 27
- local gold-correct: True

## Op Histogram

- Slice: 11
- Mul: 10
- Sub: 2
- Cast: 1
- ReduceSum: 1
- Sum: 1
- Concat: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +3.474792
- cost 314: score 19.250607, delta +4.527794

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 3x3 -> output 3x3

input:
```text
200
020
002
```

output:
```text
200
040
002
```

### train[2]
input 8x8 -> output 8x8

input:
```text
00900000
00090000
00009000
00000900
90000090
09000009
00900000
00090000
```

output:
```text
00900000
00040000
00009000
00000400
90000090
04000004
00900000
00040000
```

### train[3]
input 6x6 -> output 6x6

input:
```text
003000
000300
000030
300003
030000
003000
```

output:
```text
003000
000400
000030
300004
040000
003000
```

### test[1]
input 12x12 -> output 12x12

input:
```text
000060000000
600006000000
060000600000
006000060000
000600006000
000060000600
000006000060
000000600006
600000060000
060000006000
006000000600
000600000060
```

output:
```text
000060000000
600004000000
040000600000
006000040000
000400006000
000060000400
000004000060
000000600004
600000040000
040000006000
006000000400
000400000060
```

### arc-gen[1]
input 11x11 -> output 11x11

input:
```text
00000000000
00000000000
00000000000
00000000000
00000000000
00000000000
90000000000
09000000000
00900000000
00090000000
00009000000
```

output:
```text
00000000000
00000000000
00000000000
00000000000
00000000000
00000000000
90000000000
04000000000
00900000000
00040000000
00009000000
```

### arc-gen[2]
input 4x4 -> output 4x4

input:
```text
5000
0500
0050
0005
```

output:
```text
5000
0400
0050
0004
```

### arc-gen[3]
input 4x4 -> output 4x4

input:
```text
1000
0100
0010
0001
```

output:
```text
1000
0400
0010
0004
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 252 --onnx path/to/candidate.onnx
```
