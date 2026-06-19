# Task 301 Golf Brief

## Current Net
- path: `artifacts/optimized/task301.onnx`
- file size: 3736 bytes
- cost: 14991
- score: 15.384795
- memory: 14915
- params: 76
- nodes: 50
- value_info tensors after shape inference: 50
- local gold-correct: True

## Op Histogram

- Cast: 10
- ReduceSum: 4
- Greater: 4
- Less: 4
- ReduceMax: 3
- Unsqueeze: 3
- And: 3
- Not: 2
- Mul: 2
- Add: 2
- Squeeze: 2
- Gather: 2
- Where: 2
- Slice: 1
- Conv: 1
- Neg: 1
- TopK: 1
- Sub: 1
- OneHot: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +2.812811
- cost 314: score 19.250607, delta +3.865812

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 10x7 -> output 10x7

input:
```text
0220000
0000300
1110000
0000000
0555555
0000000
6666600
0004444
0000000
8888888
```

output:
```text
0000000
0000000
0000000
0000003
0000022
0000111
0004444
0066666
0555555
8888888
```

### train[2]
input 7x4 -> output 7x4

input:
```text
0001
0000
2220
0000
0330
0000
8888
```

output:
```text
0000
0000
0000
0001
0033
0222
8888
```

### train[3]
input 3x3 -> output 3x3

input:
```text
220
040
888
```

output:
```text
004
022
888
```

### test[1]
input 11x8 -> output 11x8

input:
```text
66600000
00001111
00000000
04444400
00000770
00000000
22222220
00333333
09000000
00000000
88888888
```

output:
```text
00000000
00000000
00000000
00000009
00000077
00000666
00001111
00044444
00333333
02222222
88888888
```

### arc-gen[1]
input 10x7 -> output 10x7

input:
```text
0111110
0660000
0000000
2000000
5555550
0000000
9999000
0000000
0004440
8888888
```

output:
```text
0000000
0000000
0000000
0000002
0000066
0000444
0009999
0011111
0555555
8888888
```

### arc-gen[2]
input 3x3 -> output 3x3

input:
```text
110
009
888
```

output:
```text
009
011
888
```

### arc-gen[3]
input 7x5 -> output 7x5

input:
```text
00111
00660
00300
99990
00000
00000
88888
```

output:
```text
00000
00000
00003
00066
00111
09999
88888
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 301 --onnx path/to/candidate.onnx
```
