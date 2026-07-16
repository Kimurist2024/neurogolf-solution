# Task 248 Golf Brief

## Current Net
- path: `artifacts/optimized/task248.onnx`
- file size: 7827 bytes
- cost: 6442
- score: 16.229406
- memory: 4630
- params: 1812
- nodes: 10
- value_info tensors after shape inference: 9
- local gold-correct: True

## Op Histogram

- ReduceSum: 3
- Cast: 2
- Greater: 1
- Reshape: 1
- Sub: 1
- Gather: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +1.968200
- cost 314: score 19.250607, delta +3.021201

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 6 remaining

### train[1]
input 10x2 -> output 10x2

input:
```text
00
00
00
00
00
00
00
00
00
10
```

output:
```text
01
10
01
10
01
10
01
10
01
10
```

### train[2]
input 10x3 -> output 10x3

input:
```text
000
000
000
000
000
000
000
000
000
100
```

output:
```text
010
100
010
001
010
100
010
001
010
100
```

### train[3]
input 10x4 -> output 10x4

input:
```text
0000
0000
0000
0000
0000
0000
0000
0000
0000
1000
```

output:
```text
0001
0010
0100
1000
0100
0010
0001
0010
0100
1000
```

### test[1]
input 10x5 -> output 10x5

input:
```text
00000
00000
00000
00000
00000
00000
00000
00000
00000
10000
```

output:
```text
01000
10000
01000
00100
00010
00001
00010
00100
01000
10000
```

### arc-gen[1]
input 10x10 -> output 10x10

input:
```text
0000000000
0000000000
0000000000
0000000000
0000000000
0000000000
0000000000
0000000000
0000000000
1000000000
```

output:
```text
0000000001
0000000010
0000000100
0000001000
0000010000
0000100000
0001000000
0010000000
0100000000
1000000000
```

### arc-gen[2]
input 10x3 -> output 10x3

input:
```text
000
000
000
000
000
000
000
000
000
100
```

output:
```text
010
100
010
001
010
100
010
001
010
100
```

### arc-gen[3]
input 10x7 -> output 10x7

input:
```text
0000000
0000000
0000000
0000000
0000000
0000000
0000000
0000000
0000000
1000000
```

output:
```text
0001000
0000100
0000010
0000001
0000010
0000100
0001000
0010000
0100000
1000000
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 248 --onnx path/to/candidate.onnx
```
