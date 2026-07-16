# Task 288 Golf Brief

## Current Net
- path: `artifacts/optimized/task288.onnx`
- file size: 2804 bytes
- cost: 17742
- score: 15.216310
- memory: 17672
- params: 70
- nodes: 28
- value_info tensors after shape inference: 27
- local gold-correct: True

## Op Histogram

- Sub: 4
- Squeeze: 4
- ArgMax: 3
- Gather: 3
- ReduceMax: 2
- Add: 2
- Less: 2
- Equal: 2
- And: 2
- Slice: 1
- Unsqueeze: 1
- Or: 1
- Where: 1

## Targets

- cost 900: score 18.197605, delta +2.981295
- cost 314: score 19.250607, delta +4.034297

## Examples
- train: 4 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 3x3 -> output 3x3

input:
```text
000
020
242
```

output:
```text
404
020
242
```

### train[2]
input 5x5 -> output 5x5

input:
```text
00000
00000
00000
00800
88388
```

output:
```text
00000
30003
03030
00800
88388
```

### train[3]
input 5x5 -> output 5x5

input:
```text
00000
00000
00000
06660
61116
```

output:
```text
00000
00000
10001
06660
61116
```

### train[4]
input 7x7 -> output 7x7

input:
```text
0000000
0000000
0000000
0000000
0000000
0022200
2244422
```

output:
```text
0000000
0000000
0000000
4000004
0400040
0022200
2244422
```

### test[1]
input 9x9 -> output 9x9

input:
```text
000000000
000000000
000000000
000000000
000000000
000000000
000000000
000888000
888222888
```

output:
```text
000000000
000000000
000000000
000000000
200000002
020000020
002000200
000888000
888222888
```

### arc-gen[1]
input 7x7 -> output 7x7

input:
```text
0000000
0000000
0000000
0000000
0000000
0008000
8883888
```

output:
```text
0000000
0000000
3000003
0300030
0030300
0008000
8883888
```

### arc-gen[2]
input 5x5 -> output 5x5

input:
```text
00000
00000
00000
00900
99299
```

output:
```text
00000
20002
02020
00900
99299
```

### arc-gen[3]
input 5x5 -> output 5x5

input:
```text
00000
00000
00000
00200
22122
```

output:
```text
00000
10001
01010
00200
22122
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 288 --onnx path/to/candidate.onnx
```
