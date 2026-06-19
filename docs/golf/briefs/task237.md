# Task 237 Golf Brief

## Current Net
- path: `artifacts/optimized/task237.onnx`
- file size: 1637 bytes
- cost: 14324
- score: 15.430308
- memory: 14283
- params: 41
- nodes: 25
- value_info tensors after shape inference: 24
- local gold-correct: True

## Op Histogram

- Mul: 4
- ReduceMax: 3
- Slice: 2
- Cast: 2
- Sub: 2
- MaxPool: 2
- ReduceSum: 1
- Sum: 1
- Conv: 1
- Equal: 1
- Greater: 1
- And: 1
- Where: 1
- Max: 1
- Concat: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +2.767297
- cost 314: score 19.250607, delta +3.820299

## Examples
- train: 4 shown
- test: 1 shown
- arc-gen: 3 shown, 258 remaining

### train[1]
input 6x6 -> output 6x6

input:
```text
000000
002000
000000
030000
000000
000000
```

output:
```text
000000
002222
000002
033333
000003
000003
```

### train[2]
input 3x3 -> output 3x3

input:
```text
000
060
000
```

output:
```text
000
066
006
```

### train[3]
input 6x6 -> output 6x6

input:
```text
000000
080000
000000
000000
000500
000000
```

output:
```text
000000
088888
000008
000008
000555
000005
```

### train[4]
input 7x5 -> output 7x5

input:
```text
00000
00800
00000
07000
00000
00600
00000
```

output:
```text
00000
00888
00008
07777
00007
00666
00006
```

### test[1]
input 7x8 -> output 7x8

input:
```text
00080000
00000000
00700000
00000000
00000200
00000000
00000000
```

output:
```text
00088888
00000008
00777777
00000007
00000222
00000002
00000002
```

### arc-gen[1]
input 3x7 -> output 3x7

input:
```text
0000000
0600000
0000000
```

output:
```text
0000000
0666666
0000006
```

### arc-gen[2]
input 5x3 -> output 5x3

input:
```text
400
000
000
010
000
```

output:
```text
444
004
004
011
001
```

### arc-gen[3]
input 6x3 -> output 6x3

input:
```text
700
000
000
030
000
000
```

output:
```text
777
007
007
033
003
003
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 237 --onnx path/to/candidate.onnx
```
