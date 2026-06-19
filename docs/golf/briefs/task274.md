# Task 274 Golf Brief

## Current Net
- path: `artifacts/optimized/task274.onnx`
- file size: 1332 bytes
- cost: 8077
- score: 16.003224
- memory: 7964
- params: 113
- nodes: 16
- value_info tensors after shape inference: 15
- local gold-correct: True

## Op Histogram

- Slice: 2
- ReduceSum: 2
- Squeeze: 2
- Greater: 2
- Cast: 2
- ArgMax: 2
- Sub: 2
- Gather: 1
- Conv: 1

## Targets

- cost 900: score 18.197605, delta +2.194381
- cost 314: score 19.250607, delta +3.247383

## Examples
- train: 6 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 6x6 -> output 3x3

input:
```text
000000
050050
050050
050050
058850
055550
```

output:
```text
888
000
000
```

### train[2]
input 9x9 -> output 3x3

input:
```text
000000000
005000500
005000500
005000500
005000500
005888500
005888500
005888500
005555500
```

output:
```text
888
008
000
```

### train[3]
input 9x9 -> output 3x3

input:
```text
000000000
000000000
050000050
050000050
050000050
058888850
058888850
058888850
055555550
```

output:
```text
888
000
000
```

### train[4]
input 9x9 -> output 3x3

input:
```text
000000000
000000000
005000500
005000500
005888500
005888500
005888500
005888500
005555500
```

output:
```text
880
000
000
```

### train[5]
input 5x6 -> output 3x3

input:
```text
000000
050050
058850
058850
055550
```

output:
```text
800
000
000
```

### train[6]
input 7x7 -> output 3x3

input:
```text
0000000
0000000
0500050
0500050
0588850
0588850
0555550
```

output:
```text
880
000
000
```

### test[1]
input 9x9 -> output 3x3

input:
```text
000000000
005000500
005888500
005888500
005888500
005888500
005888500
005888500
005555500
```

output:
```text
800
000
000
```

### arc-gen[1]
input 7x10 -> output 3x3

input:
```text
0000000000
0500000050
0500000050
0500000050
0500000050
0588888850
0555555550
```

output:
```text
888
008
000
```

### arc-gen[2]
input 6x8 -> output 3x3

input:
```text
00000000
00500500
00588500
00588500
00588500
00555500
```

output:
```text
800
000
000
```

### arc-gen[3]
input 7x8 -> output 3x3

input:
```text
00000000
00500500
00588500
00588500
00588500
00588500
00555500
```

output:
```text
800
000
000
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 274 --onnx path/to/candidate.onnx
```
