# Task 348 Golf Brief

## Current Net
- path: `artifacts/optimized/task348.onnx`
- file size: 1950 bytes
- cost: 7207
- score: 16.117192
- memory: 7068
- params: 139
- nodes: 28
- value_info tensors after shape inference: 27
- local gold-correct: True

## Op Histogram

- And: 4
- ReduceMax: 3
- Mul: 3
- Sub: 3
- Slice: 2
- ReduceSum: 2
- Not: 2
- Greater: 1
- Abs: 1
- GreaterOrEqual: 1
- Div: 1
- Floor: 1
- Less: 1
- Or: 1
- Concat: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +2.080413
- cost 314: score 19.250607, delta +3.133415

## Examples
- train: 2 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 5x7 -> output 5x7

input:
```text
0007000
0007000
0007000
0007000
0000000
```

output:
```text
8787878
0787870
0087800
0007000
0000000
```

### train[2]
input 7x8 -> output 7x8

input:
```text
00700000
00700000
00700000
00700000
00700000
00000000
00000000
```

output:
```text
78787870
78787800
78787000
08780000
00700000
00000000
00000000
```

### test[1]
input 9x9 -> output 9x9

input:
```text
000007000
000007000
000007000
000007000
000007000
000007000
000007000
000000000
000000000
```

output:
```text
878787878
878787878
078787878
008787878
000787870
000087800
000007000
000000000
000000000
```

### arc-gen[1]
input 5x9 -> output 5x9

input:
```text
000007000
000007000
000007000
000000000
000000000
```

output:
```text
000787870
000087800
000007000
000000000
000000000
```

### arc-gen[2]
input 7x5 -> output 7x5

input:
```text
00700
00700
00700
00700
00000
00000
00000
```

output:
```text
78787
78787
08780
00700
00000
00000
00000
```

### arc-gen[3]
input 8x5 -> output 8x5

input:
```text
00700
00700
00700
00000
00000
00000
00000
00000
```

output:
```text
78787
08780
00700
00000
00000
00000
00000
00000
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 348 --onnx path/to/candidate.onnx
```
