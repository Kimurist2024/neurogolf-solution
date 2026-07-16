# Task 357 Golf Brief

## Current Net
- path: `artifacts/optimized/task357.onnx`
- file size: 1767 bytes
- cost: 6572
- score: 16.209427
- memory: 6532
- params: 40
- nodes: 24
- value_info tensors after shape inference: 23
- local gold-correct: True

## Op Histogram

- Sub: 5
- And: 4
- ReduceSum: 2
- Greater: 2
- Mul: 2
- Slice: 1
- ReduceMax: 1
- Div: 1
- Floor: 1
- Abs: 1
- Less: 1
- Not: 1
- Concat: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +1.988179
- cost 314: score 19.250607, delta +3.041180

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
81
18
81
18
81
18
81
18
81
18
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
818
188
818
881
818
188
818
881
818
188
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
8881
8818
8188
1888
8188
8818
8881
8818
8188
1888
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
81888
18888
81888
88188
88818
88881
88818
88188
81888
18888
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
8888888881
8888888818
8888888188
8888881888
8888818888
8888188888
8881888888
8818888888
8188888888
1888888888
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
818
188
818
881
818
188
818
881
818
188
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
8881888
8888188
8888818
8888881
8888818
8888188
8881888
8818888
8188888
1888888
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 357 --onnx path/to/candidate.onnx
```
