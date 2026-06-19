# Task 082 Golf Brief

## Current Net
- path: `artifacts/optimized/task082.onnx`
- file size: 1638 bytes
- cost: 9716
- score: 15.818471
- memory: 9660
- params: 56
- nodes: 10
- value_info tensors after shape inference: 9
- local gold-correct: True

## Op Histogram

- Slice: 2
- Concat: 2
- ReduceSum: 2
- Conv: 1
- Tile: 1
- Sub: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +2.379135
- cost 314: score 19.250607, delta +3.432136

## Examples
- train: 2 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 6x10 -> output 6x10

input:
```text
0200080000
0000000000
0000000000
0000000000
0000000000
0000000000
```

output:
```text
0200080000
2020808000
0200080000
2020808000
0200080000
2020808000
```

### train[2]
input 6x7 -> output 6x7

input:
```text
0400000
0000000
0000000
0000000
0000000
0000000
```

output:
```text
0400000
4040000
0400000
4040000
0400000
4040000
```

### test[1]
input 6x12 -> output 6x12

input:
```text
003000600700
000000000000
000000000000
000000000000
000000000000
000000000000
```

output:
```text
003000600700
030306067070
003000600700
030306067070
003000600700
030306067070
```

### arc-gen[1]
input 6x13 -> output 6x13

input:
```text
0800900600040
0000000000000
0000000000000
0000000000000
0000000000000
0000000000000
```

output:
```text
0800900600040
8089096060404
0800900600040
8089096060404
0800900600040
8089096060404
```

### arc-gen[2]
input 6x6 -> output 6x6

input:
```text
009000
000000
000000
000000
000000
000000
```

output:
```text
009000
090900
009000
090900
009000
090900
```

### arc-gen[3]
input 6x6 -> output 6x6

input:
```text
002000
000000
000000
000000
000000
000000
```

output:
```text
002000
020200
002000
020200
002000
020200
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 82 --onnx path/to/candidate.onnx
```
