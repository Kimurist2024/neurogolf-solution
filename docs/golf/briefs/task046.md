# Task 046 Golf Brief

## Current Net
- path: `artifacts/optimized/task046.onnx`
- file size: 4609 bytes
- cost: 31217
- score: 14.651282
- memory: 30304
- params: 913
- nodes: 70
- value_info tensors after shape inference: 69
- local gold-correct: True

## Op Histogram

- Slice: 9
- Sub: 9
- Mul: 8
- Pad: 7
- Abs: 6
- Less: 6
- Where: 6
- MatMul: 4
- Sum: 3
- Gather: 2
- ReduceMax: 2
- Transpose: 2
- Cast: 1
- Conv: 1
- ReduceMin: 1
- ReduceSum: 1
- MaxPool: 1
- Concat: 1

## Targets

- cost 900: score 18.197605, delta +3.546323
- cost 314: score 19.250607, delta +4.599325

## Examples
- train: 4 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 3x9 -> output 3x7

input:
```text
050000000
220510522
000050000
```

output:
```text
0211000
2201222
0000000
```

### train[2]
input 3x11 -> output 3x9

input:
```text
00051500000
22000000333
05000005300
```

output:
```text
000000000
220000333
021113300
```

### train[3]
input 3x11 -> output 3x9

input:
```text
00000050000
22205880000
00500000566
```

output:
```text
000000000
222008666
002888000
```

### train[4]
input 3x11 -> output 3x8

input:
```text
01500000220
11005205200
00000500000
```

output:
```text
01122022
11002220
00000000
```

### test[1]
input 3x11 -> output 3x8

input:
```text
05051005058
22001053008
00005000000
```

output:
```text
02110000
22010388
00013308
```

### arc-gen[1]
input 3x12 -> output 3x9

input:
```text
111500000500
100005750800
000000000880
```

output:
```text
111177780
100000080
000000088
```

### arc-gen[2]
input 3x14 -> output 3x11

input:
```text
00000000050000
22505445088880
00000000000000
```

output:
```text
00000000000
22244448000
00000008888
```

### arc-gen[3]
input 3x10 -> output 3x7

input:
```text
0000850000
7750500500
0000000330
```

output:
```text
0008830
7778033
0000000
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 46 --onnx path/to/candidate.onnx
```
