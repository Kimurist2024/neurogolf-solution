# Task 174 Golf Brief

## Current Net
- path: `artifacts/optimized/task174.onnx`
- file size: 3808 bytes
- cost: 32245
- score: 14.618882
- memory: 32163
- params: 82
- nodes: 103
- value_info tensors after shape inference: 102
- local gold-correct: True

## Op Histogram

- Mul: 19
- Cast: 14
- Sub: 13
- Unsqueeze: 12
- ReduceSum: 11
- Equal: 7
- Add: 6
- ReduceMax: 5
- LessOrEqual: 3
- ReduceMin: 2
- Where: 2
- MatMul: 2
- Slice: 1
- Squeeze: 1
- Less: 1
- ArgMax: 1
- Transpose: 1
- Concat: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +3.578724
- cost 314: score 19.250607, delta +4.631725

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 10x10 -> output 2x4

input:
```text
0000000000
0220000000
0022200770
0000007070
0000000000
0000000000
0006666000
0000660000
0000000000
0000000000
```

output:
```text
6666
0660
```

### train[2]
input 10x10 -> output 2x2

input:
```text
0000000000
0044000000
0044008880
0000008088
0000000000
0000000000
0000000000
0022220000
0222000000
0000000000
```

output:
```text
44
44
```

### train[3]
input 10x10 -> output 2x4

input:
```text
0000000000
0330000000
0030050050
0030055550
0000000000
0000000000
0000000000
0008880000
8888088000
0000000000
```

output:
```text
5005
5555
```

### test[1]
input 10x10 -> output 3x4

input:
```text
0000000000
0000030030
0990033330
0990000030
9999000000
0000000000
0000000000
0000444440
0000400440
0000000000
```

output:
```text
0990
0990
9999
```

### arc-gen[1]
input 10x10 -> output 1x2

input:
```text
0000000000
8880000000
8800000000
0000002200
0000000000
0000000000
7770000000
7700000000
0000000000
0000000000
```

output:
```text
22
```

### arc-gen[2]
input 10x10 -> output 3x2

input:
```text
0000000000
0000000000
0006666000
0006066000
5506060000
0500000000
0503300000
0503300000
5503300000
0000000000
```

output:
```text
33
33
33
```

### arc-gen[3]
input 10x10 -> output 1x2

input:
```text
0007700000
0000700000
0000700000
0000700000
0000000000
0000000000
0444400000
0444440000
0000000880
0000000000
```

output:
```text
88
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 174 --onnx path/to/candidate.onnx
```
