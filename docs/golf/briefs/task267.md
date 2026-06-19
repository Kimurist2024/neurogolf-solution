# Task 267 Golf Brief

## Current Net
- path: `artifacts/optimized/task267.onnx`
- file size: 1202 bytes
- cost: 1207
- score: 17.904107
- memory: 1160
- params: 47
- nodes: 11
- value_info tensors after shape inference: 10
- local gold-correct: True

## Op Histogram

- Mul: 3
- Slice: 2
- Cast: 2
- Sub: 1
- ReduceSum: 1
- Equal: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +0.293498
- cost 314: score 19.250607, delta +1.346500

## Examples
- train: 2 shown
- test: 1 shown
- arc-gen: 3 shown, 258 remaining

### train[1]
input 7x7 -> output 7x7

input:
```text
0000000
0222000
0020000
0222200
0022200
0002000
4000000
```

output:
```text
0000000
0444000
0040000
0444400
0044400
0004000
0000000
```

### train[2]
input 7x7 -> output 7x7

input:
```text
0000000
0003000
0033300
0333300
0330000
0033000
6000000
```

output:
```text
0000000
0006000
0066600
0666600
0660000
0066000
0000000
```

### test[1]
input 7x7 -> output 7x7

input:
```text
0000000
0888000
0888880
0008800
0088000
0088800
2000000
```

output:
```text
0000000
0222000
0222220
0002200
0022000
0022200
0000000
```

### arc-gen[1]
input 7x7 -> output 7x7

input:
```text
0000000
0111100
0111110
0011000
0001000
0000000
2000000
```

output:
```text
0000000
0222200
0222220
0022000
0002000
0000000
0000000
```

### arc-gen[2]
input 7x7 -> output 7x7

input:
```text
0000000
0222220
0220220
0200220
0000000
0000000
9000000
```

output:
```text
0000000
0999990
0990990
0900990
0000000
0000000
0000000
```

### arc-gen[3]
input 7x7 -> output 7x7

input:
```text
0000000
0333330
0303300
0003330
0000300
0000000
5000000
```

output:
```text
0000000
0555550
0505500
0005550
0000500
0000000
0000000
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 267 --onnx path/to/candidate.onnx
```
