# Task 256 Golf Brief

## Current Net
- path: `artifacts/optimized/task256.onnx`
- file size: 1638 bytes
- cost: 15143
- score: 15.374706
- memory: 15084
- params: 59
- nodes: 41
- value_info tensors after shape inference: 40
- local gold-correct: True

## Op Histogram

- Mul: 10
- Sub: 7
- Relu: 5
- ReduceSum: 4
- Min: 4
- Cast: 3
- Slice: 2
- Sum: 2
- ReduceMax: 1
- Max: 1
- Concat: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +2.822899
- cost 314: score 19.250607, delta +3.875901

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 7x7 -> output 7x7

input:
```text
0000000
0000000
0000000
2200000
0000000
0000000
0000000
```

output:
```text
3333300
3333000
3330000
2200000
1000000
0000000
0000000
```

### train[2]
input 8x9 -> output 8x9

input:
```text
000000000
000000000
000000000
222000000
000000000
000000000
000000000
000000000
```

output:
```text
333333000
333330000
333300000
222000000
110000000
100000000
000000000
000000000
```

### train[3]
input 7x9 -> output 7x9

input:
```text
000000000
000000000
222200000
000000000
000000000
000000000
000000000
```

output:
```text
333333000
333330000
222200000
111000000
110000000
100000000
000000000
```

### test[1]
input 9x9 -> output 9x9

input:
```text
000000000
000000000
222220000
000000000
000000000
000000000
000000000
000000000
000000000
```

output:
```text
333333300
333333000
222220000
111100000
111000000
110000000
100000000
000000000
000000000
```

### arc-gen[1]
input 11x8 -> output 11x8

input:
```text
00000000
00000000
00000000
00000000
22200000
00000000
00000000
00000000
00000000
00000000
00000000
```

output:
```text
33333330
33333300
33333000
33330000
22200000
11000000
10000000
00000000
00000000
00000000
00000000
```

### arc-gen[2]
input 4x6 -> output 4x6

input:
```text
000000
220000
000000
000000
```

output:
```text
333000
220000
100000
000000
```

### arc-gen[3]
input 4x7 -> output 4x7

input:
```text
0000000
2200000
0000000
0000000
```

output:
```text
3330000
2200000
1000000
0000000
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 256 --onnx path/to/candidate.onnx
```
