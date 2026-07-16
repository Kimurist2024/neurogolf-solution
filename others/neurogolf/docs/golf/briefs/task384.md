# Task 384 Golf Brief

## Current Net
- path: `artifacts/optimized/task384.onnx`
- file size: 2705 bytes
- cost: 36016
- score: 14.508281
- memory: 35934
- params: 82
- nodes: 42
- value_info tensors after shape inference: 41
- local gold-correct: True

## Research Queue
- priority rank: 43
- recorded cost: 74662
- recorded memory: 74556
- recorded params: 106
- recorded nodes: 41

## Op Histogram

- Cast: 6
- Sub: 5
- Reshape: 4
- ArgMax: 4
- Gather: 4
- Add: 4
- Mul: 4
- Less: 4
- Slice: 2
- ReduceMax: 2
- Where: 2
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +3.689324
- cost 314: score 19.250607, delta +4.742326

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 9x9 -> output 6x8

input:
```text
000000000
000000000
004400000
044440000
004400000
000000000
000000000
000000000
000000000
```

output:
```text
00444400
00444400
44444444
44444444
00444400
00444400
```

### train[2]
input 9x9 -> output 6x6

input:
```text
000000000
000040000
000440000
000004000
000000000
000000000
000000000
000000000
000000000
```

output:
```text
004400
004400
444400
444400
000044
000044
```

### train[3]
input 9x9 -> output 8x6

input:
```text
000000000
000000000
000000000
000000000
004000000
044000000
004000000
004400000
000000000
```

output:
```text
004400
004400
444400
444400
004400
004400
004444
004444
```

### test[1]
input 9x9 -> output 6x10

input:
```text
000000000
000040400
000404040
000040400
000000000
000000000
000000000
000000000
000000000
```

output:
```text
0044004400
0044004400
4400440044
4400440044
0044004400
0044004400
```

### arc-gen[1]
input 9x9 -> output 6x10

input:
```text
000000000
044440000
044044000
040040000
000000000
000000000
000000000
000000000
000000000
```

output:
```text
4444444400
4444444400
4444004444
4444004444
4400004400
4400004400
```

### arc-gen[2]
input 9x9 -> output 8x6

input:
```text
000000000
000000000
004000000
044000000
044400000
004000000
000000000
000000000
000000000
```

output:
```text
004400
004400
444400
444400
444444
444444
004400
004400
```

### arc-gen[3]
input 9x9 -> output 6x4

input:
```text
000000000
004000000
040000000
004000000
000000000
000000000
000000000
000000000
000000000
```

output:
```text
0044
0044
4400
4400
0044
0044
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 384 --onnx path/to/candidate.onnx
```
