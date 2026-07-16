# Task 295 Golf Brief

## Current Net
- path: `artifacts/optimized/task295.onnx`
- file size: 1074 bytes
- cost: 12172
- score: 15.593106
- memory: 12107
- params: 65
- nodes: 21
- value_info tensors after shape inference: 20
- local gold-correct: True

## Op Histogram

- ReduceSum: 3
- Mul: 3
- Where: 3
- Less: 3
- Cast: 2
- Sum: 2
- Slice: 1
- Conv: 1
- Greater: 1
- Sub: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +2.604499
- cost 314: score 19.250607, delta +3.657501

## Examples
- train: 5 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 1x6 -> output 3x6

input:
```text
110000
```

output:
```text
110000
111000
111100
```

### train[2]
input 1x8 -> output 4x8

input:
```text
20000000
```

output:
```text
20000000
22000000
22200000
22220000
```

### train[3]
input 1x10 -> output 5x10

input:
```text
5550000000
```

output:
```text
5550000000
5555000000
5555500000
5555550000
5555555000
```

### train[4]
input 1x6 -> output 3x6

input:
```text
888800
```

output:
```text
888800
888880
888888
```

### train[5]
input 1x6 -> output 3x6

input:
```text
700000
```

output:
```text
700000
770000
777000
```

### test[1]
input 1x12 -> output 6x12

input:
```text
111000000000
```

output:
```text
111000000000
111100000000
111110000000
111111000000
111111100000
111111110000
```

### arc-gen[1]
input 1x14 -> output 7x14

input:
```text
88000000000000
```

output:
```text
88000000000000
88800000000000
88880000000000
88888000000000
88888800000000
88888880000000
88888888000000
```

### arc-gen[2]
input 1x6 -> output 3x6

input:
```text
999000
```

output:
```text
999000
999900
999990
```

### arc-gen[3]
input 1x6 -> output 3x6

input:
```text
222200
```

output:
```text
222200
222220
222222
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 295 --onnx path/to/candidate.onnx
```
