# Task 307 Golf Brief

## Current Net
- path: `artifacts/optimized/task307.onnx`
- file size: 387 bytes
- cost: 40
- score: 21.311121
- memory: 0
- params: 40
- nodes: 1
- value_info tensors after shape inference: 0
- local gold-correct: True

## Op Histogram

- ConvTranspose: 1

## Targets

- cost 900: score 18.197605, delta -3.113515
- cost 314: score 19.250607, delta -2.060514

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 3x3 -> output 6x6

input:
```text
051
555
250
```

output:
```text
005511
005511
555555
555555
225500
225500
```

### train[2]
input 2x2 -> output 4x4

input:
```text
21
31
```

output:
```text
2211
2211
3311
3311
```

### train[3]
input 4x4 -> output 8x8

input:
```text
2030
2130
0033
0035
```

output:
```text
22003300
22003300
22113300
22113300
00003333
00003333
00003355
00003355
```

### test[1]
input 5x5 -> output 10x10

input:
```text
20078
21100
05660
35600
05000
```

output:
```text
2200007788
2200007788
2211110000
2211110000
0055666600
0055666600
3355660000
3355660000
0055000000
0055000000
```

### arc-gen[1]
input 2x2 -> output 4x4

input:
```text
02
31
```

output:
```text
0022
0022
3311
3311
```

### arc-gen[2]
input 2x2 -> output 4x4

input:
```text
10
00
```

output:
```text
1100
1100
0000
0000
```

### arc-gen[3]
input 2x2 -> output 4x4

input:
```text
03
00
```

output:
```text
0033
0033
0000
0000
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 307 --onnx path/to/candidate.onnx
```
