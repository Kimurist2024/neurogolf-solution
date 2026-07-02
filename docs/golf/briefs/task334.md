# Task 334 Golf Brief

## Current Net
- path: `artifacts/optimized/task334.onnx`
- file size: 889 bytes
- cost: 609
- score: 18.588182
- memory: 572
- params: 37
- nodes: 9
- value_info tensors after shape inference: 8
- local gold-correct: True

## Op Histogram

- Sub: 2
- Slice: 1
- ReduceSum: 1
- ArgMax: 1
- Gather: 1
- Unsqueeze: 1
- Concat: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta -0.390576
- cost 314: score 19.250607, delta +0.662425

## Examples
- train: 7 shown
- test: 2 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 5x5 -> output 3x3

input:
```text
20000
02002
20020
00022
00220
```

output:
```text
555
050
050
```

### train[2]
input 5x5 -> output 3x3

input:
```text
00000
00111
01011
01010
00001
```

output:
```text
050
555
050
```

### train[3]
input 5x5 -> output 3x3

input:
```text
30000
00033
03300
03030
30330
```

output:
```text
005
005
555
```

### train[4]
input 5x5 -> output 3x3

input:
```text
10100
10011
11010
01010
10001
```

output:
```text
050
555
050
```

### train[5]
input 5x5 -> output 3x3

input:
```text
20202
20002
22000
20022
22202
```

output:
```text
555
050
050
```

### train[6]
input 5x5 -> output 3x3

input:
```text
02020
02220
02202
22200
00202
```

output:
```text
555
050
050
```

### train[7]
input 5x5 -> output 3x3

input:
```text
03030
33000
03000
00300
33300
```

output:
```text
005
005
555
```

### test[1]
input 5x5 -> output 3x3

input:
```text
11110
00101
01000
01001
00100
```

output:
```text
050
555
050
```

### test[2]
input 5x5 -> output 3x3

input:
```text
03033
00300
30000
00303
00003
```

output:
```text
005
005
555
```

### arc-gen[1]
input 5x5 -> output 3x3

input:
```text
20200
20200
02200
22020
20000
```

output:
```text
555
050
050
```

### arc-gen[2]
input 5x5 -> output 3x3

input:
```text
00010
00100
10011
11100
10010
```

output:
```text
050
555
050
```

### arc-gen[3]
input 5x5 -> output 3x3

input:
```text
30300
30000
33303
00033
30000
```

output:
```text
005
005
555
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 334 --onnx path/to/candidate.onnx
```
