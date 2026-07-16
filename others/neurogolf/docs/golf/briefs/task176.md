# Task 176 Golf Brief

## Current Net
- path: `artifacts/optimized/task176.onnx`
- file size: 1126 bytes
- cost: 7920
- score: 16.022854
- memory: 7800
- params: 120
- nodes: 7
- value_info tensors after shape inference: 6
- local gold-correct: True

## Op Histogram

- Mul: 2
- Slice: 1
- Cast: 1
- Conv: 1
- Sum: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +2.174752
- cost 314: score 19.250607, delta +3.227753

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 18 remaining

### train[1]
input 3x10 -> output 3x10

input:
```text
2000200020
0202020202
0020002000
```

output:
```text
2000244420
4202024202
4420002000
```

### train[2]
input 3x15 -> output 3x15

input:
```text
200020002000200
020202020202020
002000200020002
```

output:
```text
200024442000200
420202420202420
442000200024442
```

### train[3]
input 3x18 -> output 3x18

input:
```text
200020002000200020
020202020202020202
002000200020002000
```

output:
```text
200024442000200024
420202420202420202
442000200024442000
```

### test[1]
input 3x25 -> output 3x25

input:
```text
2000200020002000200020002
0202020202020202020202020
0020002000200020002000200
```

output:
```text
2000244420002000244420002
4202024202024202024202024
4420002000244420002000244
```

### arc-gen[1]
input 3x22 -> output 3x22

input:
```text
2000200020002000200020
0202020202020202020202
0020002000200020002000
```

output:
```text
2000244420002000244420
4202024202024202024202
4420002000244420002000
```

### arc-gen[2]
input 3x8 -> output 3x8

input:
```text
20002000
02020202
00200020
```

output:
```text
20002444
42020242
44200020
```

### arc-gen[3]
input 3x16 -> output 3x16

input:
```text
2000200020002000
0202020202020202
0020002000200020
```

output:
```text
2000244420002000
4202024202024202
4420002000244420
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 176 --onnx path/to/candidate.onnx
```
