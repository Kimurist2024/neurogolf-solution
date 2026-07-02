# Task 315 Golf Brief

## Current Net
- path: `artifacts/optimized/task315.onnx`
- file size: 1461 bytes
- cost: 3387
- score: 16.872300
- memory: 3348
- params: 39
- nodes: 9
- value_info tensors after shape inference: 8
- local gold-correct: True

## Op Histogram

- Slice: 2
- ConvTranspose: 1
- Tile: 1
- Mul: 1
- ReduceSum: 1
- Sub: 1
- Concat: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +1.325305
- cost 314: score 19.250607, delta +2.378307

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 3x3 -> output 9x9

input:
```text
100
210
001
```

output:
```text
000000000
000000000
000000000
100000000
210000000
001000000
000000000
000000000
000000000
```

### train[2]
input 3x3 -> output 9x9

input:
```text
012
110
200
```

output:
```text
000000012
000000110
000000200
000000000
000000000
000000000
012000000
110000000
200000000
```

### train[3]
input 3x3 -> output 9x9

input:
```text
212
021
210
```

output:
```text
212000212
021000021
210000210
000212000
000021000
000210000
212000000
021000000
210000000
```

### test[1]
input 3x3 -> output 9x9

input:
```text
122
201
120
```

output:
```text
000122122
000201201
000120120
122000000
201000000
120000000
000122000
000201000
000120000
```

### arc-gen[1]
input 3x3 -> output 9x9

input:
```text
221
020
011
```

output:
```text
221221000
020020000
011011000
000221000
000020000
000011000
000000000
000000000
000000000
```

### arc-gen[2]
input 3x3 -> output 9x9

input:
```text
020
001
000
```

output:
```text
000020000
000001000
000000000
000000000
000000000
000000000
000000000
000000000
000000000
```

### arc-gen[3]
input 3x3 -> output 9x9

input:
```text
020
000
010
```

output:
```text
000020000
000000000
000010000
000000000
000000000
000000000
000000000
000000000
000000000
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 315 --onnx path/to/candidate.onnx
```
