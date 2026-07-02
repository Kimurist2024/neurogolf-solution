# Task 380 Golf Brief

## Current Net
- path: `artifacts/optimized/task380.onnx`
- file size: 396 bytes
- cost: 728
- score: 18.409699
- memory: 720
- params: 8
- nodes: 3
- value_info tensors after shape inference: 2
- local gold-correct: True

## Op Histogram

- Slice: 1
- Transpose: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta -0.212094
- cost 314: score 19.250607, delta +0.840908

## Examples
- train: 4 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 3x3 -> output 3x3

input:
```text
900
999
999
```

output:
```text
099
099
999
```

### train[2]
input 3x3 -> output 3x3

input:
```text
666
000
660
```

output:
```text
600
606
606
```

### train[3]
input 3x3 -> output 3x3

input:
```text
009
009
999
```

output:
```text
999
009
009
```

### train[4]
input 3x3 -> output 3x3

input:
```text
202
002
022
```

output:
```text
222
002
200
```

### test[1]
input 3x3 -> output 3x3

input:
```text
000
500
055
```

output:
```text
005
005
050
```

### arc-gen[1]
input 3x3 -> output 3x3

input:
```text
777
070
077
```

output:
```text
707
777
700
```

### arc-gen[2]
input 3x3 -> output 3x3

input:
```text
040
004
000
```

output:
```text
040
400
000
```

### arc-gen[3]
input 3x3 -> output 3x3

input:
```text
010
000
010
```

output:
```text
000
101
000
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 380 --onnx path/to/candidate.onnx
```
