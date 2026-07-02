# Task 266 Golf Brief

## Current Net
- path: `artifacts/optimized/task266.onnx`
- file size: 3825 bytes
- cost: 910
- score: 18.186555
- memory: 0
- params: 910
- nodes: 1
- value_info tensors after shape inference: 0
- local gold-correct: True

## Op Histogram

- Conv: 1

## Targets

- cost 900: score 18.197605, delta +0.011050
- cost 314: score 19.250607, delta +1.064052

## Examples
- train: 4 shown
- test: 1 shown
- arc-gen: 3 shown, 12 remaining

### train[1]
input 3x5 -> output 3x5

input:
```text
00000
02000
00000
```

output:
```text
30600
00000
80700
```

### train[2]
input 3x5 -> output 3x5

input:
```text
00000
00000
00002
```

output:
```text
00000
00030
00000
```

### train[3]
input 3x5 -> output 3x5

input:
```text
00200
00000
00000
```

output:
```text
00000
08070
00000
```

### train[4]
input 3x5 -> output 3x5

input:
```text
00000
00020
00000
```

output:
```text
00306
00000
00807
```

### test[1]
input 3x5 -> output 3x5

input:
```text
00000
00002
00000
```

output:
```text
00030
00000
00080
```

### arc-gen[1]
input 3x5 -> output 3x5

input:
```text
00000
00000
20000
```

output:
```text
00000
06000
00000
```

### arc-gen[2]
input 3x5 -> output 3x5

input:
```text
00200
00000
00000
```

output:
```text
00000
08070
00000
```

### arc-gen[3]
input 3x5 -> output 3x5

input:
```text
00020
00000
00000
```

output:
```text
00000
00807
00000
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 266 --onnx path/to/candidate.onnx
```
