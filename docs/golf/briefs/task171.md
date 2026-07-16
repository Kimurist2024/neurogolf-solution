# Task 171 Golf Brief

## Current Net
- path: `artifacts/optimized/task171.onnx`
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
- arc-gen: 3 shown, 46 remaining

### train[1]
input 3x3 -> output 3x3

input:
```text
000
000
000
```

output:
```text
888
808
888
```

### train[2]
input 4x3 -> output 4x3

input:
```text
000
000
000
000
```

output:
```text
888
808
808
888
```

### train[3]
input 5x4 -> output 5x4

input:
```text
0000
0000
0000
0000
0000
```

output:
```text
8888
8008
8008
8008
8888
```

### train[4]
input 5x6 -> output 5x6

input:
```text
000000
000000
000000
000000
000000
```

output:
```text
888888
800008
800008
800008
888888
```

### test[1]
input 7x6 -> output 7x6

input:
```text
000000
000000
000000
000000
000000
000000
000000
```

output:
```text
888888
800008
800008
800008
800008
800008
888888
```

### arc-gen[1]
input 3x7 -> output 3x7

input:
```text
0000000
0000000
0000000
```

output:
```text
8888888
8000008
8888888
```

### arc-gen[2]
input 5x3 -> output 5x3

input:
```text
000
000
000
000
000
```

output:
```text
888
808
808
808
888
```

### arc-gen[3]
input 6x3 -> output 6x3

input:
```text
000
000
000
000
000
000
```

output:
```text
888
808
808
808
808
888
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 171 --onnx path/to/candidate.onnx
```
