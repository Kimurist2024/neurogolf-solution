# Task 322 Golf Brief

## Current Net
- path: `artifacts/optimized/task322.onnx`
- file size: 2222 bytes
- cost: 510
- score: 18.765589
- memory: 0
- params: 510
- nodes: 1
- value_info tensors after shape inference: 0
- local gold-correct: True

## Op Histogram

- Conv: 1

## Targets

- cost 900: score 18.197605, delta -0.567984
- cost 314: score 19.250607, delta +0.485018

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 3x3 -> output 3x3

input:
```text
006
040
300
```

output:
```text
006
046
346
```

### train[2]
input 3x3 -> output 3x3

input:
```text
020
708
000
```

output:
```text
020
728
728
```

### train[3]
input 3x3 -> output 3x3

input:
```text
400
020
000
```

output:
```text
400
420
420
```

### test[1]
input 3x3 -> output 3x3

input:
```text
408
000
070
```

output:
```text
408
408
478
```

### arc-gen[1]
input 3x3 -> output 3x3

input:
```text
020
000
700
```

output:
```text
020
020
720
```

### arc-gen[2]
input 3x3 -> output 3x3

input:
```text
800
010
003
```

output:
```text
800
810
813
```

### arc-gen[3]
input 3x3 -> output 3x3

input:
```text
006
050
000
```

output:
```text
006
056
056
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 322 --onnx path/to/candidate.onnx
```
