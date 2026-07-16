# Task 016 Golf Brief

## Current Net
- path: `artifacts/optimized/task016.onnx`
- file size: 158 bytes
- cost: 10
- score: 22.697415
- memory: 0
- params: 10
- nodes: 1
- value_info tensors after shape inference: 0
- local gold-correct: True

## Op Histogram

- Gather: 1

## Targets

- cost 900: score 18.197605, delta -4.499810
- cost 314: score 19.250607, delta -3.446808

## Examples
- train: 4 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 3x3 -> output 3x3

input:
```text
312
312
312
```

output:
```text
456
456
456
```

### train[2]
input 3x3 -> output 3x3

input:
```text
238
238
238
```

output:
```text
649
649
649
```

### train[3]
input 3x3 -> output 3x3

input:
```text
586
586
586
```

output:
```text
192
192
192
```

### train[4]
input 3x3 -> output 3x3

input:
```text
942
942
942
```

output:
```text
836
836
836
```

### test[1]
input 3x3 -> output 3x3

input:
```text
813
813
813
```

output:
```text
954
954
954
```

### arc-gen[1]
input 3x3 -> output 3x3

input:
```text
925
925
925
```

output:
```text
861
861
861
```

### arc-gen[2]
input 3x3 -> output 3x3

input:
```text
658
658
658
```

output:
```text
219
219
219
```

### arc-gen[3]
input 3x3 -> output 3x3

input:
```text
918
918
918
```

output:
```text
859
859
859
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 16 --onnx path/to/candidate.onnx
```
