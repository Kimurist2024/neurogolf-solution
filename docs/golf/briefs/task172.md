# Task 172 Golf Brief

## Current Net
- path: `artifacts/optimized/task172.onnx`
- file size: 479 bytes
- cost: 30
- score: 21.598803
- memory: 0
- params: 30
- nodes: 1
- value_info tensors after shape inference: 0
- local gold-correct: True

## Op Histogram

- Gather: 1

## Targets

- cost 900: score 18.197605, delta -3.401197
- cost 314: score 19.250607, delta -2.348196

## Examples
- train: 4 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 3x3 -> output 6x3

input:
```text
914
914
211
```

output:
```text
914
914
211
211
914
914
```

### train[2]
input 3x3 -> output 6x3

input:
```text
484
767
878
```

output:
```text
484
767
878
878
767
484
```

### train[3]
input 3x3 -> output 6x3

input:
```text
777
955
517
```

output:
```text
777
955
517
517
955
777
```

### train[4]
input 3x3 -> output 6x3

input:
```text
269
269
292
```

output:
```text
269
269
292
292
269
269
```

### test[1]
input 3x3 -> output 6x3

input:
```text
292
852
228
```

output:
```text
292
852
228
228
852
292
```

### arc-gen[1]
input 3x3 -> output 6x3

input:
```text
296
429
449
```

output:
```text
296
429
449
449
429
296
```

### arc-gen[2]
input 3x3 -> output 6x3

input:
```text
267
776
222
```

output:
```text
267
776
222
222
776
267
```

### arc-gen[3]
input 3x3 -> output 6x3

input:
```text
178
781
811
```

output:
```text
178
781
811
811
781
178
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 172 --onnx path/to/candidate.onnx
```
