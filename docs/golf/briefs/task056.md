# Task 056 Golf Brief

## Current Net
- path: `artifacts/optimized/task056.onnx`
- file size: 815 bytes
- cost: 213
- score: 19.638708
- memory: 136
- params: 77
- nodes: 9
- value_info tensors after shape inference: 8
- local gold-correct: True

## Op Histogram

- Slice: 1
- Cast: 1
- Sub: 1
- Conv: 1
- ArgMax: 1
- Squeeze: 1
- Gather: 1
- Unsqueeze: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta -1.441103
- cost 314: score 19.250607, delta -0.388101

## Examples
- train: 7 shown
- test: 3 shown
- arc-gen: 3 shown, 33 remaining

### train[1]
input 3x3 -> output 1x1

input:
```text
550
505
050
```

output:
```text
1
```

### train[2]
input 3x3 -> output 1x1

input:
```text
808
080
808
```

output:
```text
2
```

### train[3]
input 3x3 -> output 1x1

input:
```text
505
050
505
```

output:
```text
2
```

### train[4]
input 3x3 -> output 1x1

input:
```text
011
011
100
```

output:
```text
3
```

### train[5]
input 3x3 -> output 1x1

input:
```text
088
088
800
```

output:
```text
3
```

### train[6]
input 3x3 -> output 1x1

input:
```text
440
404
040
```

output:
```text
1
```

### train[7]
input 3x3 -> output 1x1

input:
```text
050
555
050
```

output:
```text
6
```

### test[1]
input 3x3 -> output 1x1

input:
```text
080
888
080
```

output:
```text
6
```

### test[2]
input 3x3 -> output 1x1

input:
```text
770
707
070
```

output:
```text
1
```

### test[3]
input 3x3 -> output 1x1

input:
```text
202
020
202
```

output:
```text
2
```

### arc-gen[1]
input 3x3 -> output 1x1

input:
```text
880
808
080
```

output:
```text
1
```

### arc-gen[2]
input 3x3 -> output 1x1

input:
```text
660
606
060
```

output:
```text
1
```

### arc-gen[3]
input 3x3 -> output 1x1

input:
```text
033
033
300
```

output:
```text
3
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 56 --onnx path/to/candidate.onnx
```
