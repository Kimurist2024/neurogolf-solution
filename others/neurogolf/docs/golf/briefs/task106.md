# Task 106 Golf Brief

## Current Net
- path: `artifacts/optimized/task106.onnx`
- file size: 2939 bytes
- cost: 1164
- score: 17.940382
- memory: 1039
- params: 125
- nodes: 13
- value_info tensors after shape inference: 12
- local gold-correct: True

## Op Histogram

- Reshape: 3
- Slice: 2
- Equal: 2
- ArgMax: 1
- Cast: 1
- Concat: 1
- Where: 1
- GatherElements: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +0.257223
- cost 314: score 19.250607, delta +1.310225

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 2x2 -> output 4x4

input:
```text
86
68
```

output:
```text
8668
6886
6886
8668
```

### train[2]
input 3x3 -> output 6x6

input:
```text
778
778
888
```

output:
```text
778877
778877
888888
888888
778877
778877
```

### train[3]
input 3x3 -> output 6x6

input:
```text
699
644
644
```

output:
```text
699666
644449
644449
944446
944446
666996
```

### test[1]
input 3x3 -> output 6x6

input:
```text
141
494
919
```

output:
```text
141941
494194
919941
149919
491494
149141
```

### arc-gen[1]
input 2x2 -> output 4x4

input:
```text
83
35
```

output:
```text
8338
3553
3553
8338
```

### arc-gen[2]
input 2x2 -> output 4x4

input:
```text
88
82
```

output:
```text
8888
8228
8228
8888
```

### arc-gen[3]
input 2x2 -> output 4x4

input:
```text
12
21
```

output:
```text
1221
2112
2112
1221
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 106 --onnx path/to/candidate.onnx
```
