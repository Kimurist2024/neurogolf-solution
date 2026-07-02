# Task 116 Golf Brief

## Current Net
- path: `artifacts/optimized/task116.onnx`
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
input 3x4 -> output 6x4

input:
```text
9959
5599
9599
```

output:
```text
9599
5599
9959
9959
5599
9599
```

### train[2]
input 3x4 -> output 6x4

input:
```text
4114
1111
4441
```

output:
```text
4441
1111
4114
4114
1111
4441
```

### train[3]
input 3x4 -> output 6x4

input:
```text
9494
9944
4444
```

output:
```text
4444
9944
9494
9494
9944
4444
```

### train[4]
input 3x4 -> output 6x4

input:
```text
3355
3553
5533
```

output:
```text
5533
3553
3355
3355
3553
5533
```

### test[1]
input 3x4 -> output 6x4

input:
```text
4499
4444
4499
```

output:
```text
4499
4444
4499
4499
4444
4499
```

### arc-gen[1]
input 3x4 -> output 6x4

input:
```text
2277
7227
7722
```

output:
```text
7722
7227
2277
2277
7227
7722
```

### arc-gen[2]
input 3x4 -> output 6x4

input:
```text
4888
4488
8888
```

output:
```text
8888
4488
4888
4888
4488
8888
```

### arc-gen[3]
input 3x4 -> output 6x4

input:
```text
5555
5655
6565
```

output:
```text
6565
5655
5555
5555
5655
6565
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 116 --onnx path/to/candidate.onnx
```
