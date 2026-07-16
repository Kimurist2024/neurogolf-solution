# Task 337 Golf Brief

## Current Net
- path: `artifacts/optimized/task337.onnx`
- file size: 159 bytes
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
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 5x5 -> output 5x5

input:
```text
27888
55654
85552
88436
65193
```

output:
```text
27555
88684
58882
55436
68193
```

### train[2]
input 3x3 -> output 3x3

input:
```text
351
458
249
```

output:
```text
381
485
249
```

### train[3]
input 3x3 -> output 3x3

input:
```text
653
575
882
```

output:
```text
683
878
552
```

### test[1]
input 4x4 -> output 4x4

input:
```text
8845
3875
3719
6488
```

output:
```text
5548
3578
3719
6455
```

### arc-gen[1]
input 5x5 -> output 5x5

input:
```text
58395
55588
72824
78888
89245
```

output:
```text
85398
88855
72524
75555
59248
```

### arc-gen[2]
input 3x3 -> output 3x3

input:
```text
655
288
788
```

output:
```text
688
255
755
```

### arc-gen[3]
input 3x3 -> output 3x3

input:
```text
828
683
756
```

output:
```text
525
653
786
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 337 --onnx path/to/candidate.onnx
```
