# Task 385 Golf Brief

## Current Net
- path: `artifacts/optimized/task385.onnx`
- file size: 175 bytes
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
- train: 2 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 10x4 -> output 10x4

input:
```text
0000
0000
0000
0000
0000
0009
0039
0039
2039
2439
```

output:
```text
2439
2039
0039
0039
0009
0009
0039
0039
2039
2439
```

### train[2]
input 10x4 -> output 10x4

input:
```text
0000
0000
0000
0000
0000
0002
0002
0802
0382
3382
```

output:
```text
3382
0382
0802
0002
0002
0002
0002
0802
0382
3382
```

### test[1]
input 10x4 -> output 10x4

input:
```text
0000
0000
0000
0000
0000
0100
7100
7130
7133
7144
```

output:
```text
7144
7133
7130
7100
0100
0100
7100
7130
7133
7144
```

### arc-gen[1]
input 10x4 -> output 10x4

input:
```text
0000
0000
0000
0000
0000
0700
6700
6700
6702
6752
```

output:
```text
6752
6702
6700
6700
0700
0700
6700
6700
6702
6752
```

### arc-gen[2]
input 10x4 -> output 10x4

input:
```text
0000
0000
0000
0000
0000
0000
0092
0092
9192
9392
```

output:
```text
9392
9192
0092
0092
0000
0000
0092
0092
9192
9392
```

### arc-gen[3]
input 10x4 -> output 10x4

input:
```text
0000
0000
0000
0000
0000
0500
3500
3500
3590
3599
```

output:
```text
3599
3590
3500
3500
0500
0500
3500
3500
3590
3599
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 385 --onnx path/to/candidate.onnx
```
