# Task 386 Golf Brief

## Current Net
- path: `artifacts/optimized/task386.onnx`
- file size: 738 bytes
- cost: 527
- score: 18.732799
- memory: 492
- params: 35
- nodes: 12
- value_info tensors after shape inference: 11
- local gold-correct: True

## Op Histogram

- Cast: 4
- Slice: 2
- Not: 2
- Or: 1
- Concat: 1
- Conv: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta -0.535194
- cost 314: score 19.250607, delta +0.517808

## Examples
- train: 5 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 4x7 -> output 4x3

input:
```text
7701500
7001500
0001505
0001550
```

output:
```text
003
033
030
003
```

### train[2]
input 4x7 -> output 4x3

input:
```text
7701500
0001000
7701500
0771550
```

output:
```text
003
333
003
000
```

### train[3]
input 4x7 -> output 4x3

input:
```text
0771500
0071055
0701550
0071000
```

output:
```text
000
300
003
330
```

### train[4]
input 4x7 -> output 4x3

input:
```text
7071550
7701555
0701000
0001505
```

output:
```text
000
000
303
030
```

### train[5]
input 4x7 -> output 4x3

input:
```text
7001050
0071500
0001555
7771555
```

output:
```text
003
030
000
000
```

### test[1]
input 4x7 -> output 4x3

input:
```text
0001050
7771550
0001500
7001555
```

output:
```text
303
000
033
000
```

### arc-gen[1]
input 4x7 -> output 4x3

input:
```text
0071555
7701005
0771500
7001000
```

output:
```text
000
000
000
033
```

### arc-gen[2]
input 4x7 -> output 4x3

input:
```text
7001005
0771050
0001500
0001055
```

output:
```text
030
300
033
300
```

### arc-gen[3]
input 4x7 -> output 4x3

input:
```text
7771055
7701000
7701550
7071555
```

output:
```text
000
003
003
000
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 386 --onnx path/to/candidate.onnx
```
