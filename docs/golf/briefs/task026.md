# Task 026 Golf Brief

## Current Net
- path: `artifacts/optimized/task026.onnx`
- file size: 1309 bytes
- cost: 724
- score: 18.415209
- memory: 690
- params: 34
- nodes: 13
- value_info tensors after shape inference: 12
- local gold-correct: True

## Op Histogram

- Sub: 5
- Slice: 2
- Cast: 2
- Mul: 1
- Sum: 1
- Concat: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta -0.217603
- cost 314: score 19.250607, delta +0.835398

## Examples
- train: 5 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 5x7 -> output 5x3

input:
```text
0991999
0091990
9091990
0001900
0991999
```

output:
```text
000
000
000
088
000
```

### train[2]
input 5x7 -> output 5x3

input:
```text
0001900
9091999
0991999
0001999
0991999
```

output:
```text
088
000
000
000
000
```

### train[3]
input 5x7 -> output 5x3

input:
```text
9001909
9001090
9001900
0991099
0091090
```

output:
```text
080
008
088
800
800
```

### train[4]
input 5x7 -> output 5x3

input:
```text
0991909
9001900
9991999
0901000
9001900
```

output:
```text
000
088
000
808
088
```

### train[5]
input 5x7 -> output 5x3

input:
```text
0991909
9091999
9991009
9001900
9991009
```

output:
```text
000
000
000
088
000
```

### test[1]
input 5x7 -> output 5x3

input:
```text
9901090
0991000
9901090
9991909
0991099
```

output:
```text
008
800
008
000
800
```

### arc-gen[1]
input 5x7 -> output 5x3

input:
```text
9901099
0001900
0991099
9991999
0001909
```

output:
```text
000
088
800
000
080
```

### arc-gen[2]
input 5x7 -> output 5x3

input:
```text
0991009
9991999
0901099
9001999
9001009
```

output:
```text
800
000
800
000
080
```

### arc-gen[3]
input 5x7 -> output 5x3

input:
```text
0001090
0901090
0991009
0001090
9091990
```

output:
```text
808
808
800
808
000
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 26 --onnx path/to/candidate.onnx
```
