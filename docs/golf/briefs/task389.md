# Task 389 Golf Brief

## Current Net
- path: `artifacts/optimized/task389.onnx`
- file size: 6222 bytes
- cost: 1510
- score: 17.680135
- memory: 0
- params: 1510
- nodes: 1
- value_info tensors after shape inference: 0
- local gold-correct: True

## Op Histogram

- Conv: 1

## Targets

- cost 900: score 18.197605, delta +0.517470
- cost 314: score 19.250607, delta +1.570472

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 3x3 -> output 3x3

input:
```text
454
555
454
```

output:
```text
040
444
040
```

### train[2]
input 5x5 -> output 5x5

input:
```text
55666
65566
66556
66655
56665
```

output:
```text
66000
06600
00660
00066
60006
```

### train[3]
input 5x5 -> output 5x5

input:
```text
95999
99559
95999
99599
99955
```

output:
```text
09000
00990
09000
00900
00099
```

### test[1]
input 5x5 -> output 5x5

input:
```text
33353
35333
35535
33353
55533
```

output:
```text
00030
03000
03303
00030
33300
```

### arc-gen[1]
input 5x5 -> output 5x5

input:
```text
58888
58888
85888
55888
58888
```

output:
```text
80000
80000
08000
88000
80000
```

### arc-gen[2]
input 3x3 -> output 3x3

input:
```text
858
858
855
```

output:
```text
080
080
088
```

### arc-gen[3]
input 3x3 -> output 3x3

input:
```text
555
885
888
```

output:
```text
888
008
000
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 389 --onnx path/to/candidate.onnx
```
