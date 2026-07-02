# Task 155 Golf Brief

## Current Net
- path: `artifacts/optimized/task155.onnx`
- file size: 618 bytes
- cost: 704
- score: 18.443222
- memory: 668
- params: 36
- nodes: 10
- value_info tensors after shape inference: 9
- local gold-correct: True

## Op Histogram

- ReduceSum: 2
- Sub: 2
- Greater: 1
- Cast: 1
- Less: 1
- Where: 1
- Squeeze: 1
- Gather: 1

## Targets

- cost 900: score 18.197605, delta -0.245616
- cost 314: score 19.250607, delta +0.807385

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 5x5 -> output 5x5

input:
```text
81214
44248
37248
27787
87748
```

output:
```text
87748
27787
37248
44248
81214
```

### train[2]
input 5x5 -> output 5x5

input:
```text
73312
18241
27872
77418
81771
```

output:
```text
81771
77418
27872
18241
73312
```

### train[3]
input 7x7 -> output 7x7

input:
```text
2743483
2371233
8743224
1121447
2431141
4874482
7384328
```

output:
```text
7384328
4874482
2431141
1121447
8743224
2371233
2743483
```

### test[1]
input 7x7 -> output 7x7

input:
```text
2813241
4411434
1111473
1123813
4111784
3284184
1471234
```

output:
```text
1471234
3284184
4111784
1123813
1111473
4411434
2813241
```

### arc-gen[1]
input 8x8 -> output 8x8

input:
```text
18427134
72144111
24141717
71221828
44371717
22127112
73174347
28374744
```

output:
```text
28374744
73174347
22127112
44371717
71221828
24141717
72144111
18427134
```

### arc-gen[2]
input 4x4 -> output 4x4

input:
```text
3778
1277
7477
8474
```

output:
```text
8474
7477
1277
3778
```

### arc-gen[3]
input 4x4 -> output 4x4

input:
```text
4118
3482
4237
2338
```

output:
```text
2338
4237
3482
4118
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 155 --onnx path/to/candidate.onnx
```
