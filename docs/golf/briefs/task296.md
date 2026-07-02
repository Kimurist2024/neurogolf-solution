# Task 296 Golf Brief

## Current Net
- path: `artifacts/optimized/task296.onnx`
- file size: 2803 bytes
- cost: 4107
- score: 16.679552
- memory: 3762
- params: 345
- nodes: 10
- value_info tensors after shape inference: 9
- local gold-correct: True

## Op Histogram

- Reshape: 2
- Slice: 1
- MatMul: 1
- Greater: 1
- Cast: 1
- ReduceSum: 1
- Sub: 1
- Concat: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +1.518053
- cost 314: score 19.250607, delta +2.571055

## Examples
- train: 5 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 5x7 -> output 3x3

input:
```text
0800080
8800088
0000000
8800088
0800080
```

output:
```text
080
888
080
```

### train[2]
input 5x7 -> output 3x3

input:
```text
2200022
0000002
0000000
0200020
2000002
```

output:
```text
222
022
202
```

### train[3]
input 5x7 -> output 3x3

input:
```text
4400040
0000044
0000000
0000000
4000004
```

output:
```text
440
044
404
```

### train[4]
input 5x7 -> output 3x3

input:
```text
4000004
0000000
0000000
0000000
4000044
```

output:
```text
404
000
444
```

### train[5]
input 5x7 -> output 3x3

input:
```text
0300030
3000003
0000000
0000000
0000003
```

output:
```text
030
303
003
```

### test[1]
input 5x7 -> output 3x3

input:
```text
0000011
1000000
0000000
0000000
0100001
```

output:
```text
011
100
011
```

### arc-gen[1]
input 5x7 -> output 3x3

input:
```text
0000000
8800088
0000000
0000000
8800080
```

output:
```text
000
888
880
```

### arc-gen[2]
input 5x7 -> output 3x3

input:
```text
5000050
0000000
0000000
0500000
0500050
```

output:
```text
550
050
050
```

### arc-gen[3]
input 5x7 -> output 3x3

input:
```text
8800008
8000008
0000000
8800008
0800008
```

output:
```text
888
888
088
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 296 --onnx path/to/candidate.onnx
```
