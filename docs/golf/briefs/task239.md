# Task 239 Golf Brief

## Current Net
- path: `artifacts/optimized/task239.onnx`
- file size: 3012 bytes
- cost: 18839
- score: 15.156316
- memory: 18661
- params: 178
- nodes: 35
- value_info tensors after shape inference: 35
- local gold-correct: True

## Research Queue
- priority rank: 23
- recorded cost: 94467
- recorded memory: 94081
- recorded params: 386
- recorded nodes: 34

## Op Histogram

- Reshape: 8
- Cast: 5
- Less: 3
- ReduceSum: 2
- Slice: 2
- Add: 2
- And: 2
- TopK: 1
- OneHot: 1
- Unsqueeze: 1
- Where: 1
- Transpose: 1
- MatMul: 1
- Greater: 1
- Concat: 1
- GreaterOrEqual: 1
- Mul: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +3.041290
- cost 314: score 19.250607, delta +4.094291

## Examples
- train: 4 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 3x3 -> output 5x3

input:
```text
221
231
111
```

output:
```text
123
120
120
100
100
```

### train[2]
input 3x4 -> output 6x4

input:
```text
3114
2224
4444
```

output:
```text
4213
4210
4200
4000
4000
4000
```

### train[3]
input 4x3 -> output 5x4

input:
```text
882
388
334
334
```

output:
```text
3842
3840
3800
3800
3000
```

### train[4]
input 4x3 -> output 6x3

input:
```text
111
221
281
281
```

output:
```text
128
128
120
120
100
100
```

### test[1]
input 4x4 -> output 6x5

input:
```text
8822
1882
1334
1111
```

output:
```text
18234
18230
18200
18000
10000
10000
```

### arc-gen[1]
input 4x3 -> output 9x2

input:
```text
333
313
313
313
```

output:
```text
31
31
31
30
30
30
30
30
30
```

### arc-gen[2]
input 4x3 -> output 10x2

input:
```text
999
929
929
999
```

output:
```text
92
92
90
90
90
90
90
90
90
90
```

### arc-gen[3]
input 4x3 -> output 8x2

input:
```text
212
212
212
212
```

output:
```text
21
21
21
21
20
20
20
20
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 239 --onnx path/to/candidate.onnx
```
