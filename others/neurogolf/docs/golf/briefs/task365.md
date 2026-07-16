# Task 365 Golf Brief

## Current Net
- path: `artifacts/optimized/task365.onnx`
- file size: 5772 bytes
- cost: 24399
- score: 14.897703
- memory: 24344
- params: 55
- nodes: 179
- value_info tensors after shape inference: 178
- local gold-correct: True

## Op Histogram

- Where: 27
- Equal: 22
- ReduceSum: 22
- CumSum: 21
- Unsqueeze: 19
- Mul: 19
- Sub: 14
- Cast: 7
- ReduceMax: 6
- Greater: 5
- Slice: 4
- Squeeze: 3
- Conv: 2
- MatMul: 2
- Gather: 1
- Concat: 1
- Sum: 1
- Transpose: 1
- Add: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +3.299903
- cost 314: score 19.250607, delta +4.352904

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 10x10 -> output 5x4

input:
```text
0000008888
0888808228
0818808888
0882808218
0888808888
0000000000
0008888880
0008882880
0008281880
0008188880
```

output:
```text
8888
8228
8888
8218
8888
```

### train[2]
input 10x10 -> output 3x3

input:
```text
1118000000
1811018818
8281081828
1118088881
8188081282
0000088818
0000011818
0822081182
0221000000
0218000000
```

output:
```text
822
221
218
```

### train[3]
input 10x10 -> output 6x4

input:
```text
2888000000
8818000000
1888000000
8882001882
8281008818
8188008288
0000008881
0000001888
0000008818
0000000000
```

output:
```text
2888
8818
1888
8882
8281
8188
```

### test[1]
input 10x10 -> output 6x3

input:
```text
2888000000
8818002810
1281008880
8888002180
0000008820
0000002810
0128201880
0881800000
0128100000
0000000000
```

output:
```text
281
888
218
882
281
188
```

### arc-gen[1]
input 10x10 -> output 5x5

input:
```text
0000000000
0000000000
0000000000
0000000000
0000011181
0000081281
8811011188
2888018818
8811012111
0000000000
```

output:
```text
11181
81281
11188
18818
12111
```

### arc-gen[2]
input 10x10 -> output 3x6

input:
```text
0000000000
0000000000
8218210000
8888880000
8828810000
0000000000
0000881888
0000281888
0000188818
0000818881
```

output:
```text
821821
888888
882881
```

### arc-gen[3]
input 10x10 -> output 5x3

input:
```text
0018800000
0012800000
0012100000
0000000000
0111000000
0228000000
0888000000
0218000000
0818000000
0000000000
```

output:
```text
111
228
888
218
818
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 365 --onnx path/to/candidate.onnx
```
