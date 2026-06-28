# Task 394 Golf Brief

## Current Net
- path: `artifacts/optimized/task394.onnx`
- file size: 3038 bytes
- cost: 4498
- score: 16.588612
- memory: 4462
- params: 36
- nodes: 79
- value_info tensors after shape inference: 78
- local gold-correct: True

## Op Histogram

- Cast: 15
- Add: 10
- Unsqueeze: 8
- Mod: 7
- Reshape: 4
- Sub: 4
- ReduceSum: 3
- Greater: 3
- Less: 3
- Where: 3
- Clip: 3
- Concat: 3
- Slice: 2
- ReduceMax: 2
- ArgMax: 2
- Gather: 2
- Mul: 2
- Sign: 1
- Expand: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +1.608993
- cost 314: score 19.250607, delta +2.661995

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 5x5 -> output 2x2

input:
```text
21212
11111
21212
00111
00212
```

output:
```text
11
21
```

### train[2]
input 4x4 -> output 1x1

input:
```text
8606
6868
8686
6868
```

output:
```text
8
```

### train[3]
input 7x7 -> output 2x2

input:
```text
2252252
2252252
5555555
2252252
2252252
5555500
2252200
```

output:
```text
55
52
```

### test[1]
input 7x7 -> output 3x3

input:
```text
8188000
1881000
8818000
8188188
1881881
8818818
8188188
```

output:
```text
188
881
818
```

### arc-gen[1]
input 4x4 -> output 1x1

input:
```text
8083
3838
8383
3838
```

output:
```text
3
```

### arc-gen[2]
input 4x4 -> output 1x1

input:
```text
6262
2222
6202
2222
```

output:
```text
6
```

### arc-gen[3]
input 4x4 -> output 1x1

input:
```text
8282
2828
8282
2028
```

output:
```text
8
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 394 --onnx path/to/candidate.onnx
```
