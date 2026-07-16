# Task 153 Golf Brief

## Current Net
- path: `artifacts/optimized/task153.onnx`
- file size: 8954 bytes
- cost: 20583
- score: 15.067779
- memory: 20491
- params: 92
- nodes: 72
- value_info tensors after shape inference: 72
- local gold-correct: True

## Op Histogram

- Reshape: 12
- Cast: 8
- Gather: 6
- Unsqueeze: 6
- Add: 6
- ReduceMax: 4
- MatMul: 4
- Slice: 3
- And: 3
- Transpose: 3
- ReduceSum: 2
- Greater: 2
- ArgMax: 2
- Sub: 2
- Pad: 2
- TopK: 1
- Mul: 1
- GatherElements: 1
- Equal: 1
- Less: 1
- Concat: 1
- OneHot: 1

## Targets

- cost 900: score 18.197605, delta +3.129826
- cost 314: score 19.250607, delta +4.182828

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 258 remaining

### train[1]
input 10x10 -> output 3x3

input:
```text
0000000000
0000000000
0330000000
0300000000
0300000000
0000000000
0000000000
0000000007
0000000077
0000000077
```

output:
```text
337
377
377
```

### train[2]
input 10x10 -> output 3x3

input:
```text
0000000040
0000000044
0006660000
0000660000
0000060000
0000000000
0000000000
0000000000
0000000000
0000000000
```

output:
```text
666
466
446
```

### train[3]
input 10x10 -> output 3x3

input:
```text
0000000000
0000000000
0000000000
0000000000
0000300000
0003330000
0000000000
0000000000
0111000000
0101000000
```

output:
```text
111
131
333
```

### test[1]
input 10x10 -> output 3x3

input:
```text
0000000000
0000000000
0000200000
0002200000
0000000000
0000000000
0000000880
0000000800
0000000888
0000000000
```

output:
```text
882
822
888
```

### arc-gen[1]
input 10x10 -> output 3x3

input:
```text
0000000000
0000666000
0000606000
0000006000
0000000000
0000000000
0000000040
0000000440
0000000000
0000000000
```

output:
```text
666
646
446
```

### arc-gen[2]
input 10x10 -> output 3x3

input:
```text
0000000000
0000000000
0550000000
0050000000
0000000000
0000000000
0000000000
0000000888
0000000800
0000000880
```

output:
```text
888
855
885
```

### arc-gen[3]
input 10x10 -> output 3x3

input:
```text
0000000000
0000000000
0000000000
0000000000
0000777000
0000770000
0000700000
0000000000
0009000000
0099000000
```

output:
```text
777
779
799
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 153 --onnx path/to/candidate.onnx
```
