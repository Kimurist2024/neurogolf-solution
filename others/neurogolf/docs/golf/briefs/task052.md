# Task 052 Golf Brief

## Current Net
- path: `artifacts/optimized/task052.onnx`
- file size: 756 bytes
- cost: 1120
- score: 17.978916
- memory: 1077
- params: 43
- nodes: 11
- value_info tensors after shape inference: 10
- local gold-correct: True

## Op Histogram

- Mul: 2
- Slice: 1
- ReduceSum: 1
- ReduceMax: 1
- Cast: 1
- Equal: 1
- Where: 1
- Sub: 1
- Sum: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +0.218689
- cost 314: score 19.250607, delta +1.271691

## Examples
- train: 4 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 3x3 -> output 3x3

input:
```text
444
232
233
```

output:
```text
555
000
000
```

### train[2]
input 3x3 -> output 3x3

input:
```text
733
666
377
```

output:
```text
000
555
000
```

### train[3]
input 3x3 -> output 3x3

input:
```text
292
444
999
```

output:
```text
000
555
555
```

### train[4]
input 3x3 -> output 3x3

input:
```text
224
224
111
```

output:
```text
000
000
555
```

### test[1]
input 3x3 -> output 3x3

input:
```text
444
323
888
```

output:
```text
555
000
555
```

### arc-gen[1]
input 3x3 -> output 3x3

input:
```text
999
622
262
```

output:
```text
555
000
000
```

### arc-gen[2]
input 3x3 -> output 3x3

input:
```text
677
777
877
```

output:
```text
000
555
000
```

### arc-gen[3]
input 3x3 -> output 3x3

input:
```text
119
999
111
```

output:
```text
000
555
555
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 52 --onnx path/to/candidate.onnx
```
