# Task 167 Golf Brief

## Current Net
- path: `artifacts/optimized/task167.onnx`
- file size: 1420 bytes
- cost: 683
- score: 18.473505
- memory: 595
- params: 88
- nodes: 9
- value_info tensors after shape inference: 8
- local gold-correct: True

## Op Histogram

- ReduceSum: 2
- Slice: 1
- Greater: 1
- Cast: 1
- Sub: 1
- Gather: 1
- Conv: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta -0.275900
- cost 314: score 19.250607, delta +0.777102

## Examples
- train: 5 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 3x3 -> output 3x3

input:
```text
222
323
333
```

output:
```text
500
050
005
```

### train[2]
input 3x3 -> output 3x3

input:
```text
333
422
442
```

output:
```text
005
050
500
```

### train[3]
input 3x3 -> output 3x3

input:
```text
444
444
444
```

output:
```text
555
000
000
```

### train[4]
input 3x3 -> output 3x3

input:
```text
333
333
333
```

output:
```text
555
000
000
```

### train[5]
input 3x3 -> output 3x3

input:
```text
444
444
333
```

output:
```text
500
050
005
```

### test[1]
input 3x3 -> output 3x3

input:
```text
444
232
323
```

output:
```text
005
050
500
```

### arc-gen[1]
input 3x3 -> output 3x3

input:
```text
423
342
233
```

output:
```text
005
050
500
```

### arc-gen[2]
input 3x3 -> output 3x3

input:
```text
333
333
333
```

output:
```text
555
000
000
```

### arc-gen[3]
input 3x3 -> output 3x3

input:
```text
444
222
442
```

output:
```text
500
050
005
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 167 --onnx path/to/candidate.onnx
```
