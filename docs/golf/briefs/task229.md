# Task 229 Golf Brief

## Current Net
- path: `artifacts/optimized/task229.onnx`
- file size: 1379 bytes
- cost: 1409
- score: 17.749364
- memory: 1364
- params: 45
- nodes: 14
- value_info tensors after shape inference: 13
- local gold-correct: True

## Op Histogram

- ReduceSum: 2
- Reshape: 2
- Where: 2
- Slice: 1
- Cast: 1
- ArgMax: 1
- Equal: 1
- Sub: 1
- Mul: 1
- Sum: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +0.448241
- cost 314: score 19.250607, delta +1.501243

## Examples
- train: 4 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 3x3 -> output 3x3

input:
```text
222
218
288
```

output:
```text
222
255
255
```

### train[2]
input 3x3 -> output 3x3

input:
```text
111
813
822
```

output:
```text
111
515
555
```

### train[3]
input 3x3 -> output 3x3

input:
```text
222
882
222
```

output:
```text
222
552
222
```

### train[4]
input 3x3 -> output 3x3

input:
```text
338
444
811
```

output:
```text
555
444
555
```

### test[1]
input 3x3 -> output 3x3

input:
```text
132
332
132
```

output:
```text
535
335
535
```

### arc-gen[1]
input 3x3 -> output 3x3

input:
```text
249
729
922
```

output:
```text
255
525
522
```

### arc-gen[2]
input 3x3 -> output 3x3

input:
```text
776
667
777
```

output:
```text
775
557
777
```

### arc-gen[3]
input 3x3 -> output 3x3

input:
```text
911
919
191
```

output:
```text
511
515
151
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 229 --onnx path/to/candidate.onnx
```
