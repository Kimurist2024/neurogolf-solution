# Task 235 Golf Brief

## Current Net
- path: `artifacts/optimized/task235.onnx`
- file size: 3142 bytes
- cost: 1230
- score: 17.885231
- memory: 1116
- params: 114
- nodes: 35
- value_info tensors after shape inference: 34
- local gold-correct: True

## Op Histogram

- Slice: 3
- Cast: 3
- Reshape: 3
- Mul: 3
- Sub: 3
- MatMul: 3
- ArgMax: 3
- Gather: 3
- OneHot: 3
- Unsqueeze: 3
- Expand: 3
- Concat: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +0.312375
- cost 314: score 19.250607, delta +1.365376

## Examples
- train: 4 shown
- test: 1 shown
- arc-gen: 3 shown, 61 remaining

### train[1]
input 4x14 -> output 3x3

input:
```text
55550555505555
55550500500550
55550500500550
55550555505555
```

output:
```text
222
888
333
```

### train[2]
input 4x14 -> output 3x3

input:
```text
55550555505555
05500555505555
05500500505555
55550500505555
```

output:
```text
333
444
222
```

### train[3]
input 4x14 -> output 3x3

input:
```text
55550555505555
50050555505555
50050555505005
55550555505005
```

output:
```text
888
222
444
```

### train[4]
input 4x14 -> output 3x3

input:
```text
55550555505555
55550555505555
55550500505555
55550500505555
```

output:
```text
222
444
222
```

### test[1]
input 4x14 -> output 3x3

input:
```text
55550555505555
55550055005005
50050055005005
50050555505555
```

output:
```text
444
333
888
```

### arc-gen[1]
input 4x14 -> output 3x3

input:
```text
55550555505555
55550500500550
55550500500550
55550555505555
```

output:
```text
222
888
333
```

### arc-gen[2]
input 4x14 -> output 3x3

input:
```text
55550555505555
55550555505555
55550500505555
55550500505555
```

output:
```text
222
444
222
```

### arc-gen[3]
input 4x14 -> output 3x3

input:
```text
55550555505555
55550500505555
55550500505555
55550555505555
```

output:
```text
222
888
222
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 235 --onnx path/to/candidate.onnx
```
