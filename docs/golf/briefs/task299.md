# Task 299 Golf Brief

## Current Net
- path: `artifacts/optimized/task299.onnx`
- file size: 1109 bytes
- cost: 1682
- score: 17.572261
- memory: 1620
- params: 62
- nodes: 21
- value_info tensors after shape inference: 20
- local gold-correct: True

## Op Histogram

- Cast: 8
- And: 3
- Slice: 2
- ReduceMax: 2
- Not: 2
- Or: 2
- Concat: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +0.625344
- cost 314: score 19.250607, delta +1.678346

## Examples
- train: 2 shown
- test: 1 shown
- arc-gen: 3 shown, 15 remaining

### train[1]
input 6x6 -> output 6x6

input:
```text
000080
000080
220000
000000
000000
000000
```

output:
```text
000080
000080
222242
000080
000080
000080
```

### train[2]
input 6x6 -> output 6x6

input:
```text
080000
080000
000000
000022
000000
000000
```

output:
```text
080000
080000
080000
242222
080000
080000
```

### test[1]
input 6x6 -> output 6x6

input:
```text
000800
000800
000000
000000
220000
000000
```

output:
```text
000800
000800
000800
000800
222422
000800
```

### arc-gen[1]
input 6x6 -> output 6x6

input:
```text
000800
000800
000000
000000
000022
000000
```

output:
```text
000800
000800
000800
000800
222422
000800
```

### arc-gen[2]
input 6x6 -> output 6x6

input:
```text
000800
000800
220000
000000
000000
000000
```

output:
```text
000800
000800
222422
000800
000800
000800
```

### arc-gen[3]
input 6x6 -> output 6x6

input:
```text
008000
008000
000000
220000
000000
000000
```

output:
```text
008000
008000
008000
224222
008000
008000
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 299 --onnx path/to/candidate.onnx
```
