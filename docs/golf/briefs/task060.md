# Task 060 Golf Brief

## Current Net
- path: `artifacts/optimized/task060.onnx`
- file size: 2458 bytes
- cost: 5581
- score: 16.372877
- memory: 5400
- params: 181
- nodes: 20
- value_info tensors after shape inference: 19
- local gold-correct: True

## Op Histogram

- Mul: 7
- Slice: 4
- Cast: 4
- Sub: 3
- Sum: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +1.824728
- cost 314: score 19.250607, delta +2.877730

## Examples
- train: 2 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 5x11 -> output 5x11

input:
```text
00000000000
10000000002
00000000000
00000000000
00000000000
```

output:
```text
00000000000
11111522222
00000000000
00000000000
00000000000
```

### train[2]
input 5x11 -> output 5x11

input:
```text
00000000000
00000000000
00000000000
30000000007
00000000000
```

output:
```text
00000000000
00000000000
00000000000
33333577777
00000000000
```

### test[1]
input 5x11 -> output 5x11

input:
```text
00000000000
40000000008
00000000000
00000000000
60000000009
```

output:
```text
00000000000
44444588888
00000000000
00000000000
66666599999
```

### arc-gen[1]
input 5x11 -> output 5x11

input:
```text
00000000000
00000000000
00000000000
30000000006
00000000000
```

output:
```text
00000000000
00000000000
00000000000
33333566666
00000000000
```

### arc-gen[2]
input 5x11 -> output 5x11

input:
```text
00000000000
00000000000
20000000009
00000000000
00000000000
```

output:
```text
00000000000
00000000000
22222599999
00000000000
00000000000
```

### arc-gen[3]
input 5x11 -> output 5x11

input:
```text
00000000000
00000000000
00000000000
20000000001
00000000000
```

output:
```text
00000000000
00000000000
00000000000
22222511111
00000000000
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 60 --onnx path/to/candidate.onnx
```
