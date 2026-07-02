# Task 210 Golf Brief

## Current Net
- path: `artifacts/optimized/task210.onnx`
- file size: 479 bytes
- cost: 30
- score: 21.598803
- memory: 0
- params: 30
- nodes: 1
- value_info tensors after shape inference: 0
- local gold-correct: True

## Op Histogram

- Gather: 1

## Targets

- cost 900: score 18.197605, delta -3.401197
- cost 314: score 19.250607, delta -2.348196

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 3x3 -> output 6x3

input:
```text
110
111
000
```

output:
```text
110
111
000
000
111
110
```

### train[2]
input 3x3 -> output 6x3

input:
```text
000
101
110
```

output:
```text
000
101
110
110
101
000
```

### train[3]
input 3x3 -> output 6x3

input:
```text
000
001
001
```

output:
```text
000
001
001
001
001
000
```

### test[1]
input 3x3 -> output 6x3

input:
```text
000
001
100
```

output:
```text
000
001
100
100
001
000
```

### arc-gen[1]
input 3x3 -> output 6x3

input:
```text
001
110
011
```

output:
```text
001
110
011
011
110
001
```

### arc-gen[2]
input 3x3 -> output 6x3

input:
```text
100
011
000
```

output:
```text
100
011
000
000
011
100
```

### arc-gen[3]
input 3x3 -> output 6x3

input:
```text
111
110
110
```

output:
```text
111
110
110
110
110
111
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 210 --onnx path/to/candidate.onnx
```
