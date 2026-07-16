# Task 053 Golf Brief

## Current Net
- path: `artifacts/optimized/task053.onnx`
- file size: 175 bytes
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
- train: 4 shown
- test: 2 shown
- arc-gen: 3 shown, 51 remaining

### train[1]
input 3x3 -> output 3x3

input:
```text
111
000
000
```

output:
```text
000
111
000
```

### train[2]
input 3x3 -> output 3x3

input:
```text
000
111
000
```

output:
```text
000
000
111
```

### train[3]
input 3x3 -> output 3x3

input:
```text
010
110
000
```

output:
```text
000
010
110
```

### train[4]
input 3x3 -> output 3x3

input:
```text
022
002
000
```

output:
```text
000
022
002
```

### test[1]
input 3x3 -> output 3x3

input:
```text
200
200
000
```

output:
```text
000
200
200
```

### test[2]
input 3x3 -> output 3x3

input:
```text
000
010
000
```

output:
```text
000
000
010
```

### arc-gen[1]
input 3x3 -> output 3x3

input:
```text
001
010
000
```

output:
```text
000
001
010
```

### arc-gen[2]
input 3x3 -> output 3x3

input:
```text
100
100
000
```

output:
```text
000
100
100
```

### arc-gen[3]
input 3x3 -> output 3x3

input:
```text
222
000
000
```

output:
```text
000
222
000
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 53 --onnx path/to/candidate.onnx
```
