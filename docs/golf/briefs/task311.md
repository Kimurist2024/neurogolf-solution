# Task 311 Golf Brief

## Current Net
- path: `artifacts/optimized/task311.onnx`
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
input 3x3 -> output 3x6

input:
```text
070
007
077
```

output:
```text
070070
007700
077770
```

### train[2]
input 3x3 -> output 3x6

input:
```text
000
077
000
```

output:
```text
000000
077770
000000
```

### train[3]
input 3x3 -> output 3x6

input:
```text
000
700
000
```

output:
```text
000000
700007
000000
```

### test[1]
input 3x3 -> output 3x6

input:
```text
770
070
007
```

output:
```text
770077
070070
007700
```

### arc-gen[1]
input 3x3 -> output 3x6

input:
```text
007
770
077
```

output:
```text
007700
770077
077770
```

### arc-gen[2]
input 3x3 -> output 3x6

input:
```text
700
077
000
```

output:
```text
700007
077770
000000
```

### arc-gen[3]
input 3x3 -> output 3x6

input:
```text
777
770
770
```

output:
```text
777777
770077
770077
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 311 --onnx path/to/candidate.onnx
```
