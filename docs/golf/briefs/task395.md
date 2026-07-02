# Task 395 Golf Brief

## Current Net
- path: `artifacts/optimized/task395.onnx`
- file size: 738 bytes
- cost: 404
- score: 18.998585
- memory: 369
- params: 35
- nodes: 12
- value_info tensors after shape inference: 11
- local gold-correct: True

## Op Histogram

- Cast: 4
- Slice: 2
- Not: 2
- Or: 1
- Concat: 1
- Conv: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta -0.800980
- cost 314: score 19.250607, delta +0.252022

## Examples
- train: 5 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 6x3 -> output 3x3

input:
```text
099
099
999
010
001
111
```

output:
```text
200
200
000
```

### train[2]
input 6x3 -> output 3x3

input:
```text
909
099
009
101
100
100
```

output:
```text
020
000
020
```

### train[3]
input 6x3 -> output 3x3

input:
```text
090
909
900
000
001
100
```

output:
```text
202
020
022
```

### train[4]
input 6x3 -> output 3x3

input:
```text
009
999
090
100
011
001
```

output:
```text
020
000
200
```

### train[5]
input 6x3 -> output 3x3

input:
```text
090
099
099
000
111
101
```

output:
```text
202
000
000
```

### test[1]
input 6x3 -> output 3x3

input:
```text
909
009
909
011
010
100
```

output:
```text
000
200
020
```

### arc-gen[1]
input 6x3 -> output 3x3

input:
```text
009
990
099
100
111
001
```

output:
```text
020
000
200
```

### arc-gen[2]
input 6x3 -> output 3x3

input:
```text
900
099
000
000
001
010
```

output:
```text
022
200
202
```

### arc-gen[3]
input 6x3 -> output 3x3

input:
```text
999
990
990
101
011
000
```

output:
```text
000
000
002
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 395 --onnx path/to/candidate.onnx
```
