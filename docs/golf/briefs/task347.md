# Task 347 Golf Brief

## Current Net
- path: `artifacts/optimized/task347.onnx`
- file size: 723 bytes
- cost: 395
- score: 19.021114
- memory: 360
- params: 35
- nodes: 11
- value_info tensors after shape inference: 10
- local gold-correct: True

## Op Histogram

- Cast: 4
- Slice: 2
- Or: 1
- Not: 1
- Concat: 1
- Conv: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta -0.823509
- cost 314: score 19.250607, delta +0.229493

## Examples
- train: 5 shown
- test: 2 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 3x6 -> output 3x3

input:
```text
440330
400300
000003
```

output:
```text
660
600
006
```

### train[2]
input 3x6 -> output 3x3

input:
```text
404330
400300
004300
```

output:
```text
666
600
606
```

### train[3]
input 3x6 -> output 3x3

input:
```text
004030
044303
440003
```

output:
```text
066
666
666
```

### train[4]
input 3x6 -> output 3x3

input:
```text
440300
000003
400000
```

output:
```text
660
006
600
```

### train[5]
input 3x6 -> output 3x3

input:
```text
000030
400000
004330
```

output:
```text
060
600
666
```

### test[1]
input 3x6 -> output 3x3

input:
```text
044300
400330
040300
```

output:
```text
666
660
660
```

### test[2]
input 3x6 -> output 3x3

input:
```text
004030
040333
400300
```

output:
```text
066
666
600
```

### arc-gen[1]
input 3x6 -> output 3x3

input:
```text
004300
440333
044003
```

output:
```text
606
666
066
```

### arc-gen[2]
input 3x6 -> output 3x3

input:
```text
400000
044003
000030
```

output:
```text
600
066
060
```

### arc-gen[3]
input 3x6 -> output 3x3

input:
```text
444303
440033
440000
```

output:
```text
666
666
660
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 347 --onnx path/to/candidate.onnx
```
