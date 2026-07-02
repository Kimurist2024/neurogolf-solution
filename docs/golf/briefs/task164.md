# Task 164 Golf Brief

## Current Net
- path: `artifacts/optimized/task164.onnx`
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
- train: 4 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 3x3 -> output 3x6

input:
```text
666
161
886
```

output:
```text
666666
161161
886688
```

### train[2]
input 3x3 -> output 3x6

input:
```text
681
611
116
```

output:
```text
681186
611116
116611
```

### train[3]
input 3x3 -> output 3x6

input:
```text
111
816
688
```

output:
```text
111111
816618
688886
```

### train[4]
input 3x3 -> output 3x6

input:
```text
111
166
666
```

output:
```text
111111
166661
666666
```

### test[1]
input 3x3 -> output 3x6

input:
```text
686
868
161
```

output:
```text
686686
868868
161161
```

### arc-gen[1]
input 3x3 -> output 3x6

input:
```text
818
618
166
```

output:
```text
818818
618816
166661
```

### arc-gen[2]
input 3x3 -> output 3x6

input:
```text
168
881
188
```

output:
```text
168861
881188
188881
```

### arc-gen[3]
input 3x3 -> output 3x6

input:
```text
161
186
681
```

output:
```text
161161
186681
681186
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 164 --onnx path/to/candidate.onnx
```
