# Task 259 Golf Brief

## Current Net
- path: `artifacts/optimized/task259.onnx`
- file size: 3882 bytes
- cost: 6843
- score: 16.169018
- memory: 6793
- params: 50
- nodes: 30
- value_info tensors after shape inference: 29
- local gold-correct: True

## Op Histogram

- ArgMax: 4
- Where: 4
- Gather: 3
- Slice: 2
- ReduceSum: 2
- Greater: 2
- Cast: 2
- Sub: 2
- Add: 2
- LessOrEqual: 2
- Unsqueeze: 2
- And: 1
- Concat: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +2.028587
- cost 314: score 19.250607, delta +3.081589

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 5x7 -> output 3x3

input:
```text
1111111
1221111
1223111
1112111
1111111
```

output:
```text
220
223
002
```

### train[2]
input 7x7 -> output 2x3

input:
```text
1111111
1131211
1131211
1111111
1111111
1111111
1111111
```

output:
```text
302
302
```

### train[3]
input 7x6 -> output 3x2

input:
```text
111111
111111
155111
155111
166111
111111
111111
```

output:
```text
55
55
66
```

### test[1]
input 6x6 -> output 2x2

input:
```text
111111
111111
111211
112311
111111
111111
```

output:
```text
02
23
```

### arc-gen[1]
input 5x7 -> output 2x3

input:
```text
1181111
1111811
1111111
1111111
1111111
```

output:
```text
800
008
```

### arc-gen[2]
input 6x5 -> output 2x2

input:
```text
11111
11111
11111
11111
11121
11112
```

output:
```text
20
02
```

### arc-gen[3]
input 6x5 -> output 2x2

input:
```text
11111
11111
11147
11141
11111
11111
```

output:
```text
47
40
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 259 --onnx path/to/candidate.onnx
```
