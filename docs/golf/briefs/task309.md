# Task 309 Golf Brief

## Current Net
- path: `artifacts/optimized/task309.onnx`
- file size: 217 bytes
- cost: 10
- score: 22.697415
- memory: 0
- params: 10
- nodes: 1
- value_info tensors after shape inference: 0
- local gold-correct: True

## Op Histogram

- Gather: 1

## Targets

- cost 900: score 18.197605, delta -4.499810
- cost 314: score 19.250607, delta -3.446808

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 258 remaining

### train[1]
input 3x6 -> output 3x6

input:
```text
188778
117718
711778
```

output:
```text
188558
115518
511558
```

### train[2]
input 3x4 -> output 3x4

input:
```text
7771
1817
7117
```

output:
```text
5551
1815
5115
```

### train[3]
input 3x5 -> output 3x5

input:
```text
18171
78811
71887
```

output:
```text
18151
58811
51885
```

### test[1]
input 3x5 -> output 3x5

input:
```text
17717
81777
87178
```

output:
```text
15515
81555
85158
```

### arc-gen[1]
input 3x6 -> output 3x6

input:
```text
187181
778117
711117
```

output:
```text
185181
558115
511115
```

### arc-gen[2]
input 3x4 -> output 3x4

input:
```text
7888
1188
8788
```

output:
```text
5888
1188
8588
```

### arc-gen[3]
input 3x4 -> output 3x4

input:
```text
7118
7781
7178
```

output:
```text
5118
5581
5158
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 309 --onnx path/to/candidate.onnx
```
