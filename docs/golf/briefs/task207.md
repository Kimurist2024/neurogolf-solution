# Task 207 Golf Brief

## Current Net
- path: `artifacts/optimized/task207.onnx`
- file size: 1979 bytes
- cost: 1875
- score: 17.463636
- memory: 1840
- params: 35
- nodes: 25
- value_info tensors after shape inference: 24
- local gold-correct: True

## Op Histogram

- Slice: 4
- Cast: 4
- Where: 4
- ReduceSum: 4
- Concat: 2
- Sum: 1
- Greater: 1
- Neg: 1
- Reshape: 1
- ArgMax: 1
- Gather: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +0.733969
- cost 314: score 19.250607, delta +1.786971

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 258 remaining

### train[1]
input 5x5 -> output 2x2

input:
```text
02002
22022
00000
02022
22020
```

output:
```text
22
20
```

### train[2]
input 5x5 -> output 2x2

input:
```text
10010
01001
00000
10010
11001
```

output:
```text
10
11
```

### train[3]
input 5x5 -> output 2x2

input:
```text
88008
80080
00000
88088
80080
```

output:
```text
08
80
```

### test[1]
input 5x5 -> output 2x2

input:
```text
55050
05005
00000
55055
05005
```

output:
```text
50
05
```

### arc-gen[1]
input 5x5 -> output 2x2

input:
```text
20002
02020
00000
20020
02002
```

output:
```text
02
20
```

### arc-gen[2]
input 5x5 -> output 2x2

input:
```text
00000
99099
00000
00009
99090
```

output:
```text
09
90
```

### arc-gen[3]
input 5x5 -> output 2x2

input:
```text
70007
07070
00000
70070
07007
```

output:
```text
07
70
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 207 --onnx path/to/candidate.onnx
```
