# Task 144 Golf Brief

## Current Net
- path: `artifacts/optimized/task144.onnx`
- file size: 1321 bytes
- cost: 430
- score: 18.936215
- memory: 384
- params: 46
- nodes: 8
- value_info tensors after shape inference: 7
- local gold-correct: True

## Op Histogram

- Slice: 2
- Cast: 2
- Mul: 1
- Sub: 1
- Concat: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta -0.738610
- cost 314: score 19.250607, delta +0.314392

## Examples
- train: 4 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 9x4 -> output 4x4

input:
```text
7707
0770
0777
0770
4444
0000
0202
2220
2002
```

output:
```text
0030
3000
0000
0000
```

### train[2]
input 9x4 -> output 4x4

input:
```text
0077
0077
0770
7700
4444
2020
0202
0220
0020
```

output:
```text
0300
3000
3003
0003
```

### train[3]
input 9x4 -> output 4x4

input:
```text
0007
0777
0700
0777
4444
0020
0222
2200
0202
```

output:
```text
3300
3000
0033
3000
```

### train[4]
input 9x4 -> output 4x4

input:
```text
7070
0077
7077
7700
4444
0022
0000
2002
0202
```

output:
```text
0300
3300
0300
0030
```

### test[1]
input 9x4 -> output 4x4

input:
```text
7777
0777
7000
7070
4444
0222
0000
2022
0200
```

output:
```text
0000
3000
0300
0003
```

### arc-gen[1]
input 9x4 -> output 4x4

input:
```text
0077
7007
7700
7770
4444
0220
0000
2000
2220
```

output:
```text
3000
0330
0033
0003
```

### arc-gen[2]
input 9x4 -> output 4x4

input:
```text
7000
7700
0000
0070
4444
2020
0022
0000
0220
```

output:
```text
0303
0000
3333
3003
```

### arc-gen[3]
input 9x4 -> output 4x4

input:
```text
7777
7077
0707
0770
4444
0022
0222
2202
0200
```

output:
```text
0000
0000
0030
3003
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 144 --onnx path/to/candidate.onnx
```
