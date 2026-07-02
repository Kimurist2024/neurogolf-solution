# Task 227 Golf Brief

## Current Net
- path: `artifacts/optimized/task227.onnx`
- file size: 738 bytes
- cost: 691
- score: 18.461860
- memory: 656
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

- cost 900: score 18.197605, delta -0.264255
- cost 314: score 19.250607, delta +0.788747

## Examples
- train: 4 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 8x4 -> output 4x4

input:
```text
0000
0330
0000
3003
0001
1011
1111
0101
```

output:
```text
2220
0000
0000
0020
```

### train[2]
input 8x4 -> output 4x4

input:
```text
3333
0330
0033
3000
0001
0001
0100
1001
```

output:
```text
0000
2000
2000
0220
```

### train[3]
input 8x4 -> output 4x4

input:
```text
0330
0303
0030
3333
1111
1100
1100
0110
```

output:
```text
0000
0020
0002
0000
```

### train[4]
input 8x4 -> output 4x4

input:
```text
3333
3000
3033
3303
1110
0111
1011
0111
```

output:
```text
0000
0000
0200
0000
```

### test[1]
input 8x4 -> output 4x4

input:
```text
0303
3330
0003
3330
0011
0011
0100
1100
```

output:
```text
2000
0000
2020
0002
```

### arc-gen[1]
input 8x4 -> output 4x4

input:
```text
0033
3003
3300
3330
0110
0000
1000
1110
```

output:
```text
2000
0220
0022
0002
```

### arc-gen[2]
input 8x4 -> output 4x4

input:
```text
3000
3300
0000
0030
1010
0011
0000
0110
```

output:
```text
0202
0000
2222
2002
```

### arc-gen[3]
input 8x4 -> output 4x4

input:
```text
3333
3033
0303
0330
0011
0111
1101
0100
```

output:
```text
0000
0000
0020
2002
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 227 --onnx path/to/candidate.onnx
```
