# Task 236 Golf Brief

## Current Net
- path: `artifacts/optimized/task236.onnx`
- file size: 623 bytes
- cost: 641
- score: 18.536971
- memory: 640
- params: 1
- nodes: 10
- value_info tensors after shape inference: 9
- local gold-correct: True

## Op Histogram

- Sub: 3
- Slice: 2
- Cast: 2
- Abs: 1
- Concat: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta -0.339365
- cost 314: score 19.250607, delta +0.713636

## Examples
- train: 4 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 9x4 -> output 4x4

input:
```text
0101
0001
1010
0001
4444
0202
0002
2002
2220
```

output:
```text
0000
0000
0033
3333
```

### train[2]
input 9x4 -> output 4x4

input:
```text
1100
1010
1101
0110
4444
0222
2020
2222
2222
```

output:
```text
3033
0000
0030
3003
```

### train[3]
input 9x4 -> output 4x4

input:
```text
0100
1011
1110
1110
4444
0000
0202
2202
0200
```

output:
```text
0300
3330
0033
3030
```

### train[4]
input 9x4 -> output 4x4

input:
```text
1011
0001
1100
0011
4444
0222
0222
2022
2222
```

output:
```text
3300
0330
0333
3300
```

### test[1]
input 9x4 -> output 4x4

input:
```text
1011
0111
0010
1011
4444
2202
0020
2002
0202
```

output:
```text
0330
0303
3033
3330
```

### arc-gen[1]
input 9x4 -> output 4x4

input:
```text
0011
1001
1100
1110
4444
0220
0000
2000
2220
```

output:
```text
0303
3003
0300
0000
```

### arc-gen[2]
input 9x4 -> output 4x4

input:
```text
1000
1100
0000
0010
4444
2020
0022
0000
0220
```

output:
```text
0030
3333
0000
0300
```

### arc-gen[3]
input 9x4 -> output 4x4

input:
```text
1111
1011
0101
0110
4444
0022
0222
2202
0200
```

output:
```text
3300
3300
3000
0030
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 236 --onnx path/to/candidate.onnx
```
