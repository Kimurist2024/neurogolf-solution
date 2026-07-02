# Task 318 Golf Brief

## Current Net
- path: `artifacts/optimized/task318.onnx`
- file size: 708 bytes
- cost: 737
- score: 18.397412
- memory: 736
- params: 1
- nodes: 13
- value_info tensors after shape inference: 12
- local gold-correct: True

## Op Histogram

- Sub: 5
- Slice: 2
- Cast: 2
- Mul: 1
- Sum: 1
- Concat: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta -0.199807
- cost 314: score 19.250607, delta +0.853195

## Examples
- train: 4 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 9x4 -> output 4x4

input:
```text
1100
0101
0100
1010
4444
2222
0022
2200
0022
```

output:
```text
3333
0333
3300
3033
```

### train[2]
input 9x4 -> output 4x4

input:
```text
1110
0101
0011
1101
4444
0002
0002
2222
2202
```

output:
```text
3333
0303
3333
3303
```

### train[3]
input 9x4 -> output 4x4

input:
```text
1100
1010
1101
1111
4444
2202
0020
0200
2020
```

output:
```text
3303
3030
3303
3333
```

### train[4]
input 9x4 -> output 4x4

input:
```text
1010
1101
1011
0101
4444
2200
0020
2200
0020
```

output:
```text
3330
3333
3333
0333
```

### test[1]
input 9x4 -> output 4x4

input:
```text
1010
1010
0100
1010
4444
2200
0020
0202
2220
```

output:
```text
3330
3030
0303
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
0333
3003
3300
3330
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
3030
3333
0000
0330
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
3333
3333
3303
0330
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 318 --onnx path/to/candidate.onnx
```
