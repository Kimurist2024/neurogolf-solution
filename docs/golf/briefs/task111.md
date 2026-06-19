# Task 111 Golf Brief

## Current Net
- path: `artifacts/optimized/task111.onnx`
- file size: 1499 bytes
- cost: 797
- score: 18.319145
- memory: 772
- params: 25
- nodes: 15
- value_info tensors after shape inference: 14
- local gold-correct: True

## Op Histogram

- Add: 3
- Slice: 2
- ReduceSum: 2
- ArgMax: 2
- Reshape: 2
- Concat: 2
- Sub: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta -0.121540
- cost 314: score 19.250607, delta +0.931462

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 258 remaining

### train[1]
input 10x10 -> output 3x3

input:
```text
0000000000
0000000011
0005000110
0001000010
0011100000
0001100000
0000000000
0000001100
0000011100
0000001100
```

output:
```text
010
111
011
```

### train[2]
input 10x10 -> output 3x3

input:
```text
0000000000
0000000500
0000004400
0040000040
0404000400
0044000000
0000000000
0000000000
0000000000
0000000000
```

output:
```text
440
004
040
```

### train[3]
input 10x10 -> output 3x3

input:
```text
0000000000
0000000000
0022000000
0202000000
0020000500
0000000220
0000002200
0000000200
0000000000
0000000000
```

output:
```text
022
220
020
```

### test[1]
input 10x10 -> output 3x3

input:
```text
0000005000
0000003000
0000033000
0000003300
0030000000
0330000000
0030003000
0000033300
0000003300
0000000000
```

output:
```text
030
330
033
```

### arc-gen[1]
input 10x10 -> output 3x3

input:
```text
0000000000
0000000000
0000000000
0000000000
0050000000
0444000000
0040000440
0404000044
0000000400
0000000000
```

output:
```text
444
040
404
```

### arc-gen[2]
input 10x10 -> output 3x3

input:
```text
0000005000
0000001100
0000010100
0000001000
0000000000
0000000000
0000000000
0000000110
0000000111
0000000001
```

output:
```text
011
101
010
```

### arc-gen[3]
input 10x10 -> output 3x3

input:
```text
0000000050
9990000099
9090000999
0900000900
0000000000
0000000000
0000099900
0000000900
0000099000
0000000000
```

output:
```text
099
999
900
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 111 --onnx path/to/candidate.onnx
```
