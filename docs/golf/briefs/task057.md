# Task 057 Golf Brief

## Current Net
- path: `artifacts/optimized/task057.onnx`
- file size: 1422 bytes
- cost: 5177
- score: 16.448019
- memory: 5160
- params: 17
- nodes: 17
- value_info tensors after shape inference: 16
- local gold-correct: True

## Op Histogram

- ReduceSum: 2
- Less: 2
- Cast: 2
- ArgMax: 2
- Reshape: 2
- Add: 2
- Gather: 2
- Slice: 1
- Concat: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +1.749586
- cost 314: score 19.250607, delta +2.802588

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 258 remaining

### train[1]
input 8x8 -> output 3x6

input:
```text
00000000
08800000
00800000
08880000
00000000
00000000
00000000
00000000
```

output:
```text
880880
080080
888888
```

### train[2]
input 8x8 -> output 3x6

input:
```text
00000000
00000000
00000000
00000000
00000000
00020000
00222000
00220000
```

output:
```text
020020
222222
220220
```

### train[3]
input 8x8 -> output 3x6

input:
```text
00000000
00000110
00001000
00000100
00000000
00000000
00000000
00000000
```

output:
```text
011011
100100
010010
```

### test[1]
input 8x8 -> output 3x6

input:
```text
00000000
00000000
00000000
00000000
00030000
03330000
03000000
00000000
```

output:
```text
003003
333333
300300
```

### arc-gen[1]
input 8x8 -> output 3x6

input:
```text
40000000
40400000
04000000
00000000
00000000
00000000
00000000
00000000
```

output:
```text
400400
404404
040040
```

### arc-gen[2]
input 8x8 -> output 3x6

input:
```text
00000000
00000000
00000000
00000000
03330000
00300000
00330000
00000000
```

output:
```text
333333
030030
033033
```

### arc-gen[3]
input 8x8 -> output 3x6

input:
```text
00000000
01000000
00100000
11000000
00000000
00000000
00000000
00000000
```

output:
```text
010010
001001
110110
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 57 --onnx path/to/candidate.onnx
```
