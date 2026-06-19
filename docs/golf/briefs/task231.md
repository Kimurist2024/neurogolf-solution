# Task 231 Golf Brief

## Current Net
- path: `artifacts/optimized/task231.onnx`
- file size: 1978 bytes
- cost: 28296
- score: 14.749524
- memory: 28195
- params: 101
- nodes: 44
- value_info tensors after shape inference: 43
- local gold-correct: True

## Op Histogram

- Gather: 9
- Less: 5
- Sub: 5
- Cast: 4
- Abs: 4
- ReduceMax: 4
- Where: 3
- And: 2
- Mul: 2
- Conv: 1
- ReduceSum: 1
- Add: 1
- Not: 1
- Sum: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +3.448081
- cost 314: score 19.250607, delta +4.501083

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 5x6 -> output 5x12

input:
```text
000000
282828
282828
000000
000000
```

output:
```text
000000000000
282828282828
282828282828
000000000000
000000000000
```

### train[2]
input 5x7 -> output 5x14

input:
```text
0000000
0000000
2332332
0000000
0000000
```

output:
```text
00000000000000
00000000000000
23323323323323
00000000000000
00000000000000
```

### train[3]
input 5x8 -> output 5x16

input:
```text
00000000
00000000
12212212
21221221
00000000
```

output:
```text
0000000000000000
0000000000000000
1221221221221221
2122122122122122
0000000000000000
```

### test[1]
input 5x9 -> output 5x18

input:
```text
000000000
311311311
311311311
000000000
000000000
```

output:
```text
000000000000000000
311311311311311311
311311311311311311
000000000000000000
000000000000000000
```

### arc-gen[1]
input 5x10 -> output 5x20

input:
```text
0000000000
1191191191
0000000000
0000000000
0000000000
```

output:
```text
00000000000000000000
11911911911911911911
00000000000000000000
00000000000000000000
00000000000000000000
```

### arc-gen[2]
input 5x6 -> output 5x12

input:
```text
000000
000000
797979
000000
000000
```

output:
```text
000000000000
000000000000
797979797979
000000000000
000000000000
```

### arc-gen[3]
input 5x6 -> output 5x12

input:
```text
000000
000000
676767
000000
000000
```

output:
```text
000000000000
000000000000
676767676767
000000000000
000000000000
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 231 --onnx path/to/candidate.onnx
```
