# Task 048 Golf Brief

## Current Net
- path: `artifacts/optimized/task048.onnx`
- file size: 4064 bytes
- cost: 5349
- score: 16.415335
- memory: 5283
- params: 66
- nodes: 52
- value_info tensors after shape inference: 51
- local gold-correct: True

## Op Histogram

- Greater: 13
- Conv: 12
- Where: 12
- Cast: 4
- Slice: 2
- Reshape: 2
- Sum: 1
- ArgMax: 1
- OneHot: 1
- Mul: 1
- ReduceSum: 1
- Gather: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +1.782270
- cost 314: score 19.250607, delta +2.835272

## Examples
- train: 6 shown
- test: 2 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 5x5 -> output 1x1

input:
```text
00808
22800
22008
00022
88022
```

output:
```text
0
```

### train[2]
input 5x7 -> output 1x1

input:
```text
0800000
2208880
2288022
0080022
0800800
```

output:
```text
8
```

### train[3]
input 6x7 -> output 1x1

input:
```text
8228800
0220008
0880080
0080008
8088822
8000022
```

output:
```text
8
```

### train[4]
input 6x7 -> output 1x1

input:
```text
8800220
0880228
0008080
8000000
0220808
0228808
```

output:
```text
0
```

### train[5]
input 6x7 -> output 1x1

input:
```text
8000080
0022080
8022000
0080080
0082208
8002280
```

output:
```text
8
```

### train[6]
input 6x6 -> output 1x1

input:
```text
800228
808220
000080
228080
220008
088080
```

output:
```text
0
```

### test[1]
input 8x6 -> output 1x1

input:
```text
228808
220800
880008
088800
808008
008220
800220
080008
```

output:
```text
8
```

### test[2]
input 8x6 -> output 1x1

input:
```text
080000
000822
088822
080008
000800
822008
022000
080880
```

output:
```text
0
```

### arc-gen[1]
input 8x5 -> output 1x1

input:
```text
00888
88228
00220
08800
82288
82288
08000
08000
```

output:
```text
8
```

### arc-gen[2]
input 7x5 -> output 1x1

input:
```text
00088
00000
00080
80022
22822
22088
08000
```

output:
```text
8
```

### arc-gen[3]
input 8x5 -> output 1x1

input:
```text
80000
88822
08822
08080
00880
00228
08228
80088
```

output:
```text
8
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 48 --onnx path/to/candidate.onnx
```
