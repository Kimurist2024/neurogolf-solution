# Task 147 Golf Brief

## Current Net
- path: `artifacts/optimized/task147.onnx`
- file size: 3825 bytes
- cost: 910
- score: 18.186555
- memory: 0
- params: 910
- nodes: 1
- value_info tensors after shape inference: 0
- local gold-correct: True

## Op Histogram

- Conv: 1

## Targets

- cost 900: score 18.197605, delta +0.011050
- cost 314: score 19.250607, delta +1.064052

## Examples
- train: 4 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 3x3 -> output 3x3

input:
```text
330
030
303
```

output:
```text
880
080
303
```

### train[2]
input 4x6 -> output 4x6

input:
```text
030003
033300
000030
030000
```

output:
```text
080003
088800
000030
030000
```

### train[3]
input 4x4 -> output 4x4

input:
```text
3303
3300
3003
0033
```

output:
```text
8803
8800
8008
0088
```

### train[4]
input 5x6 -> output 5x6

input:
```text
330000
030030
300000
033000
033003
```

output:
```text
880000
080030
300000
088000
088003
```

### test[1]
input 5x5 -> output 5x5

input:
```text
30303
33300
00003
03300
03300
```

output:
```text
80803
88800
00003
08800
08800
```

### arc-gen[1]
input 6x3 -> output 6x3

input:
```text
003
333
333
330
030
033
```

output:
```text
008
888
888
880
080
088
```

### arc-gen[2]
input 5x3 -> output 5x3

input:
```text
000
330
000
000
030
```

output:
```text
000
880
000
000
030
```

### arc-gen[3]
input 6x3 -> output 6x3

input:
```text
300
003
333
003
300
030
```

output:
```text
300
008
888
008
300
030
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 147 --onnx path/to/candidate.onnx
```
