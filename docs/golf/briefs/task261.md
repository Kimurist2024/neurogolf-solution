# Task 261 Golf Brief

## Current Net
- path: `artifacts/optimized/task261.onnx`
- file size: 995 bytes
- cost: 200
- score: 19.701683
- memory: 0
- params: 200
- nodes: 1
- value_info tensors after shape inference: 0
- local gold-correct: True

## Op Histogram

- Conv: 1

## Targets

- cost 900: score 18.197605, delta -1.504077
- cost 314: score 19.250607, delta -0.451076

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 258 remaining

### train[1]
input 5x5 -> output 5x5

input:
```text
88000
88000
00000
00000
00000
```

output:
```text
00000
22000
22000
00000
00000
```

### train[2]
input 3x3 -> output 3x3

input:
```text
080
000
000
```

output:
```text
000
020
000
```

### train[3]
input 5x5 -> output 5x5

input:
```text
00000
08880
00000
00000
00000
```

output:
```text
00000
00000
02220
00000
00000
```

### test[1]
input 5x5 -> output 5x5

input:
```text
00800
08800
00800
00000
00000
```

output:
```text
00000
00200
02200
00200
00000
```

### arc-gen[1]
input 7x7 -> output 7x7

input:
```text
0000800
0000888
0000008
0000880
0000000
0000000
0000000
```

output:
```text
0000000
0000200
0000222
0000002
0000220
0000000
0000000
```

### arc-gen[2]
input 3x3 -> output 3x3

input:
```text
080
000
000
```

output:
```text
000
020
000
```

### arc-gen[3]
input 3x3 -> output 3x3

input:
```text
008
000
000
```

output:
```text
000
002
000
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 261 --onnx path/to/candidate.onnx
```
