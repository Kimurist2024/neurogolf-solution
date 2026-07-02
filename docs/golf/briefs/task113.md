# Task 113 Golf Brief

## Current Net
- path: `artifacts/optimized/task113.onnx`
- file size: 499 bytes
- cost: 30
- score: 21.598803
- memory: 0
- params: 30
- nodes: 1
- value_info tensors after shape inference: 0
- local gold-correct: True

## Op Histogram

- Gather: 1

## Targets

- cost 900: score 18.197605, delta -3.401197
- cost 314: score 19.250607, delta -2.348196

## Examples
- train: 2 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 10x3 -> output 10x3

input:
```text
222
222
333
000
000
000
000
000
000
000
```

output:
```text
222
222
333
000
000
000
000
333
222
222
```

### train[2]
input 10x5 -> output 10x5

input:
```text
22222
88888
00000
00000
00000
00000
00000
00000
00000
00000
```

output:
```text
22222
88888
00000
00000
00000
00000
00000
00000
88888
22222
```

### test[1]
input 10x6 -> output 10x6

input:
```text
333333
555555
555555
000000
000000
000000
000000
000000
000000
000000
```

output:
```text
333333
555555
555555
000000
000000
000000
000000
555555
555555
333333
```

### arc-gen[1]
input 10x10 -> output 10x10

input:
```text
8888888888
3333333333
0000000000
0000000000
0000000000
0000000000
0000000000
0000000000
0000000000
0000000000
```

output:
```text
8888888888
3333333333
0000000000
0000000000
0000000000
0000000000
0000000000
0000000000
3333333333
8888888888
```

### arc-gen[2]
input 10x3 -> output 10x3

input:
```text
999
999
222
000
000
000
000
000
000
000
```

output:
```text
999
999
222
000
000
000
000
222
999
999
```

### arc-gen[3]
input 10x3 -> output 10x3

input:
```text
222
111
666
000
000
000
000
000
000
000
```

output:
```text
222
111
666
000
000
000
000
666
111
222
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 113 --onnx path/to/candidate.onnx
```
