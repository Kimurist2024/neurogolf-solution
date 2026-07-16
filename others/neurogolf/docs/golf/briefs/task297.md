# Task 297 Golf Brief

## Current Net
- path: `artifacts/optimized/task297.onnx`
- file size: 3233 bytes
- cost: 9171
- score: 15.876198
- memory: 9076
- params: 95
- nodes: 24
- value_info tensors after shape inference: 23
- local gold-correct: True

## Op Histogram

- Where: 4
- Slice: 3
- Cast: 3
- ReduceSum: 3
- Mul: 2
- Equal: 2
- And: 2
- Div: 1
- Floor: 1
- Sub: 1
- Sum: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +2.321407
- cost 314: score 19.250607, delta +3.374409

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 258 remaining

### train[1]
input 8x3 -> output 8x3

input:
```text
214
555
000
000
000
000
000
000
```

output:
```text
214
555
222
111
444
222
111
444
```

### train[2]
input 10x4 -> output 10x4

input:
```text
3214
5555
0000
0000
0000
0000
0000
0000
0000
0000
```

output:
```text
3214
5555
3333
2222
1111
4444
3333
2222
1111
4444
```

### train[3]
input 6x2 -> output 6x2

input:
```text
83
55
00
00
00
00
```

output:
```text
83
55
88
33
88
33
```

### test[1]
input 12x5 -> output 12x5

input:
```text
12348
55555
00000
00000
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
12348
55555
11111
22222
33333
44444
88888
11111
22222
33333
44444
88888
```

### arc-gen[1]
input 14x6 -> output 14x6

input:
```text
274916
555555
000000
000000
000000
000000
000000
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
274916
555555
222222
777777
444444
999999
111111
666666
222222
777777
444444
999999
111111
666666
```

### arc-gen[2]
input 6x2 -> output 6x2

input:
```text
76
55
00
00
00
00
```

output:
```text
76
55
77
66
77
66
```

### arc-gen[3]
input 6x2 -> output 6x2

input:
```text
91
55
00
00
00
00
```

output:
```text
91
55
99
11
99
11
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 297 --onnx path/to/candidate.onnx
```
