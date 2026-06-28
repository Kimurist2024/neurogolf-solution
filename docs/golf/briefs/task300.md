# Task 300 Golf Brief

## Current Net
- path: `artifacts/optimized/task300.onnx`
- file size: 2258 bytes
- cost: 4673
- score: 16.550443
- memory: 4648
- params: 25
- nodes: 25
- value_info tensors after shape inference: 24
- local gold-correct: True

## Research Queue
- priority rank: 41
- recorded cost: 77546
- recorded memory: 77438
- recorded params: 108
- recorded nodes: 43

## Op Histogram

- ArgMax: 5
- Reshape: 4
- Sub: 4
- Add: 3
- Concat: 3
- ReduceMax: 2
- ReduceSum: 1
- Gather: 1
- Slice: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +1.647162
- cost 314: score 19.250607, delta +2.700164

## Examples
- train: 4 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 7x13 -> output 4x3

input:
```text
0000000000000
0220033000000
0020003001000
0022000011000
0222000011100
0000000000000
0000000000000
```

output:
```text
220
020
022
222
```

### train[2]
input 5x10 -> output 3x2

input:
```text
0000000660
0300440060
3330440000
0300440000
0000000000
```

output:
```text
44
44
44
```

### train[3]
input 6x11 -> output 4x3

input:
```text
00000000000
08880000770
00800020070
08800220070
08800020070
00000000000
```

output:
```text
888
080
880
880
```

### train[4]
input 7x9 -> output 3x3

input:
```text
000000000
000700222
000770020
000070222
888000000
080000000
000000000
```

output:
```text
222
020
222
```

### test[1]
input 9x9 -> output 4x3

input:
```text
000000000
400000000
440333000
040333000
000303000
000303000
000000066
055500666
055000660
```

output:
```text
333
333
303
303
```

### arc-gen[1]
input 8x10 -> output 3x3

input:
```text
0000000000
0000005000
0000005500
0888000000
0888006660
0800006600
0000000000
0000000000
```

output:
```text
888
888
800
```

### arc-gen[2]
input 7x12 -> output 3x3

input:
```text
000055000000
000055000000
000055509990
000000009990
000077000990
000007000000
000000000000
```

output:
```text
999
999
099
```

### arc-gen[3]
input 7x13 -> output 3x3

input:
```text
0000022200000
0000020200000
0000000000000
0000000000000
0000000000666
0000700000666
0000777000066
```

output:
```text
666
666
066
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 300 --onnx path/to/candidate.onnx
```
