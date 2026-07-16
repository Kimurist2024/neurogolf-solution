# Task 268 Golf Brief

## Current Net
- path: `artifacts/optimized/task268.onnx`
- file size: 4399 bytes
- cost: 18665
- score: 15.165595
- memory: 18608
- params: 57
- nodes: 153
- value_info tensors after shape inference: 152
- local gold-correct: True

## Op Histogram

- And: 38
- Greater: 16
- Sub: 14
- Where: 13
- Equal: 12
- Or: 12
- Add: 7
- ReduceMin: 6
- ReduceMax: 6
- GreaterOrEqual: 6
- LessOrEqual: 6
- Cast: 5
- ReduceSum: 5
- Slice: 2
- Not: 2
- Less: 2
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +3.032011
- cost 314: score 19.250607, delta +4.085012

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 7x7 -> output 7x7

input:
```text
0000000
0000000
0000000
0000000
0660660
0600060
0666660
```

output:
```text
0004000
4004004
0404040
0044400
0664660
0644460
0666660
```

### train[2]
input 9x9 -> output 9x9

input:
```text
000000000
000000000
000077777
000070007
000000007
000000007
000000007
000070007
000077777
```

output:
```text
400000000
040000000
004077777
000474447
444444447
444444447
444444447
000474447
004077777
```

### train[3]
input 6x6 -> output 6x6

input:
```text
333333
300003
300003
330033
000000
000000
```

output:
```text
333333
344443
344443
334433
044440
404404
```

### test[1]
input 10x10 -> output 10x10

input:
```text
0222200000
0200200000
0200000000
0200000000
0200000000
0200000000
0200000000
0200200000
0222200000
0000000000
```

output:
```text
0222204000
0244240000
0244444444
0244444444
0244444444
0244444444
0244444444
0244240000
0222204000
0000000400
```

### arc-gen[1]
input 9x9 -> output 9x9

input:
```text
000000000
000000000
000000000
000000000
111111000
100001000
100000000
100001000
111111000
```

output:
```text
000000000
000000000
000000000
000000004
111111040
144441400
144444444
144441400
111111040
```

### arc-gen[2]
input 5x5 -> output 5x5

input:
```text
99900
90900
90000
90900
99900
```

output:
```text
99904
94940
94444
94940
99904
```

### arc-gen[3]
input 5x5 -> output 5x5

input:
```text
00777
00707
00007
00707
00777
```

output:
```text
40777
04747
44447
04747
40777
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 268 --onnx path/to/candidate.onnx
```
