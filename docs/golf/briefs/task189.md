# Task 189 Golf Brief

## Current Net
- path: `artifacts/optimized/task189.onnx`
- file size: 5411 bytes
- cost: 24583
- score: 14.890190
- memory: 24324
- params: 259
- nodes: 110
- value_info tensors after shape inference: 109
- local gold-correct: True

## Op Histogram

- Mul: 32
- Slice: 24
- Cast: 24
- Sum: 9
- Pad: 5
- ReduceSum: 4
- Greater: 4
- Sub: 4
- Where: 4

## Targets

- cost 900: score 18.197605, delta +3.307416
- cost 314: score 19.250607, delta +4.360417

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 9x9 -> output 6x6

input:
```text
248000000
168000000
888888888
008030030
008333333
008030030
008030030
008333333
008030030
```

output:
```text
020040
222444
020040
010060
111666
010060
```

### train[2]
input 9x9 -> output 6x6

input:
```text
000000812
000000841
888888888
003303800
330000800
330303800
000030800
333333800
000030800
```

output:
```text
001202
110000
110202
000010
444111
000010
```

### train[3]
input 9x9 -> output 6x6

input:
```text
008003003
008003003
008330330
008000030
008030300
008030003
888888888
248000000
658000000
```

output:
```text
002004
002004
220440
000050
060500
060005
```

### test[1]
input 9x9 -> output 6x6

input:
```text
000300800
330303800
030303800
033300800
030003800
003000800
888888888
000000821
000000847
```

output:
```text
000100
220101
020101
044700
040007
004000
```

### arc-gen[1]
input 9x9 -> output 6x6

input:
```text
303003800
330003800
000003800
300333800
003330800
033300800
888888888
000000826
000000854
```

output:
```text
202006
220006
000006
500444
005440
055400
```

### arc-gen[2]
input 9x9 -> output 6x6

input:
```text
003303800
300000800
330003800
030300800
000000800
330003800
888888888
000000871
000000846
```

output:
```text
007101
700000
770001
040600
000000
440006
```

### arc-gen[3]
input 9x9 -> output 6x6

input:
```text
000000816
000000854
888888888
033333800
303033800
000330800
333033800
303033800
330000800
```

output:
```text
011666
101066
000660
555044
505044
550000
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 189 --onnx path/to/candidate.onnx
```
