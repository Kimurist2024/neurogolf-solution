# Task 304 Golf Brief

## Current Net
- path: `artifacts/optimized/task304.onnx`
- file size: 884 bytes
- cost: 10812
- score: 15.711588
- memory: 10777
- params: 35
- nodes: 13
- value_info tensors after shape inference: 12
- local gold-correct: True

## Op Histogram

- ReduceSum: 3
- Slice: 1
- ReduceMax: 1
- Equal: 1
- Where: 1
- ConvTranspose: 1
- Tile: 1
- Mul: 1
- Sub: 1
- Concat: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +2.486017
- cost 314: score 19.250607, delta +3.539019

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 3x3 -> output 9x9

input:
```text
387
938
793
```

output:
```text
387000000
938000000
793000000
000387000
000938000
000793000
000000387
000000938
000000793
```

### train[2]
input 3x3 -> output 9x9

input:
```text
868
338
888
```

output:
```text
868000868
338000338
888000888
000000868
000000338
000000888
868868868
338338338
888888888
```

### train[3]
input 3x3 -> output 9x9

input:
```text
699
468
998
```

output:
```text
000699699
000468468
000998998
000000000
000000000
000000000
699699000
468468000
998998000
```

### test[1]
input 3x3 -> output 9x9

input:
```text
117
741
517
```

output:
```text
117117000
741741000
517517000
000000117
000000741
000000517
000117000
000741000
000517000
```

### arc-gen[1]
input 3x3 -> output 9x9

input:
```text
888
383
858
```

output:
```text
888888888
383383383
858858858
000888000
000383000
000858000
888000888
383000383
858000858
```

### arc-gen[2]
input 3x3 -> output 9x9

input:
```text
888
288
828
```

output:
```text
888888888
288288288
828828828
000888888
000288288
000828828
888000888
288000288
828000828
```

### arc-gen[3]
input 3x3 -> output 9x9

input:
```text
122
182
821
```

output:
```text
000122122
000182182
000821821
000000122
000000182
000000821
000122000
000182000
000821000
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 304 --onnx path/to/candidate.onnx
```
