# Task 130 Golf Brief

## Current Net
- path: `artifacts/optimized/task130.onnx`
- file size: 384 bytes
- cost: 3786
- score: 16.760935
- memory: 3762
- params: 24
- nodes: 5
- value_info tensors after shape inference: 4
- local gold-correct: True

## Op Histogram

- Slice: 1
- AveragePool: 1
- ArgMax: 1
- Equal: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +1.436671
- cost 314: score 19.250607, delta +2.489672

## Examples
- train: 2 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 9x9 -> output 3x3

input:
```text
333000888
333000858
333000888
000757000
000777000
000777000
666005999
666000999
656050995
```

output:
```text
308
070
609
```

### train[2]
input 9x9 -> output 3x3

input:
```text
000222000
050222000
000222000
500000000
000500050
000000000
050777000
000775000
000777000
```

output:
```text
020
000
070
```

### test[1]
input 9x9 -> output 3x3

input:
```text
444000050
544000000
444050000
000333050
000333000
000333000
005999000
000959000
000999000
```

output:
```text
400
030
090
```

### arc-gen[1]
input 9x9 -> output 3x3

input:
```text
000000353
000000333
000000333
000000500
000000000
000000000
000111000
500111000
000111500
```

output:
```text
003
000
010
```

### arc-gen[2]
input 9x9 -> output 3x3

input:
```text
000666000
000666000
000666000
000000444
000000444
000000444
000000500
000000000
000000000
```

output:
```text
060
004
000
```

### arc-gen[3]
input 9x9 -> output 3x3

input:
```text
000777000
000777005
500577000
000000050
000050000
000000000
000111000
050151000
000111500
```

output:
```text
070
000
010
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 130 --onnx path/to/candidate.onnx
```
