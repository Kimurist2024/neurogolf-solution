# Task 225 Golf Brief

## Current Net
- path: `artifacts/optimized/task225.onnx`
- file size: 5688 bytes
- cost: 11078
- score: 15.687284
- memory: 10428
- params: 650
- nodes: 40
- value_info tensors after shape inference: 39
- local gold-correct: True

## Op Histogram

- Mul: 10
- Cast: 9
- Slice: 6
- Pad: 5
- Conv: 4
- ReduceSum: 2
- Sum: 2
- Sub: 1
- Concat: 1

## Targets

- cost 900: score 18.197605, delta +2.510322
- cost 314: score 19.250607, delta +3.563323

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 258 remaining

### train[1]
input 6x6 -> output 6x6

input:
```text
000000
000000
009300
007800
000000
000000
```

output:
```text
880077
880077
009300
007800
330099
330099
```

### train[2]
input 6x6 -> output 6x6

input:
```text
000000
046000
021000
000000
000000
000000
```

output:
```text
100220
046000
021000
600440
600440
000000
```

### train[3]
input 6x6 -> output 6x6

input:
```text
000000
000000
003600
005200
000000
000000
```

output:
```text
220055
220055
003600
005200
660033
660033
```

### test[1]
input 6x6 -> output 6x6

input:
```text
000000
000000
000000
003100
002500
000000
```

output:
```text
000000
550022
550022
003100
002500
110033
```

### arc-gen[1]
input 6x6 -> output 6x6

input:
```text
000000
000000
000000
083000
051000
000000
```

output:
```text
000000
100550
100550
083000
051000
300880
```

### arc-gen[2]
input 6x6 -> output 6x6

input:
```text
000000
009200
008500
000000
000000
000000
```

output:
```text
550088
009200
008500
220099
220099
000000
```

### arc-gen[3]
input 6x6 -> output 6x6

input:
```text
000000
002100
006300
000000
000000
000000
```

output:
```text
330066
002100
006300
110022
110022
000000
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 225 --onnx path/to/candidate.onnx
```
