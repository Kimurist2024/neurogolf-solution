# Task 019 Golf Brief

## Current Net
- path: `artifacts/optimized/task019.onnx`
- file size: 2709 bytes
- cost: 13649
- score: 15.478578
- memory: 13580
- params: 69
- nodes: 34
- value_info tensors after shape inference: 33
- local gold-correct: True

## Research Queue
- priority rank: 34
- recorded cost: 80850
- recorded memory: 80780
- recorded params: 70
- recorded nodes: 32

## Op Histogram

- Cast: 4
- ReduceSum: 4
- Slice: 3
- Add: 3
- ReduceMax: 2
- Mod: 2
- Less: 2
- Where: 2
- Gather: 2
- Sub: 2
- Greater: 2
- Conv: 1
- And: 1
- Concat: 1
- Squeeze: 1
- ScatterND: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +2.719027
- cost 314: score 19.250607, delta +3.772029

## Examples
- train: 4 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 2x4 -> output 4x8

input:
```text
0000
0500
```

output:
```text
80808080
05000500
80808080
05000500
```

### train[2]
input 3x4 -> output 6x8

input:
```text
0060
0000
0600
```

output:
```text
00600060
88888888
06080608
80608060
88888888
06000600
```

### train[3]
input 5x3 -> output 10x6

input:
```text
000
040
000
000
400
```

output:
```text
808808
040040
808808
088080
400400
888888
040040
808808
088080
400400
```

### train[4]
input 4x4 -> output 8x8

input:
```text
0000
0200
0000
0000
```

output:
```text
80808080
02000200
80808080
00000000
80808080
02000200
80808080
00000000
```

### test[1]
input 6x5 -> output 12x10

input:
```text
03000
00000
00000
00030
00000
03000
```

output:
```text
0300003000
8080080800
0080800808
0003000030
8080880808
8380083800
8380083800
8080080800
0080800808
0003000030
8080880808
0300003000
```

### arc-gen[1]
input 2x6 -> output 4x12

input:
```text
000000
000010
```

output:
```text
000808000808
000010000010
000808000808
000010000010
```

### arc-gen[2]
input 4x2 -> output 8x4

input:
```text
04
00
04
00
```

output:
```text
0404
8080
0404
8080
0404
8080
0404
8080
```

### arc-gen[3]
input 5x2 -> output 10x4

input:
```text
07
00
70
00
00
```

output:
```text
0707
8888
7070
0808
8080
0707
8888
7070
0808
0000
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 19 --onnx path/to/candidate.onnx
```
