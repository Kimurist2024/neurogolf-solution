# Task 249 Golf Brief

## Current Net
- path: `artifacts/optimized/task249.onnx`
- file size: 946 bytes
- cost: 792
- score: 18.325439
- memory: 758
- params: 34
- nodes: 9
- value_info tensors after shape inference: 8
- local gold-correct: True

## Op Histogram

- ReduceMax: 1
- ReduceSum: 1
- Reshape: 1
- Less: 1
- Sub: 1
- Where: 1
- Clip: 1
- Cast: 1
- Gather: 1

## Targets

- cost 900: score 18.197605, delta -0.127833
- cost 314: score 19.250607, delta +0.925168

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 258 remaining

### train[1]
input 3x3 -> output 3x6

input:
```text
050
552
000
```

output:
```text
050050
552552
000000
```

### train[2]
input 4x3 -> output 4x6

input:
```text
300
230
218
010
```

output:
```text
300300
230230
218218
010010
```

### train[3]
input 4x4 -> output 4x8

input:
```text
5230
2530
5288
0060
```

output:
```text
52305230
25302530
52885288
00600060
```

### test[1]
input 5x4 -> output 5x8

input:
```text
4000
4500
0560
6610
0001
```

output:
```text
40004000
45004500
05600560
66106610
00010001
```

### arc-gen[1]
input 3x5 -> output 3x10

input:
```text
06610
02420
04290
```

output:
```text
0661006610
0242002420
0429004290
```

### arc-gen[2]
input 4x3 -> output 4x6

input:
```text
000
880
000
000
```

output:
```text
000000
880880
000000
000000
```

### arc-gen[3]
input 4x3 -> output 4x6

input:
```text
700
006
557
007
```

output:
```text
700700
006006
557557
007007
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 249 --onnx path/to/candidate.onnx
```
