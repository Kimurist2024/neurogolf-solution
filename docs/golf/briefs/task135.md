# Task 135 Golf Brief

## Current Net
- path: `artifacts/optimized/task135.onnx`
- file size: 251 bytes
- cost: 360
- score: 19.113896
- memory: 360
- params: 0
- nodes: 2
- value_info tensors after shape inference: 1
- local gold-correct: True

## Op Histogram

- Slice: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta -0.916291
- cost 314: score 19.250607, delta +0.136711

## Examples
- train: 4 shown
- test: 1 shown
- arc-gen: 3 shown, 258 remaining

### train[1]
input 9x9 -> output 3x3

input:
```text
300700970
840660484
170000400
110910700
000077000
800170840
070992100
000000500
000240800
```

output:
```text
970
484
400
```

### train[2]
input 9x9 -> output 3x3

input:
```text
900000060
040705081
020071445
060040000
830420097
002302067
404034707
710000300
320040000
```

output:
```text
060
081
445
```

### train[3]
input 9x9 -> output 3x3

input:
```text
250060000
255700601
030001940
070600000
090001008
000000000
004000000
000100004
050000000
```

output:
```text
000
601
940
```

### train[4]
input 9x9 -> output 3x3

input:
```text
050080004
000000300
000021003
010000300
100100000
000000080
000000000
009400000
307002006
```

output:
```text
004
300
003
```

### test[1]
input 9x9 -> output 3x3

input:
```text
690010589
290608090
000009920
926008068
774070900
007001574
410075009
990000100
492000840
```

output:
```text
589
090
920
```

### arc-gen[1]
input 9x9 -> output 3x3

input:
```text
028090600
000010400
700002401
408090000
420023962
800369898
000104009
000065000
350530099
```

output:
```text
600
400
401
```

### arc-gen[2]
input 9x9 -> output 3x3

input:
```text
209900079
804122508
180475067
026200508
000600000
980098005
080900319
008004008
900759980
```

output:
```text
079
508
067
```

### arc-gen[3]
input 9x9 -> output 3x3

input:
```text
202100370
030006520
570436607
003060041
085025770
000290020
001059200
230013000
095900007
```

output:
```text
370
520
607
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 135 --onnx path/to/candidate.onnx
```
