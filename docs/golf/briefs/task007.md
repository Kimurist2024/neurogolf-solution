# Task 007 Golf Brief

## Current Net
- path: `artifacts/optimized/task007.onnx`
- file size: 1258 bytes
- cost: 7271
- score: 16.108351
- memory: 7240
- params: 31
- nodes: 13
- value_info tensors after shape inference: 12
- local gold-correct: True

## Op Histogram

- Reshape: 4
- Slice: 2
- ArgMax: 1
- Concat: 1
- ReduceMax: 1
- Tile: 1
- OneHot: 1
- Unsqueeze: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +2.089254
- cost 314: score 19.250607, delta +3.142256

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 7x7 -> output 7x7

input:
```text
2830000
8300000
3000000
0000000
0000000
0000000
0000000
```

output:
```text
2832832
8328328
3283283
2832832
8328328
3283283
2832832
```

### train[2]
input 7x7 -> output 7x7

input:
```text
0000000
0000000
0000001
0000012
0000124
0001240
0012400
```

output:
```text
2412412
4124124
1241241
2412412
4124124
1241241
2412412
```

### train[3]
input 7x7 -> output 7x7

input:
```text
0000830
0008300
0083000
0830004
8300040
3000400
0004000
```

output:
```text
4834834
8348348
3483483
4834834
8348348
3483483
4834834
```

### test[1]
input 7x7 -> output 7x7

input:
```text
0100002
1000020
0000200
0002000
0020000
0200004
2000040
```

output:
```text
2142142
1421421
4214214
2142142
1421421
4214214
2142142
```

### arc-gen[1]
input 7x7 -> output 7x7

input:
```text
0100000
1000000
0000000
0000000
0000000
0000009
0000093
```

output:
```text
3193193
1931931
9319319
3193193
1931931
9319319
3193193
```

### arc-gen[2]
input 7x7 -> output 7x7

input:
```text
4080000
0800007
8000070
0000700
0007000
0070000
0700000
```

output:
```text
4784784
7847847
8478478
4784784
7847847
8478478
4784784
```

### arc-gen[3]
input 7x7 -> output 7x7

input:
```text
1070000
0700000
7000000
0000000
0000006
0000060
0000600
```

output:
```text
1671671
6716716
7167167
1671671
6716716
7167167
1671671
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 7 --onnx path/to/candidate.onnx
```
