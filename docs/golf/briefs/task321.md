# Task 321 Golf Brief

## Current Net
- path: `artifacts/optimized/task321.onnx`
- file size: 2349 bytes
- cost: 5411
- score: 16.403811
- memory: 5376
- params: 35
- nodes: 22
- value_info tensors after shape inference: 21
- local gold-correct: True

## Op Histogram

- Conv: 6
- Mul: 4
- Slice: 3
- Cast: 3
- Sub: 3
- Pad: 2
- Sum: 1

## Targets

- cost 900: score 18.197605, delta +1.793794
- cost 314: score 19.250607, delta +2.846796

## Examples
- train: 5 shown
- test: 1 shown
- arc-gen: 3 shown, 258 remaining

### train[1]
input 4x14 -> output 4x4

input:
```text
04042990020000
04002009920100
40002000021110
44442909021101
```

output:
```text
9404
0499
4110
4444
```

### train[2]
input 4x14 -> output 4x4

input:
```text
44442909020001
44002990021000
40442000920101
00002009021010
```

output:
```text
4444
4400
4144
1090
```

### train[3]
input 4x14 -> output 4x4

input:
```text
44402990920101
04042009020100
04042009921001
40442999020001
```

output:
```text
4449
0494
1494
4944
```

### train[4]
input 4x14 -> output 4x4

input:
```text
00042000920000
44042909020000
40442099021101
04442090021111
```

output:
```text
0004
4494
4944
1444
```

### train[5]
input 4x14 -> output 4x4

input:
```text
40402000020001
44442000921100
04442099021101
04402009020101
```

output:
```text
4041
4444
1444
0441
```

### test[1]
input 4x14 -> output 4x4

input:
```text
00402909021100
44042999021110
00002099921101
04402909921000
```

output:
```text
9140
4494
1999
9449
```

### arc-gen[1]
input 4x14 -> output 4x4

input:
```text
00442099020101
40042000021000
44002900021101
44402999020000
```

output:
```text
0944
4004
4401
4440
```

### arc-gen[2]
input 4x14 -> output 4x4

input:
```text
40002909021100
44002009920101
00002000021101
00402099021000
```

output:
```text
4190
4499
1101
1940
```

### arc-gen[3]
input 4x14 -> output 4x4

input:
```text
44442009920011
40442099920101
04042990920011
04402090020000
```

output:
```text
4444
4944
9414
0440
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 321 --onnx path/to/candidate.onnx
```
