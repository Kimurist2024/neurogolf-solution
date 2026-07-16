# Task 332 Golf Brief

## Current Net
- path: `artifacts/optimized/task332.onnx`
- file size: 2405 bytes
- cost: 7288
- score: 16.106016
- memory: 7221
- params: 67
- nodes: 29
- value_info tensors after shape inference: 28
- local gold-correct: True

## Op Histogram

- Cast: 5
- ReduceSum: 3
- Greater: 3
- Where: 3
- ReduceMax: 2
- Sub: 2
- And: 2
- Less: 2
- Slice: 1
- Conv: 1
- Mod: 1
- Equal: 1
- Squeeze: 1
- OneHot: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +2.091590
- cost 314: score 19.250607, delta +3.144591

## Examples
- train: 4 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 3x10 -> output 3x10

input:
```text
5050050005
0500500500
0005005050
```

output:
```text
5050030003
0300500300
0003005050
```

### train[2]
input 3x12 -> output 3x12

input:
```text
050500505000
500050050050
005005000505
```

output:
```text
030300505000
500050030050
005003000303
```

### train[3]
input 3x13 -> output 3x13

input:
```text
0050050500050
5000505005005
0505000050500
```

output:
```text
0030050500050
3000303005003
0505000030300
```

### train[4]
input 3x14 -> output 3x14

input:
```text
00500505050500
50005000505005
05050050000050
```

output:
```text
00500303030300
50005000505003
03030050000050
```

### test[1]
input 3x17 -> output 3x17

input:
```text
00050005050050500
50500500500500050
05005050005005005
```

output:
```text
00050005050030300
30300500300500050
05003030003005003
```

### arc-gen[1]
input 3x18 -> output 3x18

input:
```text
500505000550055550
005000550005500005
050050005000000000
```

output:
```text
500303000350035350
005000530003500003
030050005000000000
```

### arc-gen[2]
input 3x11 -> output 3x11

input:
```text
00005500000
50000000050
05550055505
```

output:
```text
00003500000
30000000050
05350035303
```

### arc-gen[3]
input 3x11 -> output 3x11

input:
```text
05500005050
50005500505
00050050000
```

output:
```text
05300005050
30003500303
00050030000
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 332 --onnx path/to/candidate.onnx
```
