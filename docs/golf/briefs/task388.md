# Task 388 Golf Brief

## Current Net
- path: `artifacts/optimized/task388.onnx`
- file size: 6536 bytes
- cost: 6468
- score: 16.225378
- memory: 6282
- params: 186
- nodes: 52
- value_info tensors after shape inference: 51
- local gold-correct: True

## Op Histogram

- Pad: 19
- Where: 13
- Equal: 6
- Cast: 3
- Add: 3
- GreaterOrEqual: 2
- Slice: 1
- Conv: 1
- ReduceMax: 1
- Greater: 1
- And: 1
- ReduceSum: 1

## Targets

- cost 900: score 18.197605, delta +1.972227
- cost 314: score 19.250607, delta +3.025229

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 3x3 -> output 6x6

input:
```text
200
000
002
```

output:
```text
208208
808808
802802
208208
808808
802802
```

### train[2]
input 6x6 -> output 12x12

input:
```text
050000
000000
000000
000000
500005
000000
```

output:
```text
850008850008
880008880008
880008880008
880008880008
580005580005
880008880008
850008850008
880008880008
880008880008
880008880008
580005580005
880008880008
```

### train[3]
input 2x2 -> output 4x4

input:
```text
04
00
```

output:
```text
0404
0808
0404
0808
```

### test[1]
input 4x4 -> output 8x8

input:
```text
0030
0000
0003
3000
```

output:
```text
80388038
80888088
80838083
30883088
80388038
80888088
80838083
30883088
```

### arc-gen[1]
input 6x6 -> output 12x12

input:
```text
000000
000000
000000
000000
000000
300000
```

output:
```text
800000800000
800000800000
800000800000
800000800000
800000800000
300000300000
800000800000
800000800000
800000800000
800000800000
800000800000
300000300000
```

### arc-gen[2]
input 2x2 -> output 4x4

input:
```text
70
07
```

output:
```text
7878
8787
7878
8787
```

### arc-gen[3]
input 2x2 -> output 4x4

input:
```text
60
06
```

output:
```text
6868
8686
6868
8686
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 388 --onnx path/to/candidate.onnx
```
