# Task 100 Golf Brief

## Current Net
- path: `artifacts/optimized/task100.onnx`
- file size: 1350 bytes
- cost: 6465
- score: 16.225842
- memory: 6421
- params: 44
- nodes: 27
- value_info tensors after shape inference: 26
- local gold-correct: True

## Op Histogram

- ArgMax: 5
- ReduceSum: 3
- Greater: 3
- Where: 3
- Add: 3
- Cast: 2
- Sub: 2
- Slice: 1
- Mul: 1
- OneHot: 1
- Unsqueeze: 1
- Expand: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +1.971764
- cost 314: score 19.250607, delta +3.024765

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 10x10 -> output 2x2

input:
```text
0777700000
0700700000
0700700000
0777700000
0000000000
0008888800
0008000800
0008000800
0008888800
0000000000
```

output:
```text
88
88
```

### train[2]
input 10x10 -> output 2x2

input:
```text
6666600000
6000600000
6000600000
6666600000
0000000000
0077777700
0070000700
0070000700
0077777700
0000000000
```

output:
```text
77
77
```

### train[3]
input 10x10 -> output 2x2

input:
```text
0444444000
0400004000
0400004000
0400004000
0400004000
0400004000
0444444000
0000000222
0000000202
0000000222
```

output:
```text
44
44
```

### test[1]
input 10x10 -> output 2x2

input:
```text
3333309999
3000309009
3000309009
3000309009
3000309009
3000309009
3000309009
3000309009
3333309009
0000009999
```

output:
```text
33
33
```

### arc-gen[1]
input 10x10 -> output 2x2

input:
```text
7777777777
7000000007
7000000007
7000000007
7000000007
7000000007
7777777777
0044444000
0040004000
0044444000
```

output:
```text
77
77
```

### arc-gen[2]
input 10x10 -> output 2x2

input:
```text
0000000000
0000000000
0000000000
0000000000
9990777770
9090700070
9090700070
9990700070
0000700070
0000777770
```

output:
```text
77
77
```

### arc-gen[3]
input 10x10 -> output 2x2

input:
```text
0000000000
0000000000
0000000000
0000000000
0000000000
7770333333
7070300003
7070333333
7770000000
0000000000
```

output:
```text
33
33
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 100 --onnx path/to/candidate.onnx
```
