# Task 126 Golf Brief

## Current Net
- path: `artifacts/optimized/task126.onnx`
- file size: 1635 bytes
- cost: 22127
- score: 14.995446
- memory: 22032
- params: 95
- nodes: 17
- value_info tensors after shape inference: 16
- local gold-correct: True

## Op Histogram

- ReduceMax: 4
- Greater: 3
- Cast: 2
- Conv: 2
- And: 2
- Mul: 1
- Equal: 1
- Slice: 1
- Where: 1

## Targets

- cost 900: score 18.197605, delta +3.202159
- cost 314: score 19.250607, delta +4.255161

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 8x8 -> output 8x8

input:
```text
06660000
06060000
00000666
00000606
00000000
00000000
00000000
00000000
```

output:
```text
06660000
06060000
00000666
00000606
00000000
00000000
00000000
00400040
```

### train[2]
input 5x5 -> output 5x5

input:
```text
03330
03030
00000
00000
00000
```

output:
```text
03330
03030
00000
00000
00400
```

### train[3]
input 5x7 -> output 5x7

input:
```text
0000000
0888000
0808666
0000606
0000000
```

output:
```text
0000000
0888000
0808666
0000606
0040040
```

### test[1]
input 7x11 -> output 7x11

input:
```text
05550000000
05050888000
00000808333
00000000303
00000000000
00000000000
00000000000
```

output:
```text
05550000000
05050888000
00000808333
00000000303
00000000000
00000000000
00400040040
```

### arc-gen[1]
input 10x7 -> output 10x7

input:
```text
0000000
0000000
0999000
0909000
0000000
0000000
0000000
0000000
0000000
0000000
```

output:
```text
0000000
0000000
0999000
0909000
0000000
0000000
0000000
0000000
0000000
0040000
```

### arc-gen[2]
input 7x8 -> output 7x8

input:
```text
00000000
02220000
02020000
00000000
00000888
00000808
00000000
```

output:
```text
00000000
02220000
02020000
00000000
00000888
00000808
00400040
```

### arc-gen[3]
input 8x8 -> output 8x8

input:
```text
02220000
02020000
00000000
00000000
00000000
00000777
00000707
00000000
```

output:
```text
02220000
02020000
00000000
00000000
00000000
00000777
00000707
00400040
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 126 --onnx path/to/candidate.onnx
```
