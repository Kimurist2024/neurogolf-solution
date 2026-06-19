# Task 183 Golf Brief

## Current Net
- path: `artifacts/optimized/task183.onnx`
- file size: 2043 bytes
- cost: 19342
- score: 15.129966
- memory: 19304
- params: 38
- nodes: 55
- value_info tensors after shape inference: 54
- local gold-correct: True

## Op Histogram

- Cast: 12
- Gather: 12
- Mul: 11
- ReduceSum: 6
- Sub: 3
- Where: 3
- Less: 2
- Add: 2
- Unsqueeze: 2
- Sum: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +3.067639
- cost 314: score 19.250607, delta +4.120641

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 258 remaining

### train[1]
input 8x8 -> output 4x4

input:
```text
21000013
11111111
01080010
01880810
01008010
01808810
11111111
41000016
```

output:
```text
0200
2203
0060
4066
```

### train[2]
input 6x6 -> output 2x2

input:
```text
910014
111111
018810
018010
111111
210013
```

output:
```text
94
20
```

### train[3]
input 8x8 -> output 4x4

input:
```text
61000012
11111111
01080810
01888010
01808810
01888010
11111111
71000014
```

output:
```text
0602
6620
7044
7740
```

### test[1]
input 10x10 -> output 6x6

input:
```text
3100000014
1111111111
0108800010
0188808010
0100808010
0108088010
0188080810
0108008010
1111111111
7100000015
```

output:
```text
033000
333040
003040
070550
770505
070050
```

### arc-gen[1]
input 10x10 -> output 6x6

input:
```text
5100000016
1111111111
0180008810
0188888810
0180080010
0188008810
0108888010
0188080010
1111111111
3100000014
```

output:
```text
500066
555666
500600
330044
033440
330400
```

### arc-gen[2]
input 6x6 -> output 2x2

input:
```text
310016
111111
018010
010010
111111
710015
```

output:
```text
30
00
```

### arc-gen[3]
input 6x6 -> output 2x2

input:
```text
710013
111111
010810
010010
111111
510019
```

output:
```text
03
00
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 183 --onnx path/to/candidate.onnx
```
