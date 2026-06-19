# Task 180 Golf Brief

## Current Net
- path: `artifacts/optimized/task180.onnx`
- file size: 2525 bytes
- cost: 7450
- score: 16.084031
- memory: 6528
- params: 922
- nodes: 6
- value_info tensors after shape inference: 5
- local gold-correct: True

## Op Histogram

- Slice: 1
- Cast: 1
- Conv: 1
- ReduceMax: 1
- Mul: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +2.113575
- cost 314: score 19.250607, delta +3.166576

## Examples
- train: 5 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 8x8 -> output 4x4

input:
```text
44000050
44000000
00400050
04005500
00600090
66600009
60669900
06609000
```

output:
```text
4450
6669
6956
5560
```

### train[2]
input 8x8 -> output 4x4

input:
```text
40045500
00000055
44040500
40440555
00060909
00600900
60060909
00660009
```

output:
```text
5506
0955
6506
4555
```

### train[3]
input 8x8 -> output 4x4

input:
```text
00045000
40000500
00040050
04040050
60000990
60000909
60609990
60600000
```

output:
```text
5994
6509
6954
6454
```

### train[4]
input 8x8 -> output 4x4

input:
```text
40040505
00405005
00440055
40005005
66609099
66600999
60069909
66069099
```

output:
```text
6565
5665
6955
5695
```

### train[5]
input 8x8 -> output 4x4

input:
```text
04440555
00405505
00005000
40005000
66060099
00069090
00069099
66060909
```

output:
```text
6555
5595
5096
5606
```

### test[1]
input 8x8 -> output 4x4

input:
```text
04045000
04445055
44400555
00005000
60669990
00060900
06000099
60000900
```

output:
```text
5966
5955
4555
5900
```

### arc-gen[1]
input 8x8 -> output 4x4

input:
```text
00440550
40040000
44005000
44405550
06060000
60009009
66069990
00000990
```

output:
```text
0556
6009
5696
5550
```

### arc-gen[2]
input 8x8 -> output 4x4

input:
```text
40005050
44000055
00000000
00400550
66009099
06060909
66069990
60000909
```

output:
```text
5659
4655
6696
6559
```

### arc-gen[3]
input 8x8 -> output 4x4

input:
```text
44440055
40440555
04045505
04400500
00669999
06060099
00660099
00009000
```

output:
```text
9955
4555
5565
9540
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 180 --onnx path/to/candidate.onnx
```
