# Task 065 Golf Brief

## Current Net
- path: `artifacts/optimized/task065.onnx`
- file size: 1587 bytes
- cost: 28532
- score: 14.741218
- memory: 28476
- params: 56
- nodes: 35
- value_info tensors after shape inference: 34
- local gold-correct: True

## Op Histogram

- ReduceSum: 6
- Where: 5
- Greater: 4
- Cast: 3
- Add: 3
- Less: 2
- Mul: 2
- Pad: 2
- Gather: 2
- Slice: 1
- ReduceMax: 1
- Squeeze: 1
- Sub: 1
- Div: 1
- And: 1

## Targets

- cost 900: score 18.197605, delta +3.456387
- cost 314: score 19.250607, delta +4.509389

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 5x5 -> output 2x2

input:
```text
88388
88388
33333
88388
48388
```

output:
```text
88
48
```

### train[2]
input 7x7 -> output 3x3

input:
```text
4442444
4442414
4442444
2222222
4442444
4442444
4442444
```

output:
```text
444
414
444
```

### train[3]
input 11x11 -> output 5x5

input:
```text
33333133333
33333133333
38333133333
33333133333
33333133333
11111111111
33333133333
33333133333
33333133333
33333133333
33333133333
```

output:
```text
33333
33333
38333
33333
33333
```

### test[1]
input 13x13 -> output 6x6

input:
```text
1111110111111
1111110111111
1111110111111
1111110121111
1111110111111
1111110111111
0000000000000
1111110111111
1111110111111
1111110111111
1111110111111
1111110111111
1111110111111
```

output:
```text
111111
111111
111111
121111
111111
111111
```

### arc-gen[1]
input 11x11 -> output 5x5

input:
```text
33363533333
33333533333
33333533333
33333533333
33333533333
55555555555
33333533333
33333533333
33333533333
33333533333
33333533333
```

output:
```text
33363
33333
33333
33333
33333
```

### arc-gen[2]
input 3x3 -> output 1x1

input:
```text
798
999
797
```

output:
```text
8
```

### arc-gen[3]
input 3x3 -> output 1x1

input:
```text
962
666
969
```

output:
```text
2
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 65 --onnx path/to/candidate.onnx
```
