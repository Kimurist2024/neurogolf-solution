# Task 289 Golf Brief

## Current Net
- path: `artifacts/optimized/task289.onnx`
- file size: 1331 bytes
- cost: 20512
- score: 15.071235
- memory: 20432
- params: 80
- nodes: 24
- value_info tensors after shape inference: 23
- local gold-correct: True

## Op Histogram

- Cast: 5
- Slice: 2
- ReduceSum: 2
- Where: 2
- Div: 2
- Floor: 2
- Less: 2
- Equal: 2
- MatMul: 2
- Greater: 1
- Mul: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +3.126371
- cost 314: score 19.250607, delta +4.179372

## Examples
- train: 5 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 3x3 -> output 6x6

input:
```text
670
066
000
```

output:
```text
667700
667700
006666
006666
000000
000000
```

### train[2]
input 3x3 -> output 6x6

input:
```text
104
040
010
```

output:
```text
110044
110044
004400
004400
001100
001100
```

### train[3]
input 3x3 -> output 9x9

input:
```text
320
073
000
```

output:
```text
333222000
333222000
333222000
000777333
000777333
000777333
000000000
000000000
000000000
```

### train[4]
input 3x3 -> output 9x9

input:
```text
080
066
980
```

output:
```text
000888000
000888000
000888000
000666666
000666666
000666666
999888000
999888000
999888000
```

### train[5]
input 3x3 -> output 12x12

input:
```text
403
220
008
```

output:
```text
444400003333
444400003333
444400003333
444400003333
222222220000
222222220000
222222220000
222222220000
000000008888
000000008888
000000008888
000000008888
```

### test[1]
input 3x3 -> output 12x12

input:
```text
010
087
990
```

output:
```text
000011110000
000011110000
000011110000
000011110000
000088887777
000088887777
000088887777
000088887777
999999990000
999999990000
999999990000
999999990000
```

### arc-gen[1]
input 3x3 -> output 9x9

input:
```text
470
040
041
```

output:
```text
444777000
444777000
444777000
000444000
000444000
000444000
000444111
000444111
000444111
```

### arc-gen[2]
input 3x3 -> output 9x9

input:
```text
070
009
080
```

output:
```text
000777000
000777000
000777000
000000999
000000999
000000999
000888000
000888000
000888000
```

### arc-gen[3]
input 3x3 -> output 9x9

input:
```text
970
000
060
```

output:
```text
999777000
999777000
999777000
000000000
000000000
000000000
000666000
000666000
000666000
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 289 --onnx path/to/candidate.onnx
```
