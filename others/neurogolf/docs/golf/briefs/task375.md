# Task 375 Golf Brief

## Current Net
- path: `artifacts/optimized/task375.onnx`
- file size: 2267 bytes
- cost: 13184
- score: 15.513241
- memory: 13104
- params: 80
- nodes: 30
- value_info tensors after shape inference: 29
- local gold-correct: True

## Op Histogram

- ReduceSum: 7
- And: 4
- Greater: 3
- Equal: 3
- Cast: 2
- Mul: 2
- Sub: 2
- Abs: 2
- ArgMax: 1
- Reshape: 1
- Gather: 1
- Not: 1
- Where: 1

## Targets

- cost 900: score 18.197605, delta +2.684364
- cost 314: score 19.250607, delta +3.737366

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 51 remaining

### train[1]
input 3x3 -> output 3x3

input:
```text
111
101
111
```

output:
```text
010
101
010
```

### train[2]
input 5x5 -> output 5x5

input:
```text
22222
22222
22022
22222
22222
```

output:
```text
02220
20202
22022
20202
02220
```

### train[3]
input 7x7 -> output 7x7

input:
```text
3333333
3333333
3333333
3330333
3333333
3333333
3333333
```

output:
```text
0333330
3033303
3303033
3330333
3303033
3033303
0333330
```

### test[1]
input 11x11 -> output 11x11

input:
```text
66666666666
66666666666
66666666666
66666666666
66666666666
66666066666
66666666666
66666666666
66666666666
66666666666
66666666666
```

output:
```text
06666666660
60666666606
66066666066
66606660666
66660606666
66666066666
66660606666
66606660666
66066666066
60666666606
06666666660
```

### arc-gen[1]
input 13x13 -> output 13x13

input:
```text
2222222222222
2222222222222
2222222222222
2222222222222
2222222222222
2222222222222
2222220222222
2222222222222
2222222222222
2222222222222
2222222222222
2222222222222
2222222222222
```

output:
```text
0222222222220
2022222222202
2202222222022
2220222220222
2222022202222
2222202022222
2222220222222
2222202022222
2222022202222
2220222220222
2202222222022
2022222222202
0222222222220
```

### arc-gen[2]
input 5x5 -> output 5x5

input:
```text
66666
66666
66066
66666
66666
```

output:
```text
06660
60606
66066
60606
06660
```

### arc-gen[3]
input 5x5 -> output 5x5

input:
```text
88888
88888
88088
88888
88888
```

output:
```text
08880
80808
88088
80808
08880
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 375 --onnx path/to/candidate.onnx
```
