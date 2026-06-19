# Task 346 Golf Brief

## Current Net
- path: `artifacts/optimized/task346.onnx`
- file size: 1781 bytes
- cost: 9092
- score: 15.884850
- memory: 9069
- params: 23
- nodes: 13
- value_info tensors after shape inference: 12
- local gold-correct: True

## Op Histogram

- Add: 2
- Slice: 1
- AveragePool: 1
- ReduceMax: 1
- ReduceSum: 1
- Greater: 1
- Where: 1
- Sub: 1
- ArgMax: 1
- OneHot: 1
- Unsqueeze: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +2.312755
- cost 314: score 19.250607, delta +3.365757

## Examples
- train: 4 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 5x9 -> output 1x1

input:
```text
200002002
044400000
042400200
044400020
200002000
```

output:
```text
2
```

### train[2]
input 7x9 -> output 1x1

input:
```text
808000008
000080000
008003330
800303830
000003330
008000000
300800080
```

output:
```text
8
```

### train[3]
input 11x9 -> output 1x1

input:
```text
120002000
002000000
201202011
010020002
000000100
000000000
022200000
121200020
022200002
001000000
000200000
```

output:
```text
1
```

### train[4]
input 11x12 -> output 1x1

input:
```text
080000000038
300000080300
033800000008
000380000000
300000000080
000380000000
030000000000
000333008030
003383000000
000333000000
003000000000
```

output:
```text
8
```

### test[1]
input 12x12 -> output 1x1

input:
```text
000000000100
100000400100
000000000000
000014004000
040100000000
000001044001
100000000000
000100004000
000011100004
400014110000
000011100004
004400010000
```

output:
```text
4
```

### arc-gen[1]
input 12x6 -> output 1x1

input:
```text
000013
000003
003000
001113
101313
001110
000010
003000
000000
330000
300000
100001
```

output:
```text
3
```

### arc-gen[2]
input 10x6 -> output 1x1

input:
```text
000000
000099
022000
000000
002222
002920
202220
000000
000000
000002
```

output:
```text
9
```

### arc-gen[3]
input 12x6 -> output 1x1

input:
```text
000000
200002
000001
000000
000020
200020
001101
001110
201210
111110
000000
000000
```

output:
```text
2
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 346 --onnx path/to/candidate.onnx
```
