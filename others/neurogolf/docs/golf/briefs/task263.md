# Task 263 Golf Brief

## Current Net
- path: `artifacts/optimized/task263.onnx`
- file size: 6815 bytes
- cost: 7233
- score: 16.113591
- memory: 7190
- params: 43
- nodes: 146
- value_info tensors after shape inference: 145
- local gold-correct: True

## Op Histogram

- Cast: 24
- Mul: 20
- Sub: 19
- Slice: 18
- ReduceSum: 18
- Less: 15
- Where: 14
- Greater: 6
- Sum: 6
- And: 5
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +2.084014
- cost 314: score 19.250607, delta +3.137016

## Examples
- train: 4 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 9x3 -> output 3x3

input:
```text
606
066
606
404
044
404
888
808
888
```

output:
```text
888
808
888
```

### train[2]
input 3x12 -> output 3x3

input:
```text
200300707100
200300070100
022033707011
```

output:
```text
707
070
707
```

### train[3]
input 3x15 -> output 3x3

input:
```text
300404200800100
033444022088011
030404020080010
```

output:
```text
404
444
404
```

### train[4]
input 12x3 -> output 3x3

input:
```text
077
770
707
300
033
300
200
022
200
800
088
800
```

output:
```text
077
770
707
```

### test[1]
input 15x3 -> output 3x3

input:
```text
050
505
050
030
303
030
606
660
606
040
404
040
080
808
080
```

output:
```text
606
660
606
```

### arc-gen[1]
input 15x3 -> output 3x3

input:
```text
220
002
202
880
008
808
990
009
909
550
005
505
010
100
011
```

output:
```text
010
100
011
```

### arc-gen[2]
input 3x9 -> output 3x3

input:
```text
660220880
600200008
006002808
```

output:
```text
880
008
808
```

### arc-gen[3]
input 3x9 -> output 3x3

input:
```text
880200110
008002001
880020110
```

output:
```text
200
002
020
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 263 --onnx path/to/candidate.onnx
```
