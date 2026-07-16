# Task 292 Golf Brief

## Current Net
- path: `artifacts/optimized/task292.onnx`
- file size: 1065 bytes
- cost: 6305
- score: 16.250902
- memory: 6240
- params: 65
- nodes: 7
- value_info tensors after shape inference: 6
- local gold-correct: True

## Op Histogram

- Conv: 2
- Slice: 1
- Cast: 1
- Mul: 1
- Sum: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +1.946703
- cost 314: score 19.250607, delta +2.999705

## Examples
- train: 5 shown
- test: 1 shown
- arc-gen: 3 shown, 19 remaining

### train[1]
input 3x10 -> output 3x10

input:
```text
4040404040
4444444444
0404040404
```

output:
```text
6040406040
6446446446
0406040406
```

### train[2]
input 3x11 -> output 3x11

input:
```text
04040404040
44444444444
40404040404
```

output:
```text
04060404060
64464464464
60404060404
```

### train[3]
input 3x11 -> output 3x11

input:
```text
40404040404
44444444444
04040404040
```

output:
```text
60404060404
64464464464
04060404060
```

### train[4]
input 3x13 -> output 3x13

input:
```text
4040404040404
4444444444444
0404040404040
```

output:
```text
6040406040406
6446446446446
0406040406040
```

### train[5]
input 3x14 -> output 3x14

input:
```text
04040404040404
44444444444444
40404040404040
```

output:
```text
04060404060404
64464464464464
60404060404060
```

### test[1]
input 3x17 -> output 3x17

input:
```text
04040404040404040
44444444444444444
40404040404040404
```

output:
```text
04060404060404060
64464464464464464
60404060404060404
```

### arc-gen[1]
input 3x18 -> output 3x18

input:
```text
404040404040404040
444444444444444444
040404040404040404
```

output:
```text
604040604040604040
644644644644644644
040604040604040604
```

### arc-gen[2]
input 3x11 -> output 3x11

input:
```text
04040404040
44444444444
40404040404
```

output:
```text
04060404060
64464464464
60404060404
```

### arc-gen[3]
input 3x15 -> output 3x15

input:
```text
404040404040404
444444444444444
040404040404040
```

output:
```text
604040604040604
644644644644644
040604040604040
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 292 --onnx path/to/candidate.onnx
```
