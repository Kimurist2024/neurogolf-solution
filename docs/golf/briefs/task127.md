# Task 127 Golf Brief

## Current Net
- path: `artifacts/optimized/task127.onnx`
- file size: 3772 bytes
- cost: 900
- score: 18.197605
- memory: 0
- params: 900
- nodes: 1
- value_info tensors after shape inference: 0
- local gold-correct: True

## Op Histogram

- Conv: 1

## Targets

- cost 900: score 18.197605, delta +0.000000
- cost 314: score 19.250607, delta +1.053002

## Examples
- train: 4 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 3x11 -> output 3x11

input:
```text
00050005000
01050205010
00050005000
```

output:
```text
66657775666
66657775666
66657775666
```

### train[2]
input 3x11 -> output 3x11

input:
```text
00050005000
02050305010
00050005000
```

output:
```text
77758885666
77758885666
77758885666
```

### train[3]
input 3x11 -> output 3x11

input:
```text
00050005000
03050105040
00050005000
```

output:
```text
88856665999
88856665999
88856665999
```

### train[4]
input 7x11 -> output 7x11

input:
```text
00050005000
04050105020
00050005000
55555555555
00050005000
02050305040
00050005000
```

output:
```text
99956665777
99956665777
99956665777
55555555555
77758885999
77758885999
77758885999
```

### test[1]
input 7x11 -> output 7x11

input:
```text
00050005000
02050305040
00050005000
55555555555
00050005000
01050105030
00050005000
```

output:
```text
77758885999
77758885999
77758885999
55555555555
66656665888
66656665888
66656665888
```

### arc-gen[1]
input 3x11 -> output 3x11

input:
```text
00050005000
04050205010
00050005000
```

output:
```text
99957775666
99957775666
99957775666
```

### arc-gen[2]
input 3x11 -> output 3x11

input:
```text
00050005000
03050105020
00050005000
```

output:
```text
88856665777
88856665777
88856665777
```

### arc-gen[3]
input 3x11 -> output 3x11

input:
```text
00050005000
04050105010
00050005000
```

output:
```text
99956665666
99956665666
99956665666
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 127 --onnx path/to/candidate.onnx
```
