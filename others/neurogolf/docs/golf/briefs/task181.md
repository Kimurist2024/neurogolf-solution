# Task 181 Golf Brief

## Current Net
- path: `artifacts/optimized/task181.onnx`
- file size: 3083 bytes
- cost: 1194
- score: 17.914936
- memory: 1108
- params: 86
- nodes: 21
- value_info tensors after shape inference: 20
- local gold-correct: True

## Op Histogram

- Slice: 5
- Greater: 4
- Pad: 3
- And: 2
- Or: 2
- Where: 2
- Gather: 1
- Concat: 1
- Equal: 1

## Targets

- cost 900: score 18.197605, delta +0.282670
- cost 314: score 19.250607, delta +1.335671

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 6x9 -> output 6x9

input:
```text
000808000
000088000
000008000
000400000
000444000
000040000
```

output:
```text
808808000
880088000
800008000
000400000
000444000
000040000
```

### train[2]
input 6x9 -> output 6x9

input:
```text
000808000
000888000
000880000
000004000
000444000
000040000
```

output:
```text
000808808
000888888
000880088
000004000
000444000
000040000
```

### train[3]
input 6x9 -> output 6x9

input:
```text
000800000
000088000
000800000
000400000
000444000
000040000
```

output:
```text
008800000
880088000
008800000
000400000
000444000
000040000
```

### test[1]
input 6x9 -> output 6x9

input:
```text
000808000
000088000
000800000
000004000
000444000
000040000
```

output:
```text
000808808
000088880
000800008
000004000
000444000
000040000
```

### arc-gen[1]
input 6x9 -> output 6x9

input:
```text
000800000
000808000
000080000
000400000
000444000
000040000
```

output:
```text
008800000
808808000
080080000
000400000
000444000
000040000
```

### arc-gen[2]
input 6x9 -> output 6x9

input:
```text
000800000
000008000
000080000
000400000
000444000
000040000
```

output:
```text
008800000
800008000
080080000
000400000
000444000
000040000
```

### arc-gen[3]
input 6x9 -> output 6x9

input:
```text
000800000
000088000
000800000
000004000
000444000
000040000
```

output:
```text
000800008
000088880
000800008
000004000
000444000
000040000
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 181 --onnx path/to/candidate.onnx
```
