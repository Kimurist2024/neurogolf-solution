# Task 146 Golf Brief

## Current Net
- path: `artifacts/optimized/task146.onnx`
- file size: 1894 bytes
- cost: 2256
- score: 17.278651
- memory: 2232
- params: 24
- nodes: 29
- value_info tensors after shape inference: 28
- local gold-correct: True

## Op Histogram

- Cast: 6
- Slice: 3
- ArgMax: 3
- Transpose: 3
- Equal: 3
- ReduceSum: 3
- Less: 3
- Where: 3
- Max: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +0.918954
- cost 314: score 19.250607, delta +1.971956

## Examples
- train: 4 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 9x3 -> output 3x3

input:
```text
898
988
888
221
221
112
444
443
333
```

output:
```text
444
443
333
```

### train[2]
input 9x3 -> output 3x3

input:
```text
155
511
511
333
363
366
777
722
722
```

output:
```text
333
363
366
```

### train[3]
input 9x3 -> output 3x3

input:
```text
222
223
233
577
755
755
881
181
181
```

output:
```text
881
181
181
```

### train[4]
input 9x3 -> output 3x3

input:
```text
884
444
448
113
133
331
622
222
226
```

output:
```text
884
444
448
```

### test[1]
input 9x3 -> output 3x3

input:
```text
544
454
454
332
332
223
111
188
188
```

output:
```text
544
454
454
```

### arc-gen[1]
input 9x3 -> output 3x3

input:
```text
822
282
222
595
999
599
133
311
111
```

output:
```text
133
311
111
```

### arc-gen[2]
input 9x3 -> output 3x3

input:
```text
226
666
262
855
555
555
949
444
949
```

output:
```text
226
666
262
```

### arc-gen[3]
input 9x3 -> output 3x3

input:
```text
828
282
282
616
161
616
344
434
443
```

output:
```text
828
282
282
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 146 --onnx path/to/candidate.onnx
```
