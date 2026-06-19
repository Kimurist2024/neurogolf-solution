# Task 298 Golf Brief

## Current Net
- path: `artifacts/optimized/task298.onnx`
- file size: 2632 bytes
- cost: 14282
- score: 15.433245
- memory: 13748
- params: 534
- nodes: 32
- value_info tensors after shape inference: 31
- local gold-correct: True

## Op Histogram

- Gather: 6
- Where: 6
- Sum: 4
- Mul: 3
- Cast: 2
- Conv: 2
- ReduceMax: 2
- Equal: 2
- Slice: 1
- ReduceSum: 1
- Sub: 1
- Concat: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +2.764361
- cost 314: score 19.250607, delta +3.817362

## Examples
- train: 3 shown
- test: 2 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 6x6 -> output 6x6

input:
```text
333333
322223
320023
320023
322223
333333
```

output:
```text
000000
033330
032230
032230
033330
000000
```

### train[2]
input 6x6 -> output 6x6

input:
```text
000000
077770
076670
076670
077770
000000
```

output:
```text
666666
600006
607706
607706
600006
666666
```

### train[3]
input 8x8 -> output 8x8

input:
```text
88888888
80000008
80555508
80588508
80588508
80555508
80000008
88888888
```

output:
```text
55555555
58888885
58000085
58055085
58055085
58000085
58888885
55555555
```

### test[1]
input 6x6 -> output 6x6

input:
```text
999999
900009
901109
901109
900009
999999
```

output:
```text
111111
199991
190091
190091
199991
111111
```

### test[2]
input 8x8 -> output 8x8

input:
```text
33333333
37777773
37666673
37633673
37633673
37666673
37777773
33333333
```

output:
```text
66666666
63333336
63777736
63766736
63766736
63777736
63333336
66666666
```

### arc-gen[1]
input 6x6 -> output 6x6

input:
```text
777777
722227
720027
720027
722227
777777
```

output:
```text
000000
077770
072270
072270
077770
000000
```

### arc-gen[2]
input 6x6 -> output 6x6

input:
```text
555555
588885
581185
581185
588885
555555
```

output:
```text
111111
155551
158851
158851
155551
111111
```

### arc-gen[3]
input 6x6 -> output 6x6

input:
```text
777777
711117
710017
710017
711117
777777
```

output:
```text
000000
077770
071170
071170
077770
000000
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 298 --onnx path/to/candidate.onnx
```
