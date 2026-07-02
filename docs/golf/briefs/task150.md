# Task 150 Golf Brief

## Current Net
- path: `artifacts/optimized/task150.onnx`
- file size: 618 bytes
- cost: 704
- score: 18.443222
- memory: 668
- params: 36
- nodes: 10
- value_info tensors after shape inference: 9
- local gold-correct: True

## Op Histogram

- ReduceSum: 2
- Sub: 2
- Greater: 1
- Cast: 1
- Less: 1
- Where: 1
- Squeeze: 1
- Gather: 1

## Targets

- cost 900: score 18.197605, delta -0.245616
- cost 314: score 19.250607, delta +0.807385

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 4x4 -> output 4x4

input:
```text
6662
6162
7272
1722
```

output:
```text
2666
2616
2727
2271
```

### train[2]
input 7x7 -> output 7x7

input:
```text
7776662
6711771
7721266
2277722
7271272
6662211
6266666
```

output:
```text
2666777
1771176
6621277
2277722
2721727
1122666
6666626
```

### train[3]
input 6x6 -> output 6x6

input:
```text
127111
217726
212621
121762
271271
216277
```

output:
```text
111721
627712
126212
267121
172172
772612
```

### test[1]
input 3x3 -> output 3x3

input:
```text
761
676
622
```

output:
```text
167
676
226
```

### arc-gen[1]
input 7x7 -> output 7x7

input:
```text
6726172
6776662
7676662
2627716
6226266
2167172
1777676
```

output:
```text
2716276
2666776
2666767
6177262
6626226
2717612
6767771
```

### arc-gen[2]
input 3x3 -> output 3x3

input:
```text
162
777
266
```

output:
```text
261
777
662
```

### arc-gen[3]
input 3x3 -> output 3x3

input:
```text
766
172
721
```

output:
```text
667
271
127
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 150 --onnx path/to/candidate.onnx
```
