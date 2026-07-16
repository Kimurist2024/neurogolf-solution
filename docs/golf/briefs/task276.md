# Task 276 Golf Brief

## Current Net
- path: `artifacts/optimized/task276.onnx`
- file size: 217 bytes
- cost: 10
- score: 22.697415
- memory: 0
- params: 10
- nodes: 1
- value_info tensors after shape inference: 0
- local gold-correct: True

## Op Histogram

- Gather: 1

## Targets

- cost 900: score 18.197605, delta -4.499810
- cost 314: score 19.250607, delta -3.446808

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 3x4 -> output 3x4

input:
```text
6676
6677
7767
```

output:
```text
2272
2277
7727
```

### train[2]
input 6x4 -> output 6x4

input:
```text
7776
6676
7767
7677
7676
6667
```

output:
```text
7772
2272
7727
7277
7272
2227
```

### train[3]
input 3x6 -> output 3x6

input:
```text
776666
676777
767767
```

output:
```text
772222
272777
727727
```

### test[1]
input 4x4 -> output 4x4

input:
```text
6776
6767
7776
7676
```

output:
```text
2772
2727
7772
7272
```

### arc-gen[1]
input 4x6 -> output 4x6

input:
```text
677766
777667
776677
666667
```

output:
```text
277722
777227
772277
222227
```

### arc-gen[2]
input 5x4 -> output 5x4

input:
```text
6667
7666
6666
6767
6766
```

output:
```text
2227
7222
2222
2727
2722
```

### arc-gen[3]
input 5x4 -> output 5x4

input:
```text
7666
6777
7667
7666
7676
```

output:
```text
7222
2777
7227
7222
7272
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 276 --onnx path/to/candidate.onnx
```
