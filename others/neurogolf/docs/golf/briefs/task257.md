# Task 257 Golf Brief

## Current Net
- path: `artifacts/optimized/task257.onnx`
- file size: 1369 bytes
- cost: 1077
- score: 18.018065
- memory: 1024
- params: 53
- nodes: 20
- value_info tensors after shape inference: 19
- local gold-correct: True

## Op Histogram

- Mul: 6
- Slice: 4
- Cast: 4
- Sub: 4
- Concat: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +0.179540
- cost 314: score 19.250607, delta +1.232542

## Examples
- train: 6 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 9x9 -> output 4x4

input:
```text
077710404
777014400
000010004
700010000
111111111
000016660
008810000
808016006
000810000
```

output:
```text
6777
7778
8084
7008
```

### train[2]
input 9x9 -> output 4x4

input:
```text
777010400
707014044
070714044
000710000
111111111
008016006
000016000
000016606
888016066
```

output:
```text
7776
7074
4747
8887
```

### train[3]
input 9x9 -> output 4x4

input:
```text
007710440
000710044
777710004
070010440
111111111
008810666
000010060
000816060
800016600
```

output:
```text
0477
0047
7777
8740
```

### train[4]
input 9x9 -> output 4x4

input:
```text
770014404
707014000
700714440
707714044
111111111
008010000
008016600
008010666
080810660
```

output:
```text
7784
7670
7447
7877
```

### train[5]
input 9x9 -> output 4x4

input:
```text
770010004
700014444
707014000
077014440
111111111
808016666
008810060
000010606
888810006
```

output:
```text
7784
7444
7676
4778
```

### train[6]
input 9x9 -> output 4x4

input:
```text
700714440
077714404
777014404
777010400
111111111
880816666
088810006
080810060
880810600
```

output:
```text
7447
4777
7774
7778
```

### test[1]
input 9x9 -> output 4x4

input:
```text
777010040
077014404
777710404
700014040
111111111
000810606
800816006
808016666
080810600
```

output:
```text
7778
4774
7777
7848
```

### arc-gen[1]
input 9x9 -> output 4x4

input:
```text
007710440
700710000
770014000
777014440
111111111
080810000
800016006
880816660
000010660
```

output:
```text
0477
7007
7768
7770
```

### arc-gen[2]
input 9x9 -> output 4x4

input:
```text
700014040
770010044
000010000
007010440
111111111
880016066
080810606
880816660
800010606
```

output:
```text
7846
7744
8868
8476
```

### arc-gen[3]
input 9x9 -> output 4x4

input:
```text
777710044
707710444
070714404
077010400
111111111
008816666
080810066
008810066
000016000
```

output:
```text
7777
7477
4787
6770
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 257 --onnx path/to/candidate.onnx
```
