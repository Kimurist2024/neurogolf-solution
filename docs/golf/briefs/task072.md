# Task 072 Golf Brief

## Current Net
- path: `artifacts/optimized/task072.onnx`
- file size: 1289 bytes
- cost: 1309
- score: 17.822981
- memory: 1260
- params: 49
- nodes: 11
- value_info tensors after shape inference: 10
- local gold-correct: True

## Op Histogram

- Sub: 4
- Slice: 2
- Cast: 2
- Abs: 1
- Concat: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +0.374624
- cost 314: score 19.250607, delta +1.427626

## Examples
- train: 4 shown
- test: 2 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 13x5 -> output 6x5

input:
```text
00022
00202
20022
22002
00002
02000
44444
20000
22000
20200
00200
00022
20020
```

output:
```text
30033
33303
00333
33303
00030
33030
```

### train[2]
input 13x5 -> output 6x5

input:
```text
02222
00002
20222
00220
22220
22002
44444
00000
00200
20002
00020
02020
02220
```

output:
```text
03333
00303
00330
00300
30300
30333
```

### train[3]
input 13x5 -> output 6x5

input:
```text
22022
20222
20000
02020
22202
20200
44444
20022
00202
22000
00202
02022
02202
```

output:
```text
03000
30030
03000
03333
30330
33003
```

### train[4]
input 13x5 -> output 6x5

input:
```text
02020
22022
02220
02200
02222
20202
44444
20222
02200
20202
20002
22020
20220
```

output:
```text
33303
30333
33033
33303
30303
00033
```

### test[1]
input 13x5 -> output 6x5

input:
```text
20220
20022
22200
22222
02200
22222
44444
00022
20002
22202
02200
20220
20222
```

output:
```text
30303
00030
00003
30033
33030
03000
```

### test[2]
input 13x5 -> output 6x5

input:
```text
20202
20202
00020
02220
20220
22202
44444
22000
02222
00220
02000
02202
20000
```

output:
```text
03303
33030
00300
00330
33033
03303
```

### arc-gen[1]
input 13x5 -> output 6x5

input:
```text
00222
00222
00222
00220
00002
00022
44444
20020
22000
22020
00000
00200
22220
```

output:
```text
30303
33333
33303
00330
00303
33303
```

### arc-gen[2]
input 13x5 -> output 6x5

input:
```text
20002
20000
00002
02020
00220
00002
44444
20220
00202
22022
00020
22020
22220
```

output:
```text
00333
30303
33030
03000
33300
33333
```

### arc-gen[3]
input 13x5 -> output 6x5

input:
```text
22222
02202
02022
00022
02222
20202
44444
00002
20202
00220
00022
22002
20022
```

output:
```text
33330
33000
03303
00000
30330
00330
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 72 --onnx path/to/candidate.onnx
```
