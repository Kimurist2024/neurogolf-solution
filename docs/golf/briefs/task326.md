# Task 326 Golf Brief

## Current Net
- path: `artifacts/optimized/task326.onnx`
- file size: 251 bytes
- cost: 160
- score: 19.924826
- memory: 160
- params: 0
- nodes: 2
- value_info tensors after shape inference: 1
- local gold-correct: True

## Op Histogram

- Slice: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta -1.727221
- cost 314: score 19.250607, delta -0.674219

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 6x6 -> output 2x2

input:
```text
436406
600334
644330
036046
063043
344660
```

output:
```text
43
60
```

### train[2]
input 8x8 -> output 2x2

input:
```text
24225245
25544222
45522224
22425425
24225245
25544222
45522224
22425425
```

output:
```text
24
25
```

### train[3]
input 12x6 -> output 2x2

input:
```text
321341
144223
133224
421431
412432
233114
244113
312342
321341
144223
133224
421431
```

output:
```text
32
14
```

### test[1]
input 4x8 -> output 2x2

input:
```text
96299269
29966992
69922996
92699629
```

output:
```text
96
29
```

### arc-gen[1]
input 4x8 -> output 2x2

input:
```text
56655556
56535335
55535366
63535355
```

output:
```text
56
56
```

### arc-gen[2]
input 4x8 -> output 2x2

input:
```text
66686899
69989896
86898888
86669689
```

output:
```text
66
69
```

### arc-gen[3]
input 4x8 -> output 2x2

input:
```text
65625662
25226656
26625526
65666566
```

output:
```text
65
25
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 326 --onnx path/to/candidate.onnx
```
