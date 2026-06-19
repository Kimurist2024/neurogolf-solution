# Task 214 Golf Brief

## Current Net
- path: `artifacts/optimized/task214.onnx`
- file size: 679 bytes
- cost: 1395
- score: 17.759350
- memory: 1320
- params: 75
- nodes: 2
- value_info tensors after shape inference: 1
- local gold-correct: True

## Op Histogram

- GridSample: 1
- Pad: 1

## Targets

- cost 900: score 18.197605, delta +0.438255
- cost 314: score 19.250607, delta +1.491257

## Examples
- train: 3 shown
- test: 1 shown
- arc-gen: 3 shown, 259 remaining

### train[1]
input 3x11 -> output 3x11

input:
```text
11250005000
41150005000
44150005000
```

output:
```text
11254415144
41154115114
44151125211
```

### train[2]
input 3x11 -> output 3x11

input:
```text
63350005000
63350005000
63250005000
```

output:
```text
63356665236
63353335336
63252335336
```

### train[3]
input 3x11 -> output 3x11

input:
```text
27850005000
77850005000
88850005000
```

output:
```text
27858725888
77858775877
88858885872
```

### test[1]
input 3x11 -> output 3x11

input:
```text
33950005000
99950005000
29950005000
```

output:
```text
33952935992
99959935999
29959995933
```

### arc-gen[1]
input 3x11 -> output 3x11

input:
```text
24250005000
77450005000
22750005000
```

output:
```text
24252725722
77452745477
22757425242
```

### arc-gen[2]
input 3x11 -> output 3x11

input:
```text
66250005000
26650005000
63650005000
```

output:
```text
66256265636
26653665662
63656625266
```

### arc-gen[3]
input 3x11 -> output 3x11

input:
```text
21450005000
41250005000
42450005000
```

output:
```text
21454425424
41252115214
42454245412
```

## Verification
```bash
.venv/bin/python scripts/golf/try_candidate.py --task 214 --onnx path/to/candidate.onnx
```
