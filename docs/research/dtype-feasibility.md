# Dtype Feasibility (FP16 / BOOL / INT8) — ONNX Runtime 1.24.4 CPU, ORT_DISABLE_ALL

Snapshot: `artifacts/research_snapshot` (400 models). onnx 1.21.0, numpy 2.4.4.

## Opset histogram (domain '')

| opset | models |
|---|---|
| 9 | 7 |
| 10 | 101 |
| 11 | 33 |
| 12 | 33 |
| 13 | 86 |
| 14 | 2 |
| 16 | 43 |
| 17 | 86 |
| 18 | 8 |
| 20 | 1 |

## Op usage and dtype support

Status per dtype, aggregated over every opset version that the op actually appears at. `KERNEL_NOT_FOUND` = ORT raised NOT_IMPLEMENTED / 'Could not find an implementation' — dtype unusable for that op. `TYPE_INVALID` = ONNX type system rejects the dtype (not an ORT gap).

| op | models | opsets | float16 | bool | int8 | int32 |
|---|---|---|---|---|---|---|
| Abs | 22 | 9,10,11,12,13,16,17 | OK | — | OK | OK |
| Add | 114 | 10,11,12,13,16,17,18 | OK | — | MIXED(10:TYPE_INVALID,11:TYPE_INVALID,12:TYPE_INVALID,13:TYPE_INVALID,16:OK,17:OK,18:OK) | OK |
| And | 176 | 9,10,11,12,13,14,16,17,18,20 | — | OK | — | — |
| ArgMax | 114 | 9,10,11,12,13,16,17,18,20 | OK | — | OK | OK |
| ArgMin | 2 | 13,18 | OK | — | OK | OK |
| AveragePool | 4 | 11,12,13 | OK | — | — | — |
| Cast | 313 | 9,10,11,12,13,14,16,17,18,20 | — | — | — | — |
| Clip | 15 | 10,11,12,13,17,18 | OK | — | MIXED(10:TYPE_INVALID,11:TYPE_INVALID,12:OK,13:OK,17:OK,18:OK) | MIXED(10:TYPE_INVALID,11:TYPE_INVALID,12:OK,13:OK,17:OK,18:OK) |
| Concat | 184 | 9,10,11,12,13,14,16,17,18,20 | OK | OK | OK | — |
| Constant | 2 | 10,13 | OK | OK | OK | — |
| ConstantOfShape | 1 | 17 | OK | OK | OK | — |
| Conv | 171 | 9,10,11,12,13,14,16,17,18 | OK | — | — | — |
| ConvTranspose | 26 | 10,11,12,13,14,17,18 | OK | — | — | — |
| CumSum | 24 | 11,12,13,14,17,18 | MIXED(11:TYPE_INVALID,12:TYPE_INVALID,13:TYPE_INVALID,14:OK,17:OK,18:OK) | — | TYPE_INVALID | OK |
| Div | 43 | 10,11,12,13,16,17,18 | OK | — | MIXED(10:TYPE_INVALID,11:TYPE_INVALID,12:TYPE_INVALID,13:TYPE_INVALID,16:OK,17:OK,18:OK) | OK |
| Equal | 152 | 10,11,12,13,14,16,17,18,20 | MIXED(10:TYPE_INVALID,11:OK,12:OK,13:OK,14:OK,16:OK,17:OK,18:OK,20:OK) | OK | MIXED(10:TYPE_INVALID,11:OK,12:OK,13:OK,14:OK,16:OK,17:OK,18:OK,20:OK) | OK |
| Expand | 7 | 10,13,14,16,17 | OK | OK | OK | — |
| Flatten | 1 | 10 | OK | OK | OK | — |
| Floor | 24 | 10,11,13,16,18 | OK | — | — | — |
| Gather | 154 | 9,10,11,12,13,16,17,18,20 | OK | OK | OK | — |
| GatherElements | 5 | 12,16,17 | OK | OK | OK | — |
| GatherND | 2 | 11,17 | OK | OK | OK | — |
| GlobalMaxPool | 1 | 13 | OK | — | — | — |
| Greater | 203 | 9,10,11,12,13,14,16,17,18,20 | OK | — | OK | OK |
| GreaterOrEqual | 34 | 12,13,14,16,17,18 | OK | — | OK | OK |
| GridSample | 8 | 16 | OK | — | — | — |
| Identity | 3 | 10,14,18 | OK | OK | OK | — |
| Less | 114 | 10,11,12,13,14,16,17,18 | OK | — | OK | OK |
| LessOrEqual | 40 | 12,13,16,17,18 | OK | — | OK | OK |
| MatMul | 26 | 10,11,12,13,16,17 | OK | — | TYPE_INVALID | OK |
| MatMulInteger | 1 | 17 | — | — | OK | — |
| Max | 55 | 10,11,12,13,14,16,17,18 | OK | — | MIXED(10:TYPE_INVALID,11:TYPE_INVALID,12:OK,13:OK,14:OK,16:OK,17:OK,18:OK) | MIXED(10:TYPE_INVALID,11:TYPE_INVALID,12:OK,13:OK,14:OK,16:OK,17:OK,18:OK) |
| MaxPool | 39 | 10,11,12,13,14,16,17,18 | OK | — | MIXED(10:TYPE_INVALID,11:TYPE_INVALID,12:OK,13:OK,14:OK,16:OK,17:OK,18:OK) | — |
| Min | 35 | 10,11,12,13,14,16,17,18 | OK | — | MIXED(10:TYPE_INVALID,11:TYPE_INVALID,12:OK,13:OK,14:OK,16:OK,17:OK,18:OK) | MIXED(10:TYPE_INVALID,11:TYPE_INVALID,12:OK,13:OK,14:OK,16:OK,17:OK,18:OK) |
| Mod | 28 | 10,12,13,16,17,18 | OK | — | OK | OK |
| Mul | 227 | 9,10,11,12,13,14,16,17,18 | OK | — | MIXED(9:TYPE_INVALID,10:TYPE_INVALID,11:TYPE_INVALID,12:TYPE_INVALID,13:TYPE_INVALID,14:OK,16:OK,17:OK,18:OK) | OK |
| Neg | 17 | 10,11,12,14,16,17,18 | OK | — | OK | OK |
| Not | 105 | 9,10,11,12,13,14,16,17,18,20 | — | OK | — | — |
| OneHot | 29 | 10,11,12,13,16,17,18 | OK | KERNEL_NOT_FOUND | KERNEL_NOT_FOUND | — |
| Or | 88 | 9,10,11,12,13,16,17,20 | — | OK | — | — |
| Pad | 273 | 9,10,11,12,13,14,16,17,18,20 | OK | MIXED(9:TYPE_INVALID,10:TYPE_INVALID,11:TYPE_INVALID,12:TYPE_INVALID,13:OK,14:OK,16:OK,17:OK,18:OK,20:OK) | MIXED(9:TYPE_INVALID,10:TYPE_INVALID,11:OK,12:OK,13:OK,14:OK,16:OK,17:OK,18:OK,20:OK) | — |
| ReduceMax | 188 | 9,10,11,12,13,14,16,17,18,20 | OK | — | MIXED(9:TYPE_INVALID,10:TYPE_INVALID,11:TYPE_INVALID,12:OK,13:OK,14:OK,16:OK,17:OK,18:OK,20:OK) | OK |
| ReduceMin | 34 | 10,11,12,13,14,16,17 | OK | — | MIXED(10:TYPE_INVALID,11:TYPE_INVALID,12:OK,13:OK,14:OK,16:OK,17:OK) | OK |
| ReduceSum | 223 | 10,11,12,13,14,16,17,18 | OK | — | TYPE_INVALID | OK |
| Relu | 18 | 10,11,13,17,18 | OK | — | — | — |
| Reshape | 108 | 9,10,11,12,13,16,17,18 | OK | OK | OK | — |
| Resize | 3 | 10,13,17 | OK | KERNEL_NOT_FOUND | OK | — |
| ScatterElements | 3 | 11,16,17 | OK | OK | OK | — |
| ScatterND | 21 | 13,16,17 | OK | OK | OK | — |
| Sign | 6 | 10,11,13,17 | OK | — | OK | OK |
| Slice | 283 | 9,10,11,12,13,14,16,17,18,20 | OK | OK | OK | — |
| Split | 4 | 11,17,18 | OK | OK | OK | — |
| Sqrt | 2 | 10,17 | OK | — | — | — |
| Squeeze | 42 | 9,10,11,12,13,16,17,18 | OK | OK | OK | — |
| Sub | 236 | 9,10,11,12,13,14,16,17,18 | OK | — | MIXED(9:TYPE_INVALID,10:TYPE_INVALID,11:TYPE_INVALID,12:TYPE_INVALID,13:TYPE_INVALID,14:OK,16:OK,17:OK,18:OK) | OK |
| Sum | 109 | 9,10,11,12,13,14,16,17,18 | OK | — | TYPE_INVALID | TYPE_INVALID |
| Tile | 10 | 9,12,13,17 | OK | OK | OK | — |
| TopK | 20 | 12,13,16,17,18 | OK | — | KERNEL_NOT_FOUND | OK |
| Transpose | 22 | 10,11,12,13,17 | OK | OK | OK | — |
| Unsqueeze | 43 | 9,10,11,12,13,16,17,18 | OK | OK | OK | — |
| Where | 209 | 9,10,11,12,13,14,16,17,18 | OK | KERNEL_NOT_FOUND | KERNEL_NOT_FOUND | — |
| Xor | 3 | 13,17 | — | OK | — | — |

### Ops failing FP16 (KERNEL_NOT_FOUND)

- none

### Ops failing BOOL / INT8 / INT32 (KERNEL_NOT_FOUND)

- **OneHot**: bool, int8
- **Resize**: bool
- **TopK**: int8
- **Where**: bool, int8

## Cast support

| cast | result (per opset) |
|---|---|
| float32->float16 | OK |
| float16->float32 | OK |
| float32->bool | OK |
| bool->float32 | OK |
| float32->int8 | OK |

## End-to-end FP16 conversion sanity test

Converter: FLOAT initializers/Constant tensors -> FLOAT16, Cast(to=FLOAT) -> Cast(to=FLOAT16), boundary Cast after 'input' (float32 stays per competition IF), Cast back to float32 before 'output' only when the declared output dtype is FLOAT, Resize roi/scales kept float32, and no-op Cast(fp16->fp16) nodes stripped (leaving them in trips ORT's mandatory InsertCastTransformer with a type-mismatch error even at ORT_DISABLE_ALL).

Note: the snapshot is heterogeneous — declared output dtypes are float16 (142), float32 (141), bool (91), uint8 (25), int8 (1); many models are already fp16-internal.

```json
[
  {
    "model": "task005.onnx",
    "declared_output_dtype": "bool",
    "num_nodes": 1832,
    "runs": true,
    "output_shape": [
      1,
      10,
      30,
      30
    ],
    "max_abs_diff": 0.0,
    "masks_identical": true
  },
  {
    "model": "task279.onnx",
    "declared_output_dtype": "float32",
    "num_nodes": 52,
    "runs": true,
    "output_shape": [
      1,
      10,
      30,
      30
    ],
    "max_abs_diff": 0.0,
    "masks_identical": true
  }
]
```

## Caveats

- An fp16 'OK' means ORT executes the graph; for ops without a native fp16 CPU kernel ORT may internally wrap the node with InsertedPrecisionFreeCast (this transformer runs even at ORT_DISABLE_ALL). That affects runtime only — the scorer computes memory from the model proto via shape inference, so fp16 intermediates count 2 bytes per element regardless.
- TYPE_INVALID entries are ONNX type-system limits at that opset (e.g., Equal fp16/int8 needs opset>=11, CumSum fp16 needs opset>=14, Pad bool needs opset>=13, Pad int8 needs opset>=11, Clip int needs opset>=12, MatMul int8 is never valid — use MatMulInteger). Bumping a model's opset import lifts these.
