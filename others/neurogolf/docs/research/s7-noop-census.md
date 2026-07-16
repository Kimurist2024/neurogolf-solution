# S7 No-op Node Elimination Census

Snapshot: `artifacts/research_snapshot/` — 400 models, 400 analyzed, 0 errors.

Scoring: `score = max(1, 25 - ln(max(1, memory + params)))`. 
Memory here is the STATIC inferred-shape memory (the official scorer additionally maxes with the ORT profiler trace; shapes are fixed for these models so the static value is the operative one).

## Headline numbers

- Total no-op nodes detected: **157** across **19/400** tasks
- Total removable intermediate bytes: **14,674**
- Total removable params (orphaned constants): **10**
- Estimated total score gain: **+0.1089** (sum over tasks)
- Graph-output rewire cases (flagged): 0 (0 bytes)

## No-ops by category

| Category | Count | Attributed bytes |
|---|---:|---:|
| Identity | 8 | 11,522 |
| Cast_same_dtype | 0 | 0 |
| AddSub_zero | 65 | 982 |
| MulDiv_one | 67 | 1,670 |
| ShapeOp_same_shape | 17 | 500 |
| Transpose_identity | 0 | 0 |
| Pad_zero | 0 | 0 |
| Clip_unbounded | 0 | 0 |
| Concat_single | 0 | 0 |

Attributed bytes = each no-op's own output tensor (producer tensor for graph-output cases). The authoritative total above comes from full graph simulation (chains resolved, orphaned constants dropped), so the per-category column can differ slightly from the total.

## Global op_type histogram (top 30, all 400 models)

| op_type | count |
|---|---:|
| Cast | 3646 |
| Mul | 3313 |
| And | 2867 |
| Add | 2228 |
| Sub | 1736 |
| Where | 1618 |
| Gather | 1349 |
| Equal | 1335 |
| Greater | 1311 |
| Slice | 1117 |
| ReduceSum | 951 |
| ReduceMax | 853 |
| Max | 839 |
| Reshape | 828 |
| Min | 776 |
| Conv | 705 |
| Less | 642 |
| GreaterOrEqual | 639 |
| LessOrEqual | 609 |
| Pad | 563 |
| Sum | 502 |
| Or | 487 |
| ArgMax | 486 |
| Concat | 425 |
| Not | 360 |
| MaxPool | 314 |
| Squeeze | 258 |
| Unsqueeze | 194 |
| ReduceMin | 188 |
| Clip | 187 |

## Top 20 tasks by estimated score gain

| Task | No-ops | Removable bytes | Removable params | Score before | Score after | Delta |
|---|---:|---:|---:|---:|---:|---:|
| task096 | 4 | 9,680 | 0 | 12.9705 | 13.0300 | +0.0595 |
| task285 | 63 | 1,278 | 0 | 13.2741 | 13.2845 | +0.0104 |
| task185 | 1 | 360 | 3 | 14.5218 | 14.5321 | +0.0103 |
| task173 | 14 | 672 | 0 | 13.5465 | 13.5537 | +0.0072 |
| task255 | 2 | 1,802 | 0 | 12.2002 | 12.2052 | +0.0050 |
| task013 | 2 | 120 | 0 | 14.7603 | 14.7646 | +0.0043 |
| task034 | 4 | 240 | 0 | 13.9896 | 13.9935 | +0.0040 |
| task198 | 2 | 144 | 1 | 14.2041 | 14.2071 | +0.0030 |
| task174 | 2 | 72 | 1 | 14.5997 | 14.6020 | +0.0022 |
| task191 | 1 | 50 | 0 | 13.8729 | 13.8737 | +0.0007 |
| task077 | 8 | 64 | 1 | 13.4780 | 13.4787 | +0.0006 |
| task005 | 29 | 58 | 0 | 13.5113 | 13.5119 | +0.0006 |
| task054 | 7 | 38 | 3 | 13.5033 | 13.5037 | +0.0004 |
| task133 | 2 | 40 | 0 | 12.8376 | 12.8378 | +0.0002 |
| task076 | 2 | 16 | 0 | 13.7106 | 13.7108 | +0.0002 |
| task370 | 6 | 12 | 0 | 13.8867 | 13.8869 | +0.0002 |
| task101 | 2 | 16 | 1 | 12.6817 | 12.6818 | +0.0001 |
| task018 | 4 | 8 | 0 | 13.3383 | 13.3384 | +0.0001 |
| task071 | 2 | 4 | 0 | 13.7902 | 13.7903 | +0.0001 |
| task001 | 0 | 0 | 0 | 17.1027 | 17.1027 | +0.0000 |

## Per-task results (tasks with at least one no-op)

| Task | No-ops | Categories | Removable bytes | Removable params | Score delta |
|---|---:|---|---:|---:|---:|
| task005 | 29 | AddSub_zero:28, MulDiv_one:1 | 58 | 0 | +0.0006 |
| task013 | 2 | MulDiv_one:2 | 120 | 0 | +0.0043 |
| task018 | 4 | MulDiv_one:4 | 8 | 0 | +0.0001 |
| task034 | 4 | MulDiv_one:4 | 240 | 0 | +0.0040 |
| task054 | 7 | AddSub_zero:4, ShapeOp_same_shape:3 | 38 | 3 | +0.0004 |
| task071 | 2 | ShapeOp_same_shape:2 | 4 | 0 | +0.0001 |
| task076 | 2 | ShapeOp_same_shape:2 | 16 | 0 | +0.0002 |
| task077 | 8 | ShapeOp_same_shape:8 | 64 | 1 | +0.0006 |
| task096 | 4 | Identity:4 | 9,680 | 0 | +0.0595 |
| task101 | 2 | MulDiv_one:2 | 16 | 1 | +0.0001 |
| task133 | 2 | Identity:2 | 40 | 0 | +0.0002 |
| task173 | 14 | AddSub_zero:4, MulDiv_one:10 | 672 | 0 | +0.0072 |
| task174 | 2 | AddSub_zero:2 | 72 | 1 | +0.0022 |
| task185 | 1 | ShapeOp_same_shape:1 | 360 | 3 | +0.0103 |
| task191 | 1 | ShapeOp_same_shape:1 | 50 | 0 | +0.0007 |
| task198 | 2 | MulDiv_one:2 | 144 | 1 | +0.0030 |
| task255 | 2 | Identity:2 | 1,802 | 0 | +0.0050 |
| task285 | 63 | AddSub_zero:27, MulDiv_one:36 | 1,278 | 0 | +0.0104 |
| task370 | 6 | MulDiv_one:6 | 12 | 0 | +0.0002 |

## Caveats

- Add/Sub-zero and Mul/Div-one removals are sign-exact for the final `> 0.0` threshold but can flip the sign of float zeros (`-0.0 + 0.0 = +0.0`) feeding downstream ops; any actual transform must be verified with `outputs_bit_identical` / gold verification.
- Broadcast safety enforced: zero/one-operand eliminations require the surviving input's shape to equal the output shape.
- Orphaned-constant cleanup is one pass (constants whose only consumers were removed no-ops); deeper dead-code elimination is a separate opportunity.
