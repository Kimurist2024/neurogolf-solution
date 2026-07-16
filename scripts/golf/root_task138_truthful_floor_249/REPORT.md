# task138 truthful floor audit

## Outcome

**EARLY STOP / accepted candidates: 0 / projected gain: +0.0.**

The requested archive member
`loop_7999_13/lane_archive_all400/task138_r01_static2588.onnx` is not a
truthful-shape basis.  Its declared static memory is **2359**, while the
official-like runtime profile lifts it by 174 to memory **2533**; with params
**229**, observed cost is **2762**.  An all-output `ORT_DISABLE_ALL` trace finds
**42** declared/runtime shape mismatches.  Three `CenterCropPad` outputs are the
direct cloak roots:

| tensor | declared | runtime |
|---|---:|---:|
| `input_hid` | `[1,1,1,1]` | `[1,10,30,30]` |
| `qcol_abs_hid` | `[1,1,1,1]` | `[1,1,1,30]` |
| `qrow_abs_hid` | `[1,1,1,1]` | `[1,1,30,1]` |

The other 39 mismatches are downstream consequences: `input_f16`, the two
length-30 line scores, all coordinate masks, and the row/column interior masks
inherit static singleton shapes from those three roots.

## Truthful normalization

`analyze.py` constructs a direct normalization without inheriting the cloak:

1. remove `Shape(qcol)` and all three hiding `CenterCropPad` nodes;
2. consume the free FLOAT graph input directly, avoiding a materialized
   `[1,10,30,30]` cast;
3. widen the connected FLOAT16 Einsum arithmetic to FLOAT so it is type-compatible
   with the input; and
4. regenerate every intermediate `value_info` from actual `ORT_DISABLE_ALL`
   shapes.

The retained rejected witness is
`rejected_probes/task138_r01_truthful_direct_f32.onnx`.

| check | result |
|---|---:|
| full ONNX checker | PASS |
| strict shape inference | PASS |
| strict shape + data propagation | PASS |
| all intermediate dimensions static and positive | PASS |
| Conv-family bias UB | 0 (there are no Conv-family nodes) |
| runtime shape mismatches | 0 |
| local official known corpus | correct |
| memory | 5339 |
| params | 229 |
| cost | **5568** |

The input itself is free in the scorer.  Retaining the archive's `input_f16`
would instead materialize an 18,000-byte full-grid activation, so widening the
small projection operands is already the cheaper truthful normalization.

## Exhausted reductions and cost floor

- **Initializer alias/factor:** there are zero exact initializer alias groups.
  Deriving `qrow` from `qcol` by `Transpose` trades 30 parameter elements for a
  30-byte activation, net best case zero.  Extracting `ones30_f16` from the
  60-element power table costs at least 60 activation bytes for 30 parameters,
  so it loses.
- **UINT8/INT8 narrowing:** coordinate, table, and scatter data are already
  UINT8.  Remaining INT32/INT64 values are Gather/Scatter/Pad schema indices.
  Initializer dtype width does not lower parameter cost because parameters are
  charged by element count.
- **Constant fold / shape carrier:** the only relevant fold is
  `Shape(qcol) -> [30]`.  Making it static exposes all three
  `CenterCropPad` contradictions; it is not a truthful reduction.
- **Mask fusion:** four exact-line comparison pairs can become `Equal+Cast`,
  and two interior comparison pairs can become `And+Cast`.  A schema-realistic
  accounting saves at most **900** cost (5568 -> 4668).  Even granting that the
  replacement BOOL tensors cost nothing gives an impossible optimistic saving
  of **1080** (5568 -> 4488).
- **Concat/Gather/Scatter and terminal recoding:** the current terminal chain is
  `rolemap` 40 bytes + `Ttable/Ctable` 210 each + `Tcolor/Ccolor` 300 each =
  **1060 bytes**.  After the impossible free-mask bound above, deleting this
  entire terminal chain for free still leaves cost **3428**, which is **723
  above** authority cost 2705.  A real role/table recode can save less than
  complete deletion.

This is a fail-closed local floor, not a claim that arbitrary ONNX programs can
never express the task differently.  It is enough to rule out every requested
alias, narrowing, fold, mask fusion, and terminal recode on this projection
architecture.  The previously reviewed from-scratch truthful full-grid family
lands around 38k--62k because it must materialize 30x30 data-dependent extent
tensors, so it does not supply an alternate strict-lower route.

## POLICY90 disposition

There is no candidate below cost 2705 after truthful normalization, hence there
is no eligible survivor for the required known-266 four-config and fresh
2 x 5000 x 4 gates.  Per the lane's cost-first rule, fresh tests were not spent.
Runtime errors in the performed known/scoring and shape traces were zero.

## Reproduction

From repository root:

```bash
.venv/bin/python scripts/golf/root_task138_truthful_floor_249/analyze.py
```

`analysis.json` contains the complete 42-mismatch list, hashes, strict audit,
memory inventory, exact initializer scan, and all cost-accounting bounds.
