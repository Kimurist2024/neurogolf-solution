# Root reduce/scalar and Conv absorption scan 248

## Outcome

The all-400 scan found one strictly lower Reduce/scalar fusion, task205, and no
eligible Pad/shape-op absorption into Conv-family nodes.

Authority: `submission_base_8009.46.zip`, SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`.
No root submission, score ledger, or `others/71407` active file was changed.

## ReduceSum / ReduceMean -> scalar arithmetic census

All 400 authority graphs were inspected for a ReduceSum or ReduceMean output
with exactly one consumer, immediately followed by Mul, Div, Add, or Sub with a
constant scalar. The seven authority models that fail strict shape inference
(018, 112, 117, 170, 243, 245, 397) were still included in the raw graph census
but could not produce an eligible strict candidate.

Exactly one pattern exists:

- task205: `ReduceSum(row_mask[30,1], keepdims=0) -> tall_f`, then
  `Mul(tall_f, rowpow_thr=1.9019999504089355) -> colq_scale`.
- `tall_f` has one consumer. Both operands and results are FLOAT.
- The two nodes are replaced by one `Einsum(row_mask, rowpow_thr,
  equation="ab,->") -> colq_scale`. Model I/O dtypes and shapes are unchanged.
- Authority cost is memory/params/cost `1031/11/1042`; candidate cost is
  `1027/11/1038`. The four-byte scalar reduction intermediate is removed.
- Candidate SHA-256:
  `d0a53168d0a8313810b2f53cd68c0dd968533677f0b96523f4df375096f434b5`.

The candidate passes full ONNX checker, strict shape inference with data
propagation, ORT_DISABLE_ALL session construction, and the known authority
audit:

| ORT configuration | raw equal | error equal | shape equal |
|---|---:|---:|---:|
| DISABLE_ALL, 1 thread | 266/266 | 266/266 | 266/266 |
| DISABLE_ALL, 4 threads | 266/266 | 266/266 | 266/266 |
| ENABLE_ALL, 1 thread | 266/266 | 266/266 | 266/266 |
| ENABLE_ALL, 4 threads | 266/266 | 266/266 | 266/266 |

There are zero errors, nonfinite values, or values in `(0, 0.25)` on either
side. Floating reduction order is therefore empirically raw-identical across
all four required ORT configurations on the complete 266-case known set.

Fresh `2 seeds x 5000 x 4 configs` was not duplicated in this lane by explicit
parent coordination: another active lane is already performing the all-input /
fresh equivalence proof for this identical task205 cost-1038 construction.
Accordingly this lane records task205 as a known-four-config survivor, not as an
independently fresh-certified promotion.

## Pad -> Conv-family census

Three single-use Pad outputs feed Conv-family data inputs:

- task036 -> ConvInteger: Pad value equals the input zero point, but the pads
  tensor is dynamically constructed.
- task062 -> ConvInteger: pads are dynamic and Pad value `2` differs from input
  zero point `1`.
- task382 -> QLinearConv: Pad value equals the input zero point, but runtime
  `Where` selects the pads.

None satisfies the required static pads/axes gate, so no Pad was moved into a
Conv `pads` attribute. No runtime-invalid authority behavior is relied upon.

## Squeeze / Unsqueeze / Transpose -> Conv-family census

Eight single-use transforms feed Conv-family nodes:

- Three feed data or zero-point slots rather than weights (tasks 038, 185, and
  task382's `count_oriented`).
- Four feed weight slots from dynamic tensors (tasks 057, 121, 124, and 219).
- task382's `W` is an initializer, but its Unsqueeze axis is dynamically chosen
  as 2 or 3 to select horizontal versus vertical kernels. Replacing it by one
  offline-transformed initializer would change semantics.

Thus zero shape transforms have a truthful static weight/attribute absorption,
and zero Conv-family candidates reach the cost gate.

## Evidence

- `scan.py`, `scan.json`: reproducible all-400 Reduce/scalar census and task205
  construction/cost evidence.
- `audit.py`, `audit.json`: task205 known four-config raw/error/shape audit.
- `conv_scan.py`, `conv_scan.json`: reproducible all-400 Pad/shape-op to
  Conv-family census.
- `candidates/task205_reduce_scalar_fusion.onnx`: isolated known-four-config
  survivor; no staging or promotion performed here.

