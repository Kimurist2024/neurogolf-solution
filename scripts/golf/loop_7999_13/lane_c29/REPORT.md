# Lane C29 — task121 / task315 strict optimization

## Result

One winner is retained: **task315 cost 128 -> 124**, projected score gain
**+0.031748698314579826**.

- Candidate: `task315_tied_color_factor.onnx`
- Candidate SHA-256:
  `c18089d1e14b8b4a5bc99f15b916788e6a9e90c08cb985e68cf45c3c1cffd16b`
- Exact Wave16 member SHA-256:
  `3a5395f8b8f80a542b1716927d2c07b104587de69fba2c796180d1e3de0dcdf8`
- Cost: memory `0 -> 0`, parameters `128 -> 124`

No root submission ZIP, score ledger, pointer, CSV, or shared artifact was
modified.

## task315 true rule and the reduction

For a 3x3 grid `j`, task315 emits a 9x9 block grid.  At output coordinate
`(3*gate_row + source_row, 3*gate_col + source_col)`, the output is
`j[source_row][source_col]` exactly when
`j[gate_row][gate_col] > 1`; otherwise it is background zero.  This is the
literal expansion of `raw/task315.py`.

The incumbent's one-node tensor network has two spatial routing modes.  Mode
0 routes the gate cell by `floor(output_coordinate/3)` and mode 1 routes the
source cell by `output_coordinate % 3`.  Its color term was

`K_t(m,c) = (F_m^T L1_t F_c) * (F_m^T L2_t F_c)`.

The candidate solves a common first color factor and retains a mode-specific
second factor:

`K_t(m,c) = (F_m^T L1 F_c) * (F_m^T L2_t F_c)`.

Thus `L1` shrinks from `[2,2,2]` to `[2,2]`, and the Einsum subscript changes
from `tAB` to `AB`.  Four parameters are removed.  The existing Einsum is not
added to or enlarged: both baseline and candidate have one node and exactly
43 operands; the equation shortens from 142 to 141 bytes.  There are no
intermediate tensors, lookup tables, banned ops, functions, sparse tensors,
foreign domains, Conv-family nodes, or non-finite initializers.

## task315 mandatory validation

- Independent validator: candidate known **266/266**, errors **0**, cost
  **124**; baseline known **266/266**, errors **0**, cost **128**.
- `ORT_DISABLE_ALL`: known **266/266** and fresh **5000/5000**, errors **0**.
- Default ORT: known **266/266** and fresh **5000/5000**, errors **0**.
- Both modes have minimum positive raw value `3.138763427734375`, maximum
  non-positive raw value `0.0`, and zero non-finite values.
- Complete generator-domain proof: all **3^9 = 19,683** possible 3x3 grids
  over colors `{0,1,2}` pass.  Real-arithmetic margins are at least
  `+3.1391309727221315` for positives and at most
  `-2.6873449372548177` for in-grid negatives.
- Full ONNX checker and strict shape inference pass.  Input and output are
  truthfully `[1,10,30,30]`; the one-node graph has no charged/runtime
  intermediate shapes and no shape cloak.

Evidence: `winner_audit.json`, `task315_external_validator.json`,
`task315_tied_color_factor_fresh5000.json`, and
`task315_complete_generator_proof.json`.

## task121 outcome

No task121 winner is retained; Wave16 remains cost **125**.

The true rule finds the unique value 8, crops the surrounding 3x3 object, and
restores the center from the object's color (`max(top_row)`).  The history scan
covered 577 loose files / 25 unique hashes and found no below-125 model.

An exact ONNX-spec probe reused the existing int8 zero for Slice's axis-1
start: on an extent-10 channel dimension, starts `-12` and `0` both clamp to
zero.  It has static cost 124, but it is not admissible.  The incumbent and
probe declare the GroupNormalization and CastLike runtime tensors as
`[1,1,1,1]` although GroupNormalization preserves the actual
`[1,10,30,30]` input shape.  The probe additionally fails representative
runs in both ORT modes in QLinearConv.  It is rejected for inherited shape
cloak and runtime errors; fresh validation was not started.

Task315 history covered 568 loose files / 18 unique hashes; none was below the
Wave16 cost 128.  The retained tied-factor candidate is new to this lane.
