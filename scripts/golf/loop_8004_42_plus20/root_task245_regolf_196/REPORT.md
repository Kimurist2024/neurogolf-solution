# task245 exact positive-Selu regolf 196

## Result

An exact one-parameter reduction was found against the immutable 8009.46
authority.

- authority SHA-256: `228b6ad9f24579bc6f5840da4e5a18f08343b76a26538f500e7e77d328d6e9d5`
- candidate SHA-256: `1b777a51c55fa98ed720fb993a9305bcca2218627592d23e46d8d5a6bce91ba9`
- official cost: `385 -> 384`
- projected gain: `ln(385/384) = +0.0026007817000574403`
- candidate: `task245_selu_cost384.onnx`

This lane did not modify the root submission, score ledger, or baseline ZIP.

## Exact rewrite

The authority has four scalar `Log` outputs followed by `Div(x, two_f16)`,
where `two_f16` is the only use of a one-element float16 initializer. The
candidate:

1. retains the static batch label `n` in each preceding `Einsum`, changing a
   scalar result followed by rank-1 broadcasting into the same single value
   directly represented as shape `[1]`;
2. replaces each `Div(x, 2)` with `Selu(x, alpha=1, gamma=0.5)`; and
3. removes the now-unused `two_f16` initializer.

For every generator-valid input the four `Log` results are strictly positive:

- `common.conway_sprite(5,5,14)` refuses a deletion that would empty any row
  or column. The shifted red sprite therefore retains its fifth row/column,
  whose nonnegative source coordinate is at least4. In the current positive
  exponential encoder, the red selector and this coordinate alone make both
  red codes greater than1.
- The green object has corners six cells apart. Its bottom/right corners are
  at coordinates at least6, and every factor used by the green row/column
  reductions is nonnegative with a positive `theta_base` sum. Both green codes
  are therefore greater than1.

Thus every source takes Selu's positive branch, which is exactly `0.5*x` in
real arithmetic. The rank rewrite is also exact because batch is statically1;
it removes a one-element broadcast without changing a contraction over more
than one value.

## Numerical and runtime verification

`audit_candidate.py` exhaustively compared `Div(x,float16(2))` with
`Selu(alpha=1,gamma=0.5)` on all31,744 nonnegative finite float16 bit patterns.
The results are bitwise identical in ORT_DISABLE_ALL and default ORT, and Selu
is bitwise identical between the two modes.

Whole-model authority/candidate comparisons:

- known corpus: raw-bitwise and threshold-equivalent `267/267` in each of
  disabled/default ORT x threads1/4;
- fresh seed245196001: `5000/5000` raw-bitwise equivalent and correct in each
  of the four configurations;
- fresh seed245196002: `5000/5000` raw-bitwise equivalent and correct in each
  of the four configurations;
- runtime errors0 and candidate nonfinite values0 across all comparisons;
- `verify_fix.py --k 2000`: `ADOPT`, fresh `2000/2000`, lib and official gold
  pass, margin minimum1.0, measured cost384.

## Structural and inherited-shape status

- full checker: pass;
- strict shape inference without data propagation: pass;
- node count remains32; parameters `79 -> 78`;
- standard ops only, no Conv-family op, lookup, nested graph, sparse
  initializer, banned op, or new shape witness.

Both the LB-white authority and candidate fail `data_prop=True` at the same
pre-existing `AffineGrid` declaration (`2` inferred versus `1` declared). The
candidate introduces no additional structural mismatch and is raw-bitwise
equivalent to authority. This inherited cloak is disclosed rather than
misreported as truthful.

Machine evidence: `audit.json`. Reproduction: `build_selu.py` and
`audit_candidate.py`.
