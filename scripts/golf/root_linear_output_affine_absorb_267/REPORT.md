# Linear-output scalar/affine absorption lane 267

Decision: **NO_PROMOTION**.

## Authority and scope

- Immutable authority: `submission_base_8009.46.zip`
- SHA-256: `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
- Scanned all 400 tasks.
- Producers: `Einsum`, `MatMul`, `Gemm`, `Conv`, `ConvInteger`, `MatMulInteger`, `QLinearConv`, `QLinearMatMul`.
- Required consumer pattern: single-use producer output followed immediately by binary `Mul`, `Div`, `Add`, or `Sub`.

The scan found 67 structural sites, of which 50 had a one-element scalar operand. No direct `MatMul`, `Gemm`, `Conv`, `ConvInteger`, or `MatMulInteger` site survived into this 67-site set. The pairs were:

| Pair | Count |
|---|---:|
| Einsum -> Add | 5 |
| Einsum -> Div | 20 |
| Einsum -> Mul | 2 |
| Einsum -> Sub | 21 |
| QLinearConv -> Add | 4 |
| QLinearConv -> Mul | 1 |
| QLinearConv -> Sub | 7 |
| QLinearMatMul -> Add | 2 |
| QLinearMatMul -> Mul | 2 |
| QLinearMatMul -> Sub | 3 |

## Exact absorption census

Only two sites admitted the requested exact structural rewrite: task054 and task117, both `Einsum -> Mul` with a dynamic one-element operand that can be appended to the `Einsum` equation.

No site supplied a constant scalar suitable for exact offline rescaling of a serialized weight/bias, `Gemm` alpha/beta, or an `Einsum` constant operand. Therefore:

- offline constant absorption candidates: 0;
- serialized coefficient round-trip attempts: 0;
- inexact serialized coefficients accepted: 0;
- shared weights/scalars changed: 0.

The remaining sites were rejected before mutation: 25 dynamic `Sub`, 15 dynamic `Div`, 3 dynamic `Add`, 5 nonlinear `scalar / linear_output` cases, and 17 non-scalar operands. In particular task159 is `cEnd32 / red_count_f`; moving it into a linear kernel would introduce a reciprocal and is not an affine absorption.

Quantized-output arithmetic was not moved through quantize/round/clip semantics. The non-scalar task368 `QLinearConv -> Mul` site also has a dynamic quantized producer output and is outside scalar offline rescaling.

## Candidate audit

### task054

The rewrite appended dynamic `bcount[1]` to `Einsum(input16, B, coord, coord)` and eliminated the following `Mul`. It removed exactly the two-byte float16 intermediate `sumR2[1]`:

| Metric | Authority | Candidate |
|---|---:|---:|
| Activation bytes | 2008 | 2006 |
| Parameter bytes | 250 | 250 |
| Cost | 2258 | 2256 |

It passed full ONNX checking, strict shape inference with data propagation, static-shape checks, standard-domain/nested-graph/function/external-data restrictions, and Conv-bias UB0. Official verification returned `correct=true` and cost 2256. All 28 serialized initializers were byte-identical; no shared constant was edited.

It nevertheless failed the mandatory truthful runtime-shape gate. With graph optimization disabled, memory pattern/reuse disabled, and every statically typed node value exposed, 67 of 124 traced candidate values disagreed with their declared shape. For example, `input16` was declared `[1,1,1,1]` but ran as `[1,10,30,30]`. The unmodified authority similarly had 67 mismatches among 125 traced values. This is the known task054 shape cloak, so the candidate was rejected fail-closed.

Because truthful runtime shape precedes semantic testing, the floating-order rewrite did not reach known4 raw comparison or fresh `2 x 1000`. Running those tests could not rehabilitate a model that already violates the explicit shape gate.

### task117

The analogous `Q * N` absorption never became a lower candidate. Full checking passed, but strict data propagation failed at `AffineGrid` node `grid3`: inferred dimension 3 disagreed with declared dimension 1. It was rejected fail-closed before scoring or runtime comparison.

### task367 exclusion

`Einsum(...)->G` has seven uses, including the relevant `Mul` and `Div`, so it fails the required single-use-output gate. Its authority was also traced explicitly: 65 of 74 typed values had runtime-shape mismatches, with 33 nonfinite elements. Thus both the fanout rule and the known shape cloak exclude it.

## Verification artifacts

- `scan.py` / `scan.json`: all-400 enumeration, eligibility reasons, candidate construction, profiles.
- `audit.py` / `audit.json`: full/strict/static/UB0, official score, initializer identity, truthful runtime-shape traces, and gated raw/fresh policy.
- `result.json`: compact machine-readable decision.
- `candidates/task054_einsum_scalar_fuse.onnx`: rejected evidence candidate only; not promoted.

No root submission, stage ZIP, or other lane was modified.
