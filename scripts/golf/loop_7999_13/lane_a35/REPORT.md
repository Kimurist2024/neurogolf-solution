# A35 task002 structural audit

Against `submission_base_8000.46.zip` SHA-256
`74cb9c4a96dcf9d1d1ca4ad1e0afdb67f5218c0a06ad8e896ad1654b6548f534`,
task002 has **no strict winner**.  The incumbent remains cost **1286**
(`memory=360`, `params=926`, score `17.8407080952`).

The true generator paints the interiors of hidden honey-pot frames and then
performs a row-major surrounded-cell fill.  The hidden pot parameters are not
part of the ONNX input.  The repository's existing same-input/different-output
counterexample therefore still proves that the task is not a deterministic
function of the public input in every generated case.  The incumbent is a
useful input-only enclosure heuristic, not a literal compiler of the hidden
generator state.

The incumbent has three nodes.  `rc = Einsum(input,A)` is a 30-element
float32 vector, `Feat = Pow(rc,E)` forms the two rows `[1,rc]`, and the existing
output-direct Einsum implements the enclosure heuristic.  Its initializer
bank is:

- `A[30,30]`: 900 parameters, rank 30, 88 nonzeros; the width-three
  tridiagonal adjacency shared by every path step.
- `S[2,10]`: 20 parameters, rank 2; the two independent color modes.
- `P[2,2]`: 4 parameters, determinant `1,000,000`, rank 2; the independent
  boundary and color-preservation modes.
- `E[2,1]=[[0],[1]]`: 2 parameters; the minimum broadcast exponent bank that
  creates `[1,rc]`.

The apparent sparsity of `A` cannot be used: `sparse_initializer` is a proven
grader-error structure and is explicitly banned by the repository rules.  A
conventional exact low-rank factorization cannot reduce `A`, because its rank
is 30 (two dense factors already require at least 1800 elements).  Its Toeplitz
shift structure cannot be represented by a smaller ordinary Einsum operand
without materializing counted shift/Conv intermediates.  `S` and `P` are both
full rank on their active axes, and neither can broadcast-collapse an axis
without destroying one of the two required modes.  `E` already has two
elements, while reusing the matching `S[:,3]` values would require a reshape or
a 10-fold larger `Pow` output.  No legal one-node fusion removes the 120-byte
`rc` tensor while still producing the two independent features.

The baseline passes all 268 known examples and the external validator's full
preflight.  The external profile independently confirms cost 1286, truthful
memory 360, params 926, and zero known errors.  However, fresh generator seed
`35200000` gives only **96/100** under both `ORT_DISABLE_ALL` and default ORT,
with four semantic mismatches in each mode.  This is above the historical 95%
relaxed threshold but fails the current error-free strict gate.  Consequently
an exact output-preserving shave would inherit a failing baseline, and no
fresh5000 run or submission ZIP was produced.

Artifacts:

- `baseline_fresh10.json`: 10/10 in both ORT modes.
- `baseline_fresh100.json`: 96/100 in both ORT modes, generation errors 0.
- `baseline_external.json`: external preflight/known/profile and 20-case
  self-differential audit.

