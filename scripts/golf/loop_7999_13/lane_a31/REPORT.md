# Lane A31 — task273 / task306 strict optimization

## Result

One strict winner is retained: **task306 cost 131 -> 128**, projected score
gain **+0.023167059281533398**.

- Candidate: `task306_reuse_dp0_diag_for_s.onnx`
- Candidate SHA-256: `3d916fde816b0d5e457bd265734b9a00c0294ed2cee5e63f947bf241d4327bb1`
- Exact Wave16 member SHA-256:
  `c8aad29a01fda297893bddf3975d542246c60b692eaa08c823cbc79b53372b97`
- Cost: memory `0 -> 0`, parameters `131 -> 128`, total `131 -> 128`

No root submission ZIP, score CSV, shared artifact, or protected baseline was
modified.  The lane-only candidate ZIP exists solely for archive/Conv-UB audit.

## Exact task306 algebra

The incumbent has `S=[1,1,-1]` and
`diag(Dp0)=[-1,-1,0.5]`.  Let `c=[-1,-1,-2]`.  The candidate applies

`Dp0'[:,q] = Dp0[:,q] * c[q]`,
`Dp1'[:,q] = Dp1[:,q] * c[q]`, and
`X'[q,:] = X[q,:] / c[q]`.

Thus every individual `Dp0*X` and `Dp1*X` factor in the incumbent's one
Einsum is bit-identical, while `diag(Dp0')=S`.  Each of the twelve scalar `S[q]`
operands is replaced by the repeated-index diagonal operand `Dp0'[q,q]`, and
the three-element `S` initializer is deleted.  All scales are signed powers of
two, and `winner_audit.json` checks every factor array with exact NumPy equality.

This does not add or enlarge a giant Einsum: base and candidate both retain the
same single 69-operand Einsum.  Only twelve existing operands are rewired; the
other 57 operands and their subscripts are unchanged.

## Mandatory validation

- Independent validator: **ACCEPT_STRICT**, known **265/265**, errors **0**,
  cost **128**; baseline known 265/265, cost 131.
- Independent generic differential: **500/500 raw-bitwise identical**,
  threshold mismatches 0, asymmetric/runtime failures 0, max raw difference 0.
- Known raw differential under both ORT modes: **265/265 raw-bitwise
  identical**, errors 0 and non-finite values 0.
- Fresh generator: **5000/5000** under `ORT_DISABLE_ALL` and **5000/5000**
  under default ORT; runtime/output failures 0 in both modes.
- Full ONNX checker and strict shape inference: pass.
- Truthful shapes: the single node maps the public
  `[1,10,30,30]` input directly to `[1,10,30,30]` output; there are no hidden
  intermediate value tensors.
- Functions, sparse initializers, nested graphs, foreign domains, banned ops,
  and Conv-family nodes: absent.  Lane candidate ZIP Conv bias UB count: 0.

Evidence: `winner_audit.json`, `task306_fresh5000.json`,
`task306_external500.json`, and `task306_candidate_zip_audit.json`.

## task273 outcome

No strict winner is retained; Wave16 cost remains **193**.

The sole archived lower-cost candidate (193 -> 192) deletes the one-element
separator `S` and recompiles all eight `TfIdfVectorizer` tables for homogeneous
tokens.  It is not equivalent: on `train[0]`, the incumbent vector
`[1,3]` becomes `[4,0.75]`, and the final output is wrong.  Reproducing the
failure confirmed the cause: the separator distinguishes n-gram counts needed
to encode both the affine constant and coordinates 0..9.  Separator-free
homogeneous encodings that recover all ten coordinates require longer token
tensors (raising measured memory above the one-parameter saving) or much larger
lookup pools.  Neither is a lower real-cost, no-new-behavior candidate.

State/basis sharing probes documented in the prior immutable-baseline lane also
fail archived gold.  No task273 candidate reached the fresh/external gates.
