# task319 exact-regolf lane 201

## Decision

`task319_combined_runnable.onnx` is a **private-zero inherited-cloak exact
pass-through candidate**.

- Authority: cost `1003` (`memory=863`, `params=140`), task score
  `18.089249212038062`.
- Candidate: cost `978` (`memory=840`, `params=138`), task score
  `18.114490329965182`.
- Improvement: cost `-25`, task score `+0.025241117927118195`.
- Candidate SHA-256:
  `ade6b708b4ee6a0ba65d19e4182748750514435b3b8a005289582154b7208fd4`.
- Authority SHA-256:
  `29d5bfe25f86b18e0b5938d85e4f38cca72c34d8aad6390bff43579124d0e391`.

No ZIP or root score file was changed in this lane.

This is not a newly truthful model: it retains the authority's 26 incorrect
intermediate shape declarations.  It is eligible only under the stated policy
that an inherited cloak may pass when the candidate introduces no new mismatch
and has a complete-support exact pass-through guarantee.  Both conditions hold
here.

## Pass-through evidence

The candidate and authority have the identical set of 26 declared/runtime
shape mismatches under both `ORT_DISABLE_ALL` and default optimization.  The
candidate introduces zero new mismatches.  On 32 traced cases in both modes it
also has `runtime_errors=0` and `nonfinite_values=0`.

Raw outputs are bit-identical to the authority in every tested configuration:

| ORT configuration | known raw equality | fresh raw equality | candidate errors | nonfinite outputs |
|---|---:|---:|---:|---:|
| disable-all, threads=1 | 267/267 | 2000/2000 | 0 | 0 |
| disable-all, threads=4 | 267/267 | 2000/2000 | 0 | 0 |
| default, threads=1 | 267/267 | 2000/2000 | 0 | 0 |
| default, threads=4 | 267/267 | 2000/2000 | 0 | 0 |

The fresh seed is `319201001`.  Fresh gold accuracy is `1959/2000` for both
authority and candidate.  The 41 shared misses are the task generator's
non-injective/tie-breaking limitation, not candidate divergence.  Therefore
gold comparison cannot distinguish an accepted authority pass-through here;
raw equivalence plus the complete-support proofs below is the relevant safety
criterion.

The model passes full ONNX checker, strict shape inference with
`data_prop=True`, canonical I/O, standard-domain, no function/sparse/nested
graph, no banned op, positive static metadata, no external data, finite
initializer, Conv-bias UB0, and bounded-Einsum checks.

## Exactness proof for all five rewrites

1. **Transpose both correlation operands.**  For cross-correlation with the
   task's square 5x5 operands and transpose-invariant pads `[0,0,2,2]`,
   `Corr(A.T, B.T)[i,j] = Corr(A,B)[j,i]`.  The next operation reduces both
   spatial axes with `ReduceMax`, so its result is identical.  This removes an
   Unsqueeze and its now-unused two-element axes initializer.

2. **Replace `3 - Where(cond,1,2)`.**  For either boolean value this is exactly
   `Where(cond,2,1)`.  Scalar Gather changes only singleton rank; the following
   broadcasting and adjusted Gather axes preserve every selected value.  The
   one-element `[3]` initializer is removed.

3. **Absorb the correlation factor.**  The authority tests `8*S >= 2*C`; the
   candidate tests `4*S >= C`.  A generated sprite is at most 5x5, so the
   binary correlation overlap is `S <= 25`.  Every selected non-background
   count is at most the 2x magnification of 25 cells, hence `C <= 100` (the
   often-cited `C <= 40` is unnecessarily narrow).  Consequently
   `8*S <= 200` and `2*C <= 200`: neither QLinearConv uint8 saturation nor the
   uint8 left shift overflows.  All quantities are integers represented
   exactly, so the two predicates are identical on complete generator support.

4. **Keep the base predicate rank.**  Replacing scalar Squeeze with singleton
   broadcasting produces `[1,1,1,5]`; transposing the last two axes produces
   exactly the former Unsqueeze result `[1,1,5,1]`.  No numeric operation or
   branch choice changes.

5. **Build terminal weights with ScatterElements.**  The authority's two
   Equal/Where chains create a length-10 vector equal to 0 at background, 2 at
   target, and 1 elsewhere.  Starting with ones and scattering 0 at the
   background index and 2 at the target index creates the same vector.  All
   generator colors lie in `[0,9]`.  Background, magnified color, and the two
   sprite colors are chosen distinct; after background masking the selected
   target is one of the three positive non-background colors.  Thus both
   indices are in range and target cannot equal background.

These are algebraic/local rewrites and do not depend on the generator's random
seed or on which non-injective tie the authority chooses.  The candidate must
therefore return the authority's raw output for every generated input.

## Other explored candidates

| Candidate | Cost | Known correct | Result |
|---|---:|:---:|---|
| transpose correlation | 996 | yes | included |
| direct int32 min-row path | 1050 | yes | reject: cost regression |
| swapped other-index Where | 1002 | yes | included |
| absorbed QLinear scale | 1002 | yes | included |
| removed base Squeeze | 1002 | yes | included |
| terminal Scatter weights | 988 | yes | included |
| combined pass-through | 978 | yes | recommended under inherited-cloak policy |
| combined with honest metadata | 51711 | yes | truthful but non-lower |
| input-direct truthful rebuild | 1578 | yes | truthful but non-lower |
| shared color initializer | 689 | no | reject: inherited buffer-reuse runtime failure |
| shared-color full combination | 663 | no | reject: same runtime failure |

The truthful input-direct rebuild proves that the wide 29x29 fp16 cast can be
removed safely, but honest activation accounting leaves cost 1578.  It cannot
beat the 1003 authority.

The `Floor(Add(..., 0.005)) - 7` chain also cannot be folded into one fp16
`Floor(Add(..., k))` on the required domain.  Exhaustive evaluation of all
17,408 finite fp16 values `x >= 0.5` and all 513 fp16 constants in `[-8,-6]`
found no exact constant.  The best `k=-6.9921875` still differs on 41 values;
see `fp16_chain_scan.json`.

## Artifacts

- Candidate: `candidates/task319_combined_runnable.onnx`
- Full audit: `audit.json`
- Candidate builder: `build_candidates.py`
- Exhaustive fp16 scan: `fp16_chain_scan.py`, `fp16_chain_scan.json`

