# task377 nonnegative reduction audit 139

## Result

No candidate was admitted. The authority computes
`ReduceL1(x, axes=[2,3], keepdims=0)` where `x` is binary one-hot input cast to
float16. Thus `abs(x)=x` and `Einsum('abcd->ab', x)` is value- and
shape-equivalent; all reduced integers are in `[0,900]` and exactly
representable in float16.

The candidate passes full checker, strict data propagation, and UB0, and
removes the two-element axes initializer. However, its measured runtime memory
increases344->362, so official cost worsens409->425 despite parameters65->63.
It was rejected before runtime/fresh auditing and was not staged.

Evidence: `build.json` and `build_candidate.py`.
