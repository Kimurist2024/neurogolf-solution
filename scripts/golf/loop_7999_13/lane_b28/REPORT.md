# B28 task347 exact-reuse audit

Baseline score label: `8000.46`  
Baseline ZIP SHA-256: `74cb9c4a96dcf9d1d1ca4ad1e0afdb67f5218c0a06ad8e896ad1654b6548f534`

## Outcome

No eligible winner was found.

The baseline reports cost `41` (`23` memory + `18` parameters) and is correct on all `269/269` known examples in both ORT modes. However, it declares the `GroupNormalization` output `g` and `CastLike` output `x` as `[1,1,1,1]`; both are actually `[1,10,30,30]`. It is therefore a shape-cloaked baseline and cannot be used as the basis of a compliant candidate.

Correcting only those declarations preserves behavior and passes full checker, strict shape inference, both known ORT modes, and Conv UB checks. Its real cost is `45,036` (`45,018` memory + `18` parameters), far above `41`.

## Exact algebraic search

- Duplicate initializers: none. No initializer pair shares dtype, shape, and value.
- Factor/gauge reuse: already exhausted. Scalar `s` is shared by GroupNormalization scale/bias and every quantization scale; `z` is shared by all zero-point inputs.
- Singleton contraction: no Einsum, Gemm, or MatMul exists. The dynamic QLinearConv already reuses `x` as data and weight, while the final ten-channel ConvInteger weight is minimal for its interface.
- Common subexpressions: no duplicate node signature exists. `g`, `x`, `s`, `z`, `W`, and Slice initializer `ss` are already shared at repeated uses.
- The Slice reverses both axes of the `3×3` tensor. It cannot be folded into a positive-stride convolution without another operator or tensor.

## Existing history

Two unique older models were found:

- cost `51`, known `269/269` in both modes: rejected because it uses the same two full-canvas shape cloaks and is not cheaper;
- cost `143`, known `269/269` in both modes: fully shape-honest and Conv-UB-free, but not cheaper than `41`.

There was no cheaper shape-honest candidate to advance. Consequently fresh 5,000-case dual-mode validation and external validator 500 were not run. No root submission or shared score file was modified.

Detailed evidence is in `audit.json`, `algebraic_search.json`, and `winner_manifest.json`.
