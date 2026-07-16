# 8005.16 extended exact-rewrite sweep

**Safe adoptees: 0.**  Fourteen additional whole-payload passes covered
Einsum identity/unit removal, same-subscript/shared-operand fusion, sign and
signed-permutation absorption, single-use inlining, singleton reshape removal,
scalar initializer shaving, zero-border Conv trimming, initializer slice/product
reuse, CastLike anchor conversion, and initializer Shape/Size folding.

Only three cheaper artifacts were emitted:

- task071 CastLike-anchor conversion: 188 -> 187. Default ORT is known 265/265,
  but disable-all produces runtime buffer-shape errors on all 265 known cases;
  it also retains a 39-input giant Einsum. Rejected.
- task397 scalar shave: 364 -> 362. It inherits the incumbent's false output
  shape, 13 CenterCropPad nodes, TfIdf lookup, and default-session failure.
  Rejected.
- task333 sign absorption: 423 -> 421. It retains a 36-input giant floating
  Einsum. Rejected.

No other pass emitted a lower-cost candidate. Gain counted is `+0.0`.
