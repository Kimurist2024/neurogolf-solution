# Binary affine HardSigmoid scan 145

All 400 immutable payloads were scanned for `Add(Mul/Div(binary,alpha),beta)`
chains whose complete `{0,1}` support stays inside `[0,1]`, permitting an exact
one-node `HardSigmoid(alpha,beta)` fusion and dead-initializer removal.  No
eligible chain exists.  Safe adoptees: 0; gain `+0.0`.  Evidence: `build.json`
and `build_candidates.py`.
