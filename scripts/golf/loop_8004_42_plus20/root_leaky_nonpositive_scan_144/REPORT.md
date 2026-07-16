# Nonpositive LeakyRelu attribute scan 144

All 400 immutable 8009.46 payloads were scanned with recursive sign proofs for
`Mul(nonpositive, scalar)` nodes whose scalar initializer would become dead
after replacement by `LeakyRelu(alpha=scalar)`.  No eligible node exists.
Safe adoptees: 0; gain `+0.0`.  Evidence: `build.json` and
`build_candidates.py`.  Root files were not modified.
