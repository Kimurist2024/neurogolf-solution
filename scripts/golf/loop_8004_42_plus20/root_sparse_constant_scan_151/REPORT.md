# Sparse Constant scan 151

The alternative `Constant(sparse_value=...)` representation was tested on 262
zero-heavy initializers.  It passes full/strict inference in 256 cases and
reconstructs every dense tensor bit-identically.  However, the scorer charges
the materialized dense Constant output as runtime memory: parameter savings
are exactly offset or exceeded.  The best cases, including task158's 650-cell
zero seed, are one cost unit worse; no candidate is lower.  Safe adoptees: 0;
gain `+0.0`.  Evidence: `scan.json` and `scan.py`.
