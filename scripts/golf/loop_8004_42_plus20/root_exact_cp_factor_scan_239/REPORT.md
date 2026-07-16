# Exact CP/Kronecker initializer scan 239

## Result

No strict-lower candidate exists in the scanned family.

- Authority: `submission_base_8009.46.zip`
- Authority SHA-256:
  `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
- Authority models scanned: 400
- Explicit exclusion: task310, because its exact parity factor is already staged
- Eligible small finite floating initializers used exclusively by Einsum: 543
- Tasks containing an eligible initializer: 169 in the preliminary numeric census
- Exact algebraic plans found: 3, across tasks 013 and 398
- Buildable strict-lower candidates: 0
- Candidate files: none
- Projected score gain: 0
- Decision: `NO_CANDIDATE`

The authority ZIP, root models, staging area, and score ledger were not changed.
Kimi and `try_candidate.py` were not used.

## Exact scan boundary

The scan inspected dense serialized initializers of at most 4,096 elements,
rank 2–8, with finite floating values, where every graph occurrence is an
explicit-shape standard-domain Einsum operand. It tested:

1. fully axis-separable rank-1 outer products;
2. rank-1 Kronecker products over every nonduplicate axis bipartition;
3. exact rank-r bipartition factors found by rational column elimination;
4. binary Walsh CP factors with bit-identical stored-dtype reconstruction;
5. deduplicated one-hot CP factors;
6. repeated-label diagonal projections.

No SVD, tolerance, approximate factor, lookup table, archive cloak, or shape
cloak is used. Floating entries are converted to exact rational values for
rank decisions. A proposed factorization must then reconstruct the source
initializer byte-for-byte in its serialized dtype. Candidate construction also
requires full ONNX checking, strict shape inference with data propagation,
strictly fewer official-compatible parameters/cost, and no counted-memory
increase.

The fully separable rank-1 control found zero plans. This agrees with the
independent pre-existing exact rank-1 scanner when rerun against the same
authority, which also produced zero candidates.

## Three algebraic leads and terminal obstruction

Only three tensors had a parameter-reducing exact decomposition:

| task | initializer | source shape/params | exact factor | factor params | apparent saving |
|---:|---|---:|---|---:|---:|
| 013 | `Qor` | 2x2x2x2x2 / 32 | axis partition rank 2 | 24 | 8 |
| 398 | `K` | 3x2x2x2 / 24 | axis partition rank 2 | 22 | 2 |
| 398 | `K` | 3x2x2x2 / 24 | alternate axis partition rank 2 | 20 | 4 |

Each rank-2 replacement needs one new latent contraction index. In every
occurrence, however, the authority giant Einsum already uses all 52 legal
single-character labels (`a-zA-Z`). There is no free label that can be added
without coupling the new latent rank to an existing dynamic index and changing
the function. Adding a separate reconstruction node would create a counted
intermediate tensor and violate the required no-memory-increase boundary.

Accordingly these are recorded in `scan.json` as exact algebraic plans that
terminate with `ValueError: no free Einsum label`; they are not candidates and
no ONNX files were emitted.

## Runtime gate disposition

The requested known/fresh authority raw pass-through audit in four ORT CPU
configurations applies to structural survivors. Since the structural survivor
count is zero, no runtime audit was launched. This is not a partial runtime
claim: there is simply no candidate model to compare.

## Artifacts

- `scan.py`: reproducible exact rational/serialized-value scanner and builder
- `scan.json`: complete census, the three blocked plans, and their errors
- `candidates/`: intentionally empty
