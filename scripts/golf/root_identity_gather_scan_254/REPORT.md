# All-400 exact identity-Gather scan

## Outcome

**Accepted candidates: 0. Projected gain: +0.0.**

The immutable authority was `submission.zip`, SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`.
All 400 members were scanned.  The scan covered **511** `Gather`,
`GatherElements`, and `GatherND` nodes across **82** tasks.

The proof conditions were fail-closed:

- `Gather`: initializer indices must be a one-dimensional full-axis sequence;
  after normalizing negative indices, it must equal `0..N-1` in order, and
  output shape must equal data shape.
- `GatherElements`: initializer indices and output must have the full data
  shape, and every index at every static coordinate must equal that
  coordinate's selected-axis component after negative normalization.
- `GatherND`: every coordinate in the complete static output domain is mapped
  symbolically to its source coordinate; all must be identical, including
  `batch_dims` and negative-index normalization.
- An initializer is removed only when every use is one of the proved identity
  nodes.  The combined replacement changes each such node to `Identity`.

## Findings

Only three nodes looked like identity maps under their static declarations:

| task | node output | declaration | disposition |
|---:|---|---|---|
| 090 | `r0` | `Gather(rowbits30,[0],axis=1)`, declared data/output `[1,1]` | generated and rejected: declaration is a shape cloak |
| 382 | `red0slice` | singleton axis-2 Gather | no parameter reduction; `si` has nonidentity uses |
| 382 | `prog0slice` | singleton axis-2 Gather | no parameter reduction; same shared `si` |

No `GatherElements` or `GatherND` node passed the exhaustive identity-map
conditions.

### task090 rejected probe

`candidates/task090_identity_gather.onnx` replaces `r0` and removes the sole-use
one-element `row_idx_0` initializer.

| gate | authority | candidate |
|---|---:|---:|
| official-like memory | 916 | 641 |
| params | 134 | 133 |
| cost | 1050 | 774 |
| full checker / strict data propagation / UB0 | PASS | PASS |
| official known correctness | true | **false** |
| truthful runtime shape | false | **false** |

The declared `rowbits30` shape is `[1,1]`, but runtime carries 30 row values.
Replacing the apparently singleton Gather changes ORT allocation/reuse behavior;
known execution fails at a downstream `BitwiseAnd` with buffer shapes
`{1,1}` versus `{1,30}`.  An independent all-output runtime trace also fails at
downstream broadcasting (`10` versus `126`).  Thus the apparent identity is
created by false metadata and is not a valid tensor-map proof.

The candidate fails before the known-four raw-equivalence gate.  In the
required cost-first order, no fresh run is allowed after this runtime/known
failure.

## Fresh policy

There are zero strict-lower candidates that pass full/strict, official known,
truthful runtime shapes, and UB0.  Therefore fresh `2 x 5000 x 4` was not run.
The required threshold remains >=90% with runtime errors and shape mismatches
both zero.

## Reproduction

```bash
.venv/bin/python scripts/golf/root_identity_gather_scan_254/scan_identity_gather.py
```

`scan_result.json` contains every Gather-family proof attempt, normalized-index
reason, candidate hashes, cost profiles, strict audit, and rejection evidence.

