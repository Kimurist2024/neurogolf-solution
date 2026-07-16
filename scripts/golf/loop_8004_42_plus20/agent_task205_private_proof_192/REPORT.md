# task205 cost-937 private-zero proof lane

## Outcome

The historical cost-937 candidate is a **hard reject**.  It cannot receive the
user's private-zero exception because both independent kinds of negative
evidence are conclusive:

1. its exact SHA-256 is already in the historical `others/7805` LB-black set;
2. a fully explicit, generator-reachable input makes it return the wrong
   decoded output in both default ORT and `ORT_DISABLE_ALL`.

No exact repair of the cost-937/top2 lineage with cost below 1041 was found.
The staged cost-1041 model remains the safe choice because it is a separate,
all-valid-input algebraic rewrite of the immutable LB-white authority.

No root ZIP, score ledger, CSV, `others/`, quarantine, or other agent's stage
was modified.

## Immutable identities

| role | cost | SHA-256 | disposition |
|---|---:|---|---|
| authority | 1042 | `8a6acdc20a366ccbd32cf761285cbb2f1cbcf7d3d2ef8ea71d0fb5a3ed6f1468` | retain as proof base |
| historical lead | 937 | `bbfa8f5b79d2e8345a39a41f327ac1c2c851f3c7f388dd595c72ef951e1b3050` | hard reject |
| staged exact rewrite | 1041 | `509c1947929ab888cff4443ac5b6d808b213fa5057e1c03a2758c1717b3f9eed` | safe private-zero-equivalent |
| old authority rewrite | 1038 | `43c963c46bda5b444fb830b5495b4d71fb9dcf958e108954cdb9ef1064d9f9a8` | separate lineage; not promoted here |

The root `submission.zip` and `submission_base_8009.46.zip` were both SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
during this audit, and both contain the cost-1042 authority member.

## Explicit reachable counterexample

Authoritative generator:
`inputs/arc-gen-repo/tasks/task_8731374e.py`, SHA-256
`00826d799f300d6453dd48a75e9eef57b678e3ce676216134a91e0546a1bd4cc`.

The deterministic replay is seed `93023205`, one-based valid case `11`:

| latent parameter | value |
|---|---|
| input width × height | `15 × 30` |
| box width × height | `9 × 6` |
| box offset `(row, col)` | `(4, 2)` |
| box color / marker color | `8 / 3` |
| marker rows | `[4, 3, 1]` |
| marker columns | `[5, 6, 2]` |

This is inside every no-argument generator support bound: grid dimensions are
15..30, box dimensions are 6..10, both offsets leave a one-cell exterior
margin, there are three distinct strict-interior marker rows and columns, and
the two colors are distinct.  The full flattened color vector is recorded in
`counterexample.json`; feeding those explicit arguments back to `generate()`
reproduces the seed case byte-for-byte.  Every discrete draw has positive
probability, so this is not merely an arbitrary one-hot adversarial input.

The cost-937 graph finds the correct box origin, box color, and marker color,
but its high-power one-dimensional row score classifies relative marker row 1
as a box row.  It therefore misses six decoded marker-color pixels:

```text
gold row 1: 3 3 3 3 3 3 3 3 3
937  row 1: 8 8 3 8 8 3 3 8 8
```

That is six wrong decoded pixels / twelve differing one-hot cells.  Default
ORT and disabled-optimization ORT produce the same twelve-cell difference.
The authority, staged 1041 rewrite, and old 1038 authority rewrite all match
generator gold on this witness.

## Historical LB evidence

The lead is byte-identical to
`others/2/7805/task205_rebuilt_top2_cost937.onnx`.  The exact SHA is listed in
`agent_expand20h_92/audit/lb_history_exact_sha.json` as task205 from the
historical 7805 LB-black set, matching the task205 black result documented in
`docs/golf/private_zero_tasks.md`.  This is exact-net evidence, not a blanket
task blacklist.

The retained independent fresh runs are consistent with the counterexample:

- seed `93023205`: 4904/5000 = 98.08%, versus authority 4928/5000;
- seeds `90702051` and `90702052`: 4908/5000 and 4913/5000;
- zero runtime errors, but a nonzero correctness error rate in both ORT modes.

Thus the relaxed 90%/95% gate is irrelevant to the private-zero guarantee: an
exact LB-black record and an explicit reachable semantic failure both exist.

## Repair inspection below 1041

A direct same-lineage threshold repair was tested in memory.  Raising
`rowpow_thr` from float32 1.9617 to 1.98 clears the case-11 rounding boundary,
but immediately fails one of 266 known cases and still fails 13/1000 fresh
cases (same seed, disabled ORT; first failure case 143).  It was not serialized.

The complete retained sub-1041 archive frontier does not contain an exact
repair of the top2 lineage:

- costs 951/965/997 and the compact cost-977 rebuild are different high-arity
  marginal families with retained generator failures;
- costs 1010/1015 are exact private-zero quarantined SHAs;
- cost 1036 is runtime-invalid;
- cost 1038 is a rewrite of the authority graph, not a repair of cost 937.

The known spec-faithful 2-D scanners cost 83602
(`97b25629...13b4a`) and 484639 (`28105bb6...45c`).  This supports the decoded
reason for failure: exact localization needs 2-D rectangle/perimeter evidence,
which the cheap top2 marginal graph does not represent.  It does not claim a
mathematical lower bound over every possible ONNX graph; it establishes that
no inspected or retained repair satisfies the requested `<1041` gate.

The cost-1038 candidate is worth keeping separate from this rejection.  It
combines the authority's `ReduceSum(row_mask)` and scalar multiply into one
`Einsum`, has cost 1038, and matches the authority on retained finite tests.
However, it is not a descendant/repair of the 937 graph and this lane does not
upgrade its finite evidence to an all-input equivalence proof.

## Full gate results

`model_audit.json` independently re-runs the full checker, strict shape
inference with data propagation, standard-domain/payload checks, Conv-family
bias UB scan, truthful runtime-shape trace, official profile, and all 266 known
cases in default and disabled ORT for all four identities.  Every model passes
those structural/runtime gates, with zero known runtime errors and zero
declared/runtime mismatches.

`counterexample.json` additionally traces every intermediate on the explicit
witness in both ORT modes.  The cost-937 model exposes 40 runtime tensors with
zero shape mismatches and zero non-finite values.  Its rejection is therefore
strictly semantic, not a runtime or shape-cloak failure.

The staged 1041 rewrite retains the stronger prior proof:

- it replaces two uses of `Mul(x, 1.902)` by
  `Selu(x, alpha=1, gamma=1.902)` and removes one scalar initializer;
- both inputs are nonnegative on every valid one-hot input, so the operations
  are algebraically equal, including at zero;
- retained audit: raw equality to authority on all known cases, two fresh
  seeds × 5000, and 2000 arbitrary finite one-hot canvases in both ORT modes;
- full/strict/truthful/default+disable/UB0/runtime/nonfinite gates all pass.

Therefore its private-zero guarantee is exactly the requested kind: it cannot
introduce a hidden-output change relative to the already LB-white authority.

## Artifacts

- `reproduce_counterexample.py` and `counterexample.json`: deterministic
  support witness, dual-ORT outputs, all-intermediate shape/nonfinite trace;
- `audit_models.py` and `model_audit.json`: reproduced structural, profile,
  known-default/disable, shape, and UB gates;
- `repair_inventory.json`: machine-readable lineage and final dispositions.

Final decision: **reject cost 937; keep staged exact cost 1041**.
