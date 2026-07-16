# task328 power-of-two margin repair

## Outcome

No safe repair exists in the requested scaling family. The accepted set is
empty, projected gain is `+0.0`, and no ZIP or protected score file was
modified.

The latest comparison base is `submission_base_8005.17.zip`, SHA-256
`c48fa65401a5bd26d3ed1c556eee8f85c0a2063db313be6b96c73e86159b0a04`.
Its task328 member is unchanged from 8005.16.

## Candidate family

The cost-554 source has one final 58-input `Einsum`. `Rflip` is a finite
initializer used exactly once and only by that output node. Multiplying it by
positive `2^k` is therefore the cleanest possible uniform output scaling:

- no node, shape, domain, parameter count, or runtime tensor changes;
- checker and strict data-propagating inference continue to pass;
- cost remains `200 memory + 354 params = 554`.

## Impossibility bound

The same source SHA has a retained legal generator true logit of
`7.316870026530253e-11`. Reaching 0.25 requires `k >= 32`; reaching 1 requires
`k >= 34`.

A legal source orbit witness—size 8, opposite corners `(0,0)` and `(7,7)`,
colors 1 and 2—has maximum absolute raw value
`1.4418113088011285e34` in all of:

- `ORT_DISABLE_ALL`, threads 1 and 4;
- default optimization, threads 1 and 4.

Against float32 max, this permits at most `k = 14`. Thus the constraints are
disjoint:

```text
margin >= 0.25: k >= 32
finite high witness: k <= 14
```

The full generator has 71,136 states. Nonzero-color permutation equivariance
reduces it exactly to 143 orbit representatives. The source completed 25
representatives in all four configurations with sign correctness and no
nonfinite value; orbit 25 supplied the decisive high-magnitude witness. Once
the incompatible exponent bounds were established, completing the remaining
source orbits could not create a feasible exponent.

## Boundary artifacts

Two fixed-SHA, cost-554 candidates make the failure concrete:

| candidate | SHA-256 | result on legal witnesses |
|---|---|---|
| `task328_scale2p14.onnx` | `4af3ec72f77db91792d9be0c474d594fb2d0c0be6719c64134c8ae0dee5d9d36` | seed 328260000 case 1 has 8 false positives; case 3 has 7 true cells become non-positive |
| `task328_scale2p32.onnx` | `b3c722042039166b8af2b3de2f000221ce9e9d92b5e28c940bd0fb759be9e04d` | orbit 25 has 28 false positives and 28 true cells become non-positive |

Every failure reproduces in disabled/default ORT with threads 1 and 4. Both
artifacts happen to return finite tensors on the selected witnesses, but their
huge contraction cancellation no longer preserves signs. This is stronger
than the theoretical overflow bound: the actual implementation already
becomes semantically wrong at both boundary exponents.

Because both candidates fail legal generator witnesses, the success-only full
143-orbit and complete-known audits were not run. Neither can be a winner.

## Evidence

- `witness_audit.json`: all four configuration witness results
- `orbit_checkpoint.json`: 25 source orbit representatives and raw bounds
- `task328_scale2p14.build.json`, `task328_scale2p32.build.json`: fixed SHA construction records
- `result.json`: machine-readable disposition
- `winner_manifest.json`: empty winner set
- `build_scale.py`, `audit_orbits.py`, `audit_witnesses.py`: reproducible Codex implementation
