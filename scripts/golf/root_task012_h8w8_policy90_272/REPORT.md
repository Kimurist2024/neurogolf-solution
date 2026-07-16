# task012 8x8 normal-POLICY90 candidate 272

## Decision

Primary audit: **PASS, pending independent review before staging**.

The clean output-only depthwise Conv reduces immutable task012 from cost710
to650.  Its projected score gain is `ln(710/650) = +0.088292607146`.

- candidate SHA-256:
  `9aea31a6c01f7af21d893f6e5dde16dc947cdb17088686654f3f568845fbb947`
- authority profile: memory0 + params710 = cost710
- candidate profile: memory0 + params650 = cost650
- graph: one standard-domain Conv, weights `[10,1,8,8]`, bias `[10]`
- full checker / strict data propagation / canonical static I/O: pass
- Conv bias UB findings, runtime errors, nonfinite values, and shape
  mismatches: 0
- lookup, fixture correction, private-zero route, and shape cloak: none

## Accuracy and runtime stability

Every corpus was run in ORT disabled/default optimization with threads1/4.
All four configurations produced identical prediction masks and the same
rates:

| corpus | correct | rate |
|---|---:|---:|
| complete known | 252/265 | 95.0943% |
| complete finite latent geometry | 186/196 | 94.8980% |
| independent fresh seed272012001 | 9478/10000 | 94.78% |
| independent fresh seed272112001 | 9499/10000 | 94.99% |

The smallest absolute output margin over all primary executions was
`0.39035797119140625`; zero-margin elements were0.  Thus the result clears the
user's normal-POLICY90 threshold with no observed runtime/configuration
instability.

## Complete-support relation

The 196-state census is the full `7 * 7 * 4` column-pair/gravity geometry for
the default generator.  The generator always chooses two distinct nonzero
colors.  Candidate channels1 through9 have byte-identical weights and biases,
and group10 Conv processes those channels independently.  Therefore every
nonzero-color renaming is permutation-equivariant and the fixed-color census
covers the full color support.  The candidate is still classified POLICY90,
not exact, because ten of the196 geometry states are intentional disclosed
misses.

## Construction and evidence

The retained layout is 8x8 with top/left padding3.  The case-level MILP solved
the finite generator geometry to proven optimality (`mip_gap=0`) at186/196.
MILP output is construction evidence only; the admission decision uses the
independent ONNX runtime results above.

- `domain_milp.json`: finite-domain optimization record
- `audit.py`: primary structural/runtime audit
- `evidence.json`: complete machine-readable evidence
- `candidates/task012_h8w8_policy90.onnx`: reviewed candidate

Root submission, score ledger, and `others/71407` were not modified by the
primary lane.  Promotion is deliberately deferred until the separate review
lane passes.
