# Lane B24 — task256 / task257 strict optimization

## Result

No eligible score improvement was found. Wave16 and every shared/root artifact
remain unchanged.

| task | Wave16 cost | truthful memory + params | known, each ORT mode | shape cloak | inherited max Einsum inputs | winner |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 119 | 12 + 107 | 266/266, errors 0 | no | 63 | none |
| 257 | 114 | 0 + 114 | 269/269, errors 0 | no | 27 | none |

Pinned archive SHA-256:
`4014cbafea4862f67ebf5ff24be13149b45b333c95bfa680be7216f001a6bb3a`.

Both incumbents pass the full ONNX checker, strict data-propagating shape
inference, standard-domain/banned-op checks, finite-initializer and Conv-bias
checks. Runtime tracing finds no declared/actual shape disagreement. Their
large terminal `Einsum`s are inherited from Wave16; this lane neither adds nor
enlarges one.

## True rules

The compact Sakana functions and the authoritative ARC generator modules were
expanded independently and checked against every known pair plus 5,000 newly
generated cases per task:

- **task256 / `a65b410d`** — find the red prefix of length `L` at row `R` and
  let `T=R+L`. Above it, paint green left prefixes with lengths `T` down to
  `L+1`; retain the red length-`L` row; below it, paint blue prefixes `L-1`
  down to `1`. Readable reference: known 266/266 and fresh 5000/5000.
- **task257 / `a68b268e`** — for each of the fixed 4x4 output positions, choose
  the first nonzero cell from the four source quadrants in priority order
  TL(7), TR(4), BL(8), BR(6). Readable reference: known 269/269 and fresh
  5000/5000.

task256 is data-dependent global geometry. A conventional sound rebuild is far
above its cost-119 numerical contraction. task257 has a readable, sound
QLinearConv/Pad rebuild, but the best established variants cost 139 or 299,
both above its direct-output cost-114 incumbent.

## Complete history audit

`inventory_history.py` hashed every loose `task256*.onnx` and
`task257*.onnx` under `artifacts`, `inputs`, `others`, and `scripts`, then
reconciled them with the global ZIP harvest:

| task | loose files | distinct payloads | valid static payloads below incumbent |
|---:|---:|---:|---:|
| 256 | 575 | 22 | one |
| 257 | 582 | 28 | zero |

The sole nominally lower task256 archive has cost 91, but expands the terminal
contraction to 77 inputs and is already reproduced under the exact SHA
`5507df70...`. It fails gold and 100/100 fresh cases with a near-zero margin;
it is rejected independently for incorrectness and for enlarging the giant
contraction. task257's apparent cost `-1` entry is the old `novi` metadata
probe: required static intermediate declarations were deleted, so the scoring
pipeline cannot compute a valid nonnegative cost. It is not a candidate.

## New lower-cost probes

Thirty-one models were built from the exact Wave16 payloads. Every model:

- has a lower measured cost;
- passes full checker and strict static shape inference at build time;
- adds no node or `Einsum` operand beyond its stated memory-removal variant;
- never exceeds the incumbent's inherited 63/27-input maximum contraction.

The families were:

- task256: remove/broadcast the dynamic `[1,P]` basis; and exhaustively retain
  either singleton slice of every binary initializer axis in `B`, `M`, `QX`,
  `C`, and `S_XE`;
- task257: broadcast the output-domain mask; and retain either singleton slice
  of each binary axis in `feat`, `proj_a`, and `color`.

Dual-ORT fail-fast results:

- task256: ten models execute but fail the first known example in both modes;
  twelve more raise the same repeated-label diagonal dimension error in both
  modes. No candidate reaches one correct known case.
- task257: all nine models execute but fail the first known example in both
  modes.

The mask/basis tensors are therefore semantically necessary, not removable
bookkeeping. Since no candidate passes the known-complete gate, none is sent to
fresh-5000; fresh validation cannot rehabilitate an already-known mismatch.

## Evidence

- `audit.json` — reference, baseline, structure, runtime-shape, dual-ORT, and
  history summary
- `history_inventory.json` — 1,157 loose files deduplicated into 50 payloads
- `build_manifest.json` — all 31 lower-cost probes and pinned lineage
- `candidate_screen.json` — measured costs and dual-ORT rejection evidence
- `winner_manifest.json` — empty winner set
