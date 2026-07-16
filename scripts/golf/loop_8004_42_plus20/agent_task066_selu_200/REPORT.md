# task066 Selu regolf — lane 200

## Verdict

**ADMITTABLE LANE CANDIDATE; not merged.**  Immutable 8009.46 task066 drops
from official cost **562 to 561**, for projected score gain
`ln(562/561) = +0.001780944370994692` (8009.4617809 before display rounding).

- Candidate: `task066_selu_cost561.onnx`
- Candidate SHA256: `2e3bd402f667062b32858d3a11182d3e8050d833d2974d1d37fbadd688f4648b`
- Authority ZIP SHA256: `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
- Authority task066 SHA256: `bb8cebc8d71d275f4ec3f542d6aefea238b6c36d1cec77c0f2c1d533bf04ab4e`
- Official profile: authority `memory=346 params=216 cost=562`; candidate
  `memory=346 params=215 cost=561`.

No root ZIP/CSV or `others/71407` payload was intentionally edited, no ZIP was
created, and `try_candidate.py` was not used.

## Exact graph delta

The whitelist audit found exactly one changed node and one removed initializer:

- node 64 / output `selQ`: `Div(selLog, ln2)` becomes
  `Selu(selLog, alpha=1.0, gamma=1.4432698488235474)`;
- float16 scalar initializer `ln2=0.69287109375` is removed;
- every other node and common initializer is byte-identical.

`gamma` is the float32 reciprocal of the exact stored float16 `ln2`.

## Generator-domain support proof

The adoption condition `selF>=1` is proved for the full generator domain, not
inferred from samples.  `prove_geometry.py` exhausts every legal geometry tuple:

- S tuples: 15,336;
- U tuples: 449,928;
- including both `flip` and `xpose`: **1,861,056 cases**;
- all assertions pass; no counterexample exists.

The authority algebra gives `G = 2 * (cyan row-bitmask at the green marker
column)` and `O = (cyan row-bitmask at the red-outward adjacent column)` after
the graph's transpose canonicalization.  Mandatory generator guards imply:

| geometry | guaranteed selected bit |
|---|---|
| S, unflipped | green guard `mid-1` is shifted by `G=2*C` to meet the red guard at `mid` in `pairD`; `mid<gr0`, so it survives in `aMask` |
| S, flipped | reflection aligns the same guards in `pairU`; the turn is at least `gr0+2`, so it survives in `bMask` |
| U, unflipped | green guard `base+1`, the `G` shift, and `pairU=...>>(2)` align at `base`; `base>=gr0+2`, so it survives in `bMask` |
| U, flipped | reflection aligns them in `pairD` below `gr0`, so the bit survives in `aMask` |

Random cyan is prepended before the path and guards, so it cannot overwrite a
mandatory guard; additional cyan only adds bits to `G/O` and cannot remove the
proved bit.  If `aMask=0`, the proof supplies `bMask>0`.  If `aMask>0` but
`useB` is true, that can only be through `forceB`, which implies `hasB` and
therefore `bMask>0`.  For positive `bMask`, `bPow=bMask & (-bMask)` is nonzero.
Thus the selected uint32 mask is always positive.

Because generator size is at most 20 and `pow2[20:]` is zero, the selected mask
lies in `[1, 2**20-1]`.  Its float16 cast exposes a value `>=1` or `+inf`, so
the real ONNX-domain `Log` input is never zero/negative and `selLog` is never
negative/NaN.  `geometry_proof.json` contains the exhaustive counts and a
stable protected-tree snapshot.

## Replacement equivalence

The Selu output is **not falsely claimed bit-identical** to Div.  The isolated
operator model enumerated every uint32 value in `[0, 2**20]` in all four ORT
configurations:

| ORT configuration | inputs | fp16 `Div`/`Selu` differences | immediate uint8 carrier differences |
|---|---:|---:|---:|
| disable-all, threads 1 | 1,048,577 | 51 | 0 |
| disable-all, threads 4 | 1,048,577 | 51 | 0 |
| default, threads 1 | 1,048,577 | 51 | 0 |
| default, threads 4 | 1,048,577 | 51 | 0 |

`selQ` has exactly one consumer, `Cast(selQ -> UINT8)` producing `ti`.
The exhaustive carrier equality therefore closes the pass-through proof even
for the 51 fp16 values where multiplication and division round differently.
The candidate `ti` carrier is also bit-identical across all four ORT modes.

## Whole-model audit

Known data is gold-perfect and raw-identical to immutable authority:

| suite | ORT modes | candidate / total | final raw equal | `ti` equal | errors |
|---|---:|---:|---:|---:|---:|
| known | 4 | 266/266 each | 266/266 each | 266/266 each | 0 |

Fresh data is intentionally evaluated as an authority pass-through test.  The
immutable authority itself is not perfect on arbitrary fresh samples, so fresh
accuracy is recorded but is not used as permission to adopt:

| fresh seed | modes | authority = candidate gold | final raw equal | `ti` equal |
|---:|---:|---:|---:|---:|
| 66,200,101 | 4 | 1908/2000 each | 2000/2000 each | 2000/2000 each |
| 66,200,102 | 4 | 1882/2000 each | 2000/2000 each | 2000/2000 each |

Static/runtime gates:

- ONNX full checker: pass;
- strict shape inference: pass;
- strict shape inference with data propagation: pass;
- standard domains, no functions/nested graphs/sparse content/banned ops: pass;
- Conv-family short-bias UB findings: **0**;
- truthful runtime shape trace: 79/79 node outputs, zero mismatches in both
  disable-all and default ORT;
- runtime errors: **0**;
- candidate final nonfinite values: **0**.

The unchanged authority subgraph can expose `selF=+inf` for large uint32 masks
(known: 6 cases/config; fresh seeds: 59 and 51 cases/config).  This is inherited,
is covered by the operator bound, and produced no `selLog`, `selQ`, `ti`, or
final-output nonfinite values in the audit.  Sampled `selF` minimum was 2 and
sampled `selLog` minimum was 0.693359375 in every mode.

## Integrity note

The long audit held root `submission.zip` at
`4eb324d7...` and `all_scores.csv` at `8c99379c...` from start to finish.
During that audit, `others/71407` changed concurrently: `task349.onnx` was
timestamped 01:43:11 and `MANIFEST.json`, `REBASE_8009_46.json`, and `README.md`
at 01:44:34.  This lane never writes outside its own directory and `/tmp`; the
external tree change is preserved verbatim in `audit.json`, which is why the
audit process returned nonzero despite `summary.accepted=true`.

A subsequent independent geometry-proof run observed the protected set stable
before/after, including the then-current `others/71407` tree hash
`16d0df8abb6f0dca78d8be1d0c9e5d6dcdbd839e4ec5cf224f07fdc06651cd72`.

## Artifacts

- `build_candidate.py` — authority-bound, one-delta builder.
- `audit_candidate.py` / `audit.json` — official profile, strict/full/UB,
  4-ORT known/fresh raw audit, all-value operator proof, shape and integrity evidence.
- `prove_geometry.py` / `geometry_proof.json` — exhaustive finite geometry proof.
- `task066_selu_cost561.onnx` — lane-only candidate; not merged.
