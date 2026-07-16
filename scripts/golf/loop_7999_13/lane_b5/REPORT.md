# B5 exact-factor/rebuild wave — exact 7999.13 baseline

## Result

No candidate is safe to adopt. The lane gain is **0.0** and the projected
aggregate remains **7999.13**. No root ZIP, score ledger, CSV, or shared model
was modified.

The exact authority was `submission_base_7999.13.zip`, SHA-256
`a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`.
All baseline members were extracted read-only into this lane before analysis.

| task | member SHA-256 | memory | params | cost | result |
|---:|---|---:|---:|---:|---|
| 099 | `de18296f07fc021360fa0fec22861b284840aa44d939bfc14be85038dcc5d998` | 88 | 310 | 398 | unchanged |
| 239 | `e15519d37ccaa4f3ad478091a2eb5a6a1fe984bc602a2ffd347f03c250eb68e0` | 328 | 56 | 384 | unchanged |
| 268 | `14aa4e593cbe98d8291b33606f24fab049c5311fdcc7866067b792923736c91a` | 399 | 47 | 446 | unchanged; order-sensitive |
| 297 | `cdba3d03bf43853742508f284bf98ca5341fdb2ab50042ec895afb0069296537` | 320 | 51 | 371 | unchanged |
| 345 | `36b1e2be6488496ca637996552b505cf9b2742775b663c4210407d61b861d636` | 248 | 141 | 389 | unchanged |
| 374 | `93fb94260388ab83bc35043c0ee11ae08b1bf3e8fa962a3b47b08ba73794d24a` | 451 | 30 | 481 | unchanged; order-sensitive |
| 394 | `cb47909c49db2fab103bdbaa0be19c49d2eacc2336393086260c7754bd0ffb89` | 285 | 65 | 350 | unchanged |

## Candidate harvest and exact-factor audit

`scan_existing.py` scored 213 byte-distinct task-named historical models with
the official-like ORT-disabled profiler and the complete local known corpus.
The durable per-model evidence is `existing_scan.json`.

| task | unique scored | known-correct | strictly cheaper | cheaper + known-correct |
|---:|---:|---:|---:|---:|
| 099 | 29 | 25 | 4 | 0 |
| 239 | 22 | 20 | 2 | 0 |
| 268 | 46 | 46 | 1 | 1 (fresh-rejected below) |
| 297 | 28 | 17 | 11 | 0 |
| 345 | 25 | 24 | 1 | 0 |
| 374 | 26 | 26 | 0 | 0 |
| 394 | 33 | 27 | 4 | 0 |

The coefficient/factor and local-rule families were also checked against the
existing derivations and their generated candidates:

- task099: all four remaining 7x3 rank-2 coefficient reductions score 397 but
  fail complete known gold. Earlier RTc/DTc rank-2 and CP-carrier deletion/
  fusion variants fail for the same semantic reason. The exact cost-398
  coefficient bank is still the floor of this decoder family.
- task239: removing inactive-feature safety costs 374 and removing the inactive
  bar sentinel costs 379, but both fail known gold. The 180-byte bar/feature
  field is required to separate in-rectangle background from outside-grid
  zero-hot cells.
- task297: eleven shared-quantization-scale variants score 370, but every one
  fails known gold. The separate scales are required by the ten-label hash;
  scalar/rank-4 zero-point sharing is rejected by ORT.
- task345: the cost-365 row-2 omission fails known gold. Sparse storage of the
  zero-heavy Conv kernel is rejected by ONNX full checking, while dense
  factorization adds a counted runtime weight tensor.
- task374: the meaning-preserving `CastLike` to `Cast` rewrite removes one
  parameter but exposes the real 10x10 tensor and scores 876 rather than 480.
  No model below the exact cost-481 member survived scoring. No archive member
  or root ZIP was touched.
- task394: deleting either/both bite-coordinate gates produces costs 348/342
  but fails known gold; returning the natural 3x3 result scores 177 but fails
  the mandatory full 30x30 mask. The cost-350 Quantize-based size classifier is
  the best known-correct model in the 33-model scan.

## task268 strict rejection

The only lower-cost complete-known candidate was:

- source: `others/7907/task268_improved_rebuild.onnx`
- SHA-256: `22ea97ffce8b14fbf923a89a0cda2233d83469201732ed3f4e914e5b2b1ced69`
- baseline/candidate cost: `446 -> 327` (`399+47 -> 285+42`)
- nominal gain: `ln(446/327) = +0.3103587811228106`
- complete known: 266/266, errors 0; margin minimum 1.0

It is **not generator-sound**. The mandatory command

```text
.venv/bin/python scripts/verify_fix.py --task 268 \
  --onnx others/7907/task268_improved_rebuild.onnx \
  --k 5000 --min-fresh-rate 1.0
```

returned `REJECT`: fresh correct 2219/5000, failures 2781, rate 44.38%.
Independent 3000-case baseline differential evidence in
`../lane_harvest/external_task268.json` also reports 12 threshold mismatches.
Because task268 is archive/order-sensitive and the candidate fails the strict
fresh gate, it was not staged, zipped, or merged.

## Final decision

`winner_manifest.json` has an empty accepted list. The unsafe task268 model is
recorded only as a rejection. Every retained baseline remains the exact
7999.13 archive member; projected accepted gain is zero.

