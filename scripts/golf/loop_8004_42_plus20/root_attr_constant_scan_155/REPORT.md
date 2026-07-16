# Attribute-carrier exact scan 155

## Outcome

No candidate is admissible.  The fixed winner set is empty and projected gain
is `+0.0`.  Protected ZIPs, score ledgers, and `others/` were not changed by
this scan.

## Search

All 400 immutable 8009.46 members were scanned for two exact ONNX rewrites:

- scalar float32 `PRelu(x,slope) -> LeakyRelu(x,alpha=slope)`;
- `CastLike(x, initializer) -> Cast(x,to=initializer_dtype)` followed by dead
  initializer removal.

There were no eligible scalar-initializer PRelu nodes.  All 728 eligible
profiles were CastLike rewrites.  Five passed full checker, strict data
propagation, and the preliminary declared-shape profiler with a lower cost.

## Decisive official/runtime audit

| task | preliminary | competition actual | known/runtime result | decision |
|---:|---:|---:|---|---|
| 071 | 188 -> 187 | 188 -> 187 | disabled ORT: 265/265 runtime errors; 3 shape mismatches | reject error task |
| 133 | 4393 -> 4347 | 4393 -> 4347 | disabled ORT: 267/267 runtime errors; 30 shape mismatches; nonfinite initializer | reject error/structure |
| 216 | 1025 -> 903 | 1499 -> 903 | disabled ORT: 266/266 runtime errors; 53 shape mismatches | reject value-info/shape cloak |
| 285 | 8623 -> 7273 | 8623 -> 7273 | disabled ORT: 265/265 runtime errors; default session fails; 57 shape mismatches | reject error task |
| 388 | 85 -> 84 | 305 -> 1599 | known 266/266 raw-equal in all four configs, but actual cost regresses by1294 | reject cost |

`CastLike`'s second input was not merely a dtype reference in these graphs: its
inferred shape also participated in the deliberately compact buffer plan.
Replacing it with Cast exposed a contradictory runtime shape for tasks
071/133/216/285.  The sole executable four-configuration rewrite, task388,
materialized substantially more actual memory and was not strict-lower.

## Evidence

- `scan.py`, `scan.json`: exhaustive 728-profile preliminary scan;
- `audit_candidates.py`, `audit.json`: competition profiler, direct runtime
  shape trace, known complete x4, and immutable-authority differential audit;
- `candidates/`: rejected fixed-SHA diagnostics only.

