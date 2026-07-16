# B20 strict optimization report — task162 / task268

## Result

No candidate is eligible for adoption. The B20 winner manifest is intentionally
empty and contributes **+0.0** to the aggregate.

The authoritative Wave13 members were used directly:

- task162: SHA-256 `afcc4eaa06421d665041c082efbf6834cf75a1968e41501b59ae48b33c3f0032`,
  memory 438 + params 13 = cost 451.
- task268: SHA-256 `14aa4e593cbe98d8291b33606f24fab049c5311fdcc7866067b792923736c91a`,
  memory 399 + params 47 = cost 446.

## task162

One algebraically exact parameter shave was found:

- candidate: `task162_reuse_bool.onnx`
- SHA-256: `81e8a9a1475d462fa06792f95c7d5ce3bf7c6ca6253628f70886c2824115ada0`
- change: the final `CastLike` uses the already-computed bool `dilge_107` as
  its type witness, allowing scalar bool initializer `btmpl` to be removed.
- measured cost: 451 -> 450, nominal gain `ln(451/450) = +0.0022197567383130945`.
- disabled-ORT external validator: known 266/266, zero errors; random 100/100
  raw bitwise equal to the exact Wave13 baseline; `ACCEPT_STRICT` under that
  validator's normal preflight.

It is nevertheless rejected by the stronger lane gate. The unmodified model
and candidate both fail ORT_DEFAULT session construction with a
`CenterCropPad` shape-inference error. Under ORT_DISABLE_ALL, the runtime-shape
audit finds **261 mismatches among 267 declared intermediates** (for example,
`csp30f` is declared `[1,1,1,1]` but runs as `[1,30,30,30]`). This is a direct
shape cloak, so neither the normal adoption gate nor the exact-baseline 95%
exception applies.

An exact CSE candidate was also built. It removes 60 duplicate
`CenterCropPad` nodes and is raw-bitwise equal on 500/500 differential cases,
but its measured cost stays 451 (memory 438 + params 13), so it is not a score
improvement and still inherits the shape cloak.

Historical task162 candidates with static estimates 114--214 were checked in
the archive inventory. Their runtime-measured costs start at 828/829 and do not
beat the exact Wave13 member; they use the same underdeclared-shape technique.

## task268

No compliant local rewrite exists from this baseline:

- The exact Wave13 graph contains a `TfIdfVectorizer` with a very large n-gram
  table, which is a prohibited lookup construction.
- Its runtime-shape audit finds **32 mismatches among 45 declared
  intermediates** under both ORT modes (`_tokens` declared `[1]`, runtime
  `[30]`; repeated 1x1 CenterCropPad declarations run as 29x29--31x31).
- Removing the scalar bool `CastLike` witness with a normal `Cast` is exact but
  exposes actual memory: cost 446 -> 1344, so it is not cheaper.
- The historical cost-327 rebuild is not eligible: the independent 3000-case
  differential already found 12 threshold-output mismatches. It therefore
  cannot use the exact-bitwise >=95% exception, and it remains lookup-based.

The generator was inspected (`task_aba27056.py`, grids 5--10). A truthful
rule-based rebuild must materialize at least a spatial mask/label path before
padding to 30x30; it cannot beat the artificial cost 446 floor without the
forbidden lookup/shape-underdeclaration mechanisms.

## Evidence

- `winner_manifest.json` / `rejected_manifest.json`
- `strict_gate_audit.json`
- `runtime_shape_audit.json` and `dual_ort_smoke.json`
- `task162_reuse_bool_external100.json`
- `task162_cse_external500.json`
- `task162_reuse_bool_build.json` / `task162_cse_build.json`

No ZIP, CSV, score ledger, root artifact, or shared handcrafted model was
modified.
