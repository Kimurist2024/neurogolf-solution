# 8004.42 batch-20 expansion — low-half lane

## Outcome

The ten assigned tasks were fixed in `TASKS.txt` before exploration:
`025, 062, 008, 134, 112, 184, 168, 048, 037, 014`.

The immutable base was
`scripts/golf/loop_8004_42/submission_8004.42_fixed_rebase_meta.zip`
(SHA-256 `dc64ef5ac672fd7cc67418318e70f94cafd5396af8e25ad2939cdcd4eb1e0149`).
No fixed member, root submission, score ledger, or CSV was modified.

The scan collected 1,877 references and deduplicated them to **639 unique
models**. It applied full ONNX checking, strict static inference, standard
domain/function/subgraph/banned-op checks, Conv-family bias checks, actual
`ORT_DISABLE_ALL` profiling, and complete known-corpus verification. Numeric
winners were then subjected to the stronger strict data-propagation,
truthful-shape, fresh, and no-lookup/no-giant-Einsum gates.

**Safe accepted candidates: 0. Aggregate projected gain: +0.0.**

## Per-task result

| task | base | unique | best distinct known-complete | decision |
|---:|---:|---:|---:|---|
| 025 | 474 | 61 | none below base | no winner |
| 062 | 465 | 49 | none below base | no winner |
| 008 | 431 | 32 | 454 | dominated |
| 134 | 423 | 55 | 320 | reject: lookup tables |
| 112 | 422 | 199 | 420 | reject: strict inference/shape cloak |
| 184 | 421 | 44 | 421 | tie; no gain |
| 168 | 416 | 61 | 166 | reject: lookup + 52-input Einsum |
| 048 | 379 | 36 | 142 numeric / 378 exact rewrite | reject: fresh below 95% |
| 037 | 374 | 40 | 437668 | dominated |
| 014 | 370 | 62 | 370 | tie; no gain |

## Decisive safety findings

- **task134:** the cost-320 graph is known `266/266`, strict
  shape/data-propagation passes, runtime errors are zero, and prior independent
  fresh is `4840/5000 = 96.8%`. It is still inadmissible because ten
  `TfIdfVectorizer` nodes embed thousands of n-gram lookup entries. The
  non-TfIdf cost-412 fallback has six false intermediate shape declarations,
  so it is rejected as shape cloaking.
- **task112:** cost 420 is known `266/266` only with optimizations disabled.
  Strict shape/data propagation fails at `AffineGrid`, default ORT rejects the
  session, and runtime tracing found eleven false declarations, including an
  output declared `[1,1,1,1]` that executes as `[1,10,30,30]`.
- **task168:** cost 166 is known `265/265`, but the 258,672-byte graph stores
  four large `TfIdfVectorizer` tables and uses a final 52-input floating
  `Einsum`. The cost-285 fallback uses the same prohibited families.
- **task048:** the clean cost-378 algebraic rewrite is known `270/270`, passes
  strict structural and Conv-bias gates, and has zero runtime errors. It is
  raw-bitwise equal to the incumbent on 5,000 differential cases, but both are
  unsound on fresh data: `4467/5000 = 89.34%` with `ORT_DISABLE_ALL` and
  `4521/5000 = 90.42%` with default ORT. This is below the user's 95% gate.
  The numeric cost-142 model already failed three of five quick fresh cases.

The other six tasks had no model satisfying both strict cost decrease and the
complete known gate. No fresh run was justified for structurally dominated or
known-wrong candidates.

## Artifacts

- `RESULTS.json`: final ten-task machine-readable decisions and evidence links.
- `scan_results.json`: all 639 unique-model rows, static dispositions, actual
  costs, and known results.
- `winner_manifest_pre_fresh.json`: four numeric winners before the safety
  policy was applied; these are **not approved**.
- `winner_manifest.json`: authoritative empty acceptance manifest.
- `run_archive_scan.py`: reproducible non-promoting scan wrapper.

