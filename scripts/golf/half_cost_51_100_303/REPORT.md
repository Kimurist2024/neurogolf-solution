# Cost 51–100 optimization lane (8011.05 authority)

## Result

- Guaranteed-safe winners: **0**.
- User-authorized `POLICY95` winner: **task070, cost 66 -> 52**.
- Projected score gain: **+0.23841102344499815**
  (`20.810345257973573 -> 21.048756281418573`).
- The candidate is deliberately classified as
  `POLICY95_PRIVATE_ZERO_RISK`, not guaranteed-safe. Task070 has historical
  private-zero/LB-black lineage.
- No root submission, score ledger, or `others/71407` file was modified.

Authority:

- `submission_base_8011.05.zip`
- SHA-256 `ad96519a07bcae72017bdb92855b361374f7a34c9f40db237cb2e854615bcd56`
- 86 non-score25 tasks whose current actual cost is 51 through 100.

## POLICY95 candidate

Candidate:

- task: `task070`
- path:
  `scripts/golf/loop_7999_13/lane_archive_all400/task070_r02_static52.onnx`
- SHA-256:
  `a4c8818ae04ee8445e42907383d5d1fd003eb0537ff54d48278534e173297b60`
- current authority member SHA-256:
  `a45fe09083c363ab9aae49de2497c55356a3d5bfab324ec2ab6b6ed949cd1c92`
- measured official-like cost: 52 (authority 66)
- graph: one output-only `Einsum`, 17 inputs, zero counted intermediate
  outputs
- full ONNX checker and strict shape inference with data propagation: pass
- canonical `[1,10,30,30]` input/output; no runtime-shape cloak
- no local functions, sparse initializers, nested graphs, banned or nonstandard
  ops, nonfinite initializers, Conv bias UB, runtime errors, or nonfinite output
- known corpus: 266/266 in all four configurations
  (`ORT_DISABLE_ALL`/default x threads 1/4)
- known minimum positive margin: `999.9957885742188`; no values in `(0, 0.25)`

Independent fresh audit (the same result in each of the four ORT
configurations):

| seed | right / total | accuracy | errors | nonfinite | shape mismatch | `(0,.25)` values | min positive |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 303070101 | 1980 / 2000 | 99.00% | 0 | 0 | 0 | 0 | 999.9957885742188 |
| 303070202 | 1969 / 2000 | 98.45% | 0 | 0 | 0 | 0 | 968.7513427734375 |

Both independent streams exceed the user-authorized 95% threshold. They do
not prove private-LB correctness, hence the explicit risk classification.

## Rejected neighboring candidates

- task070 cost50, SHA-256
  `d1bbfb9409a3ba1a0a9abdf4e2eebbb799c56c63f13024b74221c7955e388dd4`:
  fresh accuracy was 98.60% and 98.20%, but fresh outputs contained respectively
  2 and 5 positive values below 0.25 (minimum 0.22824 / 0.04088). Rejected by
  the mandatory margin gate.
- task049 declared-cost candidates passed the relaxed fresh-rate screen but
  measured at actual costs 87–90 versus authority 75; rejected as regressions.
- task072 declared cost31 measured at actual cost421 versus authority78.
- task393 declared cost33/35 measured at actual cost121/123 versus authority86.

## Exhaustive census

The half-cost pass searched 9,148 loose ONNX paths (225 unique target/hash
pairs) and all 377 local ZIPs / 31,838 relevant members (728 unique
target/hash pairs). No candidate measured at or below half its authority cost.

The broadened strict-lower pass deduplicated the combined loose/ZIP history to
847 unique task/hash pairs. Of 173 models whose declared lower bound was below
the authority, 70 were known-exact before actual repricing, but **zero**
non-catalog candidate had a strictly lower actual cost after structural and
runtime gates. Historical evidence therefore supplied no guaranteed-safe
winner; task070 cost52 is the only retained relaxed-policy result.

Reproduction:

```bash
.venv/bin/python scripts/golf/half_cost_51_100_303/history_scan.py
.venv/bin/python scripts/golf/half_cost_51_100_303/zip_history_scan.py
.venv/bin/python scripts/golf/half_cost_51_100_303/strict_history_scan.py
.venv/bin/python scripts/golf/half_cost_51_100_303/audit_task070_policy95.py --variant cost52
```

Machine evidence is in `history_evidence.json`, `zip_history_evidence.json`,
`strict_history_evidence.json`, and `task070_policy95_cost52_audit.json`.
