# task159 / task199 / task259 / task301 SOUND exact-regolf

## Result

**Winner: null for all four tasks. Projected gain: +0.0.**

The exact members were extracted from `submission_base_8009.46.zip` (SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`).
No file in the root submission, score ledger, or `others/71407` was changed.

| task | authority SHA-256 | cost (memory + params) | exact result |
|---:|---|---:|---|
| 159 | `713788f565b23b9bcae618295557b6cb298345b2591524901dcdd64a7338b934` | 288 (146 + 142) | null: no dead/CSE/alias/noop saving; current graph is noncanonical and runtime-shape-cloaked |
| 199 | `d236c732d0df80270154b8ee593e17768dd54fc8dcec4aac93e752474651383e` | 261 (0 + 261) | null: already zero intermediate memory; all initializer unfoldings have full required rank |
| 259 | `ec0806a9cd6b934e59eee227c74de76881f37a4277ac7aa7bf5a88c6d0893ba6` | 187 (134 + 53) | null: no mechanical saving; current graph has false output metadata and `ScatterElements` lookup behavior |
| 301 | `f613c7078ca9e622061826376e4c628cb59e19bbba1d8ac208fc26ea0f2f4a0d` | 240 (118 + 122) | null: no mechanical saving; the sole cost-236 local-rule attempt fails the first known case |

## Mechanical and algebraic audit

- Ran 27 optimizer/cleanup profiles per task, 108 total: **0 strict-lower
  profiles**. Every successful profile retained the authority cost.
- Dead nodes, unused initializers, duplicate initializer aliases, common
  subexpressions, and obvious identity/noop nodes: **0 for every task**.
- Full checker and strict shape inference with `data_prop=True` pass for all
  four authority files.
- task199 has no counted intermediate memory. Its principal factors are full
  rank in each used unfolding: `C` rank 4, `P` rank 4, `A__R` ranks 3/4/3,
  `M` and `BT` rank 3, `DU` rank 4, and `DT` rank 3. No exact rank contraction
  or initializer alias exists.
- task301 likewise has full rank on every binary factor used by its incumbent
  contraction. No new giant Einsum, lookup, cloak, or undefined Conv-bias
  construction was introduced.
- task159 and task259 cannot yield an admissible tiny local shave under the
  requested truthful-shape/no-cloak gate: their incumbent output declarations
  are respectively `[1,10,1,1]` and `[1,1,1,1]`, while runtime outputs are
  `[1,10,30,30]`. A metadata-truthful rebuild raises the counted floor instead
  of producing a strict-lower candidate.

## Rejected task301 probe

The only strict-lower construction was
`rejected/task301_cyan_max_exact_REJECT.onnx`, SHA-256
`1e5e1242305eb6e396b13dcd11c27f72ae481ed31250858830ff8d56c96ea325`,
cost **240 -> 236**. It tried to replace the gathered cyan count `n` with the
per-channel maximum `B`.

It is permanently **REJECTED**. The very first known example has channel
counts `[42,3,2,1,4,6,5,0,7,0]`: `B=42` is the background population, while
cyan `n=7`. The probe differs from the authority in 287 raw elements and 217
decoded elements; authority is correct and the probe is not. Fresh testing
was correctly skipped because the mandatory known raw-equivalence gate failed
before fresh admission.

## Evidence and guards

- `evidence/audit.json`: actual profiles, all 108 optimizer results,
  runtime-shape traces, initializer ranks, and the task301 counterexample.
- `evidence/root_guard.json`: `submission.zip`,
  `submission_base_8009.46.zip`, and `all_scores.csv` SHA guards.
- `submission.zip` remains byte-identical to the 8009.46 authority; root
  `all_scores.csv` remains SHA-256
  `8c99379ce1c77e6894d53b593ccb5ae10f01c3c70e5666f832b1b2454c709b78`.

No candidate reached the complete-known exact gate, so there is no eligible
model on which to run the four-configuration fresh campaign. This is a
fail-closed null result, not a policy90 admission.
