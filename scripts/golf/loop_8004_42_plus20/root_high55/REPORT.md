# High-cost history pre-screen 55

Eight additional 8005.16 members and their retained history frontiers were
screened. Accepted: **0**; projected gain: **+0.0**.

- task255's apparent zero-input cost878 candidate is not a real improvement:
  the official full-known trace costs1342 versus baseline1336, it has 18
  runtime-shape cloaks, and the generator is non-functional for repeated
  identical inputs. It was already independently rejected by lane_b18.
- task107's apparent708→638 candidate has a 66-input giant Einsum, GatherND,
  13 runtime-shape cloaks, and therefore fails the structural gate.
- task396's lower histories are private-zero/lookup-scatter models. The best
  cost947 lineage is below100% fresh and was already excluded; the other
  variants retain invalid duplicate names or lookup machinery.
- task349/task138/task076/task319/task361 have no strictly lower,
  known-perfect, runtime-truthful candidate in the retained frontier.

Evidence: `history_lead_audit.json`,
`../../loop_7999_13/lane_b18/candidate_audit.json`, and the existing task396
policy90 evidence recorded in `LOOP_STATE.md`.
