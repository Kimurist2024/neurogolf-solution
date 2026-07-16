# A16 strict cost-floor audit

## Outcome

- Exact source: `submission_base_7999.13.zip`
- Source SHA-256: `a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`
- Tasks: 086, 114, 115, 193, 247, 259, 263
- Archive retained candidates: 0
- Exact-byte-distinct loose historical models: 179
- Safe winners: 0
- Verified gain: `+0.000000`

No root ZIP, CSV, score pointer, or ledger was modified by this lane.

## Baseline and history result

| Task | Exact cost | Unique loose models | Result |
|---:|---:|---:|---|
| 086 | 221 | 57 | 49 alternatives are not cheaper, six unscorable, one structural reject. |
| 114 | 194 | 16 | All 15 alternatives screen at or above baseline. |
| 115 | 197 | 23 | 13 not cheaper, one unscorable, eight structural rejects. |
| 193 | 170 | 15 | All 14 alternatives screen at or above baseline. |
| 247 | 212 | 18 | All 17 alternatives screen at or above baseline. |
| 259 | 191 | 25 | All 24 alternatives screen at or above baseline. |
| 263 | 181 | 25 | All 24 alternatives screen at or above baseline. |

The archive inventory, which covers 1,195 ZIPs, 224,111 ZIP members, and
118,938 loose observations globally, retained no candidate for any of these
seven tasks. The independent repository-wide loose pass likewise found no
structurally sound candidate whose one-case actual profile was below baseline.

## Current-model analysis

All seven exact members pass full ONNX checking, strict shape inference with
data propagation, standard-domain inspection, and Conv-bias checking. None has
an unused initializer or an identical same-shape initializer pair.

- task086 is a 137-node, 13-parameter standard graph. Its cost is dominated by
  208 units of runtime memory. The prior A7 audit found no carrier substitution
  that preserves its verified allocation while reducing cost.
- task114 is standard and has no removable initializer; every historical
  alternative already screens at or above 194.
- task115 is the only giant graph in this wave, with a 73-input Einsum. No
  lower-static archive lead survived inventory, and every loose alternative is
  cost-dominated, unscorable, or structurally forbidden.
- task193 is a one-node direct graph: all 170 cost units are parameters and no
  intermediate is charged. No historical model beats that parameter floor.
- task247, task259, and task263 are standard, bias-clean graphs with no trivial
  deduplication or unused parameter. Their complete loose histories contain no
  lower actual screen.

## Admission disposition

There is no candidate to send to complete-known or fresh validation: all sound
historical models are already at or above the exact actual cost. Fresh 5000/5000
cannot turn a non-cheaper model into a score improvement, so `winners` remains
empty in `final_manifest.json`.
