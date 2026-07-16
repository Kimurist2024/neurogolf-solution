# A13 cost-150-500 strict audit

## Outcome

- Exact source: `submission_base_7999.13.zip`
- Source SHA-256: `a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`
- Tasks: 020, 030, 031, 042, 055, 059, 064
- Retained candidates profiled: 44
- Exact-byte-distinct loose historical models screened: 227
- Safe winners: 0
- Verified gain: `+0.000000`

No root ZIP, CSV, score pointer, or ledger was written by this lane.

## Exact baseline and retained result

| Task | Exact cost | Retained | Best eligible result |
|---:|---:|---:|---|
| 020 | 156 | 8 | Four cheaper profiles fail known/runtime; two known-correct candidates also cost 156. |
| 030 | 162 | 2 | Both lower-static candidates retain a 51-input Einsum; structure reject. |
| 031 | 227 | 8 | All are known-correct, but the minimum actual cost is 227; no gain. |
| 042 | 217 | 6 | Every candidate retains a 31-input giant Einsum. |
| 055 | 234 | 8 | Every candidate retains a 19-28-input giant Einsum. |
| 059 | 156 | 8 | Every candidate retains an 18-41-input giant Einsum. |
| 064 | 271 | 4 | Every candidate retains a 58-input giant Einsum. |

No candidate passed all three prerequisite gates: strictly cheaper actual
profile, complete known correctness, and sound structure. Consequently no
candidate was eligible for fresh 5000/5000 testing.

## History coverage

The archive inventory used for this wave had already traversed 1,195 ZIPs,
224,111 ZIP members, and 118,938 loose observations, deduplicating 9,572 models
different from the exact baseline. This lane profiled the 44 lowest retained
models for the seven targets.

An additional repository-wide loose-file pass found 227 task-and-byte-distinct
models:

| Task | Unique loose models | Result |
|---:|---:|---|
| 020 | 32 | One cheaper full profile fails known data; all other sound screens are not cheaper or unscorable. |
| 030 | 28 | No structurally sound cheaper known candidate. |
| 031 | 33 | Six full known-correct profiles below the one-case screen threshold finish at cost 227 or more. |
| 042 | 31 | No structurally sound model screens below 217. |
| 055 | 40 | No structurally sound model screens below 234. |
| 059 | 29 | No structurally sound model screens below 156. |
| 064 | 34 | No structurally sound model screens below 271. |

`loose_history_scan.json` records every exact hash, path, structure verdict,
screen cost, and complete profile where applicable.

## Current-model analysis

All seven exact members pass ONNX full checking, strict shape inference,
standard-domain inspection, and the Conv-bias checker. None contains an unused
initializer or an identical same-shape initializer pair.

- task020 and task031 are already standard, non-giant graphs. Their best
  known-correct historical alternatives are cost-equal, not improvements.
- task030 is a zero-intermediate one-node tensor network. The prior B7 exact
  rational factor audit found no positive-saving factorization or static
  precontraction; its only lower mode rank is cost-neutral.
- task055's prior A7 work tested lower ranks, sparse reuse, and constrained
  refits. Lower candidates fail known/fresh data, the exact control costs more,
  and the sparse formulation fails strict shape inference.
- task042, task059, and task064 have no unused or identical initializer to
  remove. Every retained lower-static variant preserves a forbidden giant
  Einsum, while all structurally valid loose alternatives screen above the
  current actual cost.

task059's exact member is locally divergent on bundled gold under the current
arm64 scorer; this lane does not replace it, and no alternative satisfies the
strict complete-known adoption gate.

## Admission disposition

Fresh validation cannot repair an incorrect, cost-equal, costlier, unscorable,
or structurally forbidden candidate. Since the prerequisite set is empty,
fresh 5000 with either ORT mode was intentionally not run. `winners` is empty
in `final_manifest.json`.
