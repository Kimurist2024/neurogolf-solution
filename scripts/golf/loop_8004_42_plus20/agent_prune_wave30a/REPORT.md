# Latent-component prune wave30a — independent audit

## Outcome

**0 / 32 candidates accepted; all 8 tasks are complete.**  Every candidate is
strictly cheaper than its exact member in `submission_base_8005.16.zip`, but no
candidate passes the mandatory known-100% gate.  No fresh run was therefore
eligible or necessary.  Nothing was merged or promoted.

The immutable base audited was:

- archive: `submission_base_8005.16.zip`
- SHA256: `73073208ec8b5b296f1fc13300a7e4df871135f0a9edd4a682ac7b48077aba00`

## Task decisions

The known counts below are identical in `ORT_DISABLE_ALL` and default ORT, and
all listed executions have zero runtime errors.  "Best known" is the strongest
variant after screening all requested alternatives, not necessarily the
cheapest one.

| task | incumbent cost | candidate costs | best known variant | known dual | max Einsum inputs | decision |
|---:|---:|---:|---|---:|---:|---|
| 010 | 44 | 36 | r001/r002/r003 | 0/265 | 28 | reject: known failure + giant Einsum |
| 028 | 72 | 68 | r001/r002/r003 | 0/265 | 10 | reject: known failure |
| 060 | 100 | 90 | r001/r002 | 0/265 | 20 | reject: known failure + giant Einsum |
| 175 | 166 | 142, 145, 146 | r001 (cost 145) | 262/266 | 18 | reject: 4 known errors + giant Einsum |
| 229 | 40 | 30 | r002 | 90/267 | 21 | reject: 177 known errors + giant Einsum |
| 232 | 116 | 102 | r001/r002/r003/r004 | 0/266 | 11 | reject: known failure |
| 304 | 180 | 167, 168 | r004/r005 | 1/266 | 48 | reject: 265 known errors + giant Einsum |
| 315 | 124 | 112 | r002 | 19/266 | 43 | reject: 247 known errors + giant Einsum |

The nominal sum of the best possible cost gains would have been
`+1.1123955387724809`, but its verified admissible gain is exactly `+0.0`.

## Structural evidence

All 32 candidates pass full ONNX checking, strict shape inference with
`data_prop=True`, static positive shapes, standard domains, canonical I/O,
finite initializer, no nested graph/function/sparse initializer, no banned op,
no explicit lookup/cloak op, truthful profiling of every runtime node output,
and Conv-bias UB=0.

The only structural failures are the campaign's giant-Einsum gate
(`Einsum` input count > 16) for tasks 010/060/175/229/304/315.  Tasks 028 and
232 pass all structural checks but are completely wrong on the known corpus.
Several variants also expose positive values in `(0, 0.25)`; the complete
per-variant margin evidence is in `result.json`.

## Fresh/private policy

None of these eight task IDs belongs to the current 51-task private-zero or
unsound-incumbent operational catalog in `docs/golf/private_zero_tasks.md`.
The non-private threshold would therefore have been two independent fresh
seeds at >=90% in both ORT modes.  Because the prior known-100% gate failed for
every variant, all fresh runs were correctly skipped rather than spending
validation budget on already-invalid models.

## Evidence files

- `result.json`: authoritative full results, SHA256, measured costs, every
  variant's dual-ORT known counts, margin evidence, all structural checks, and
  runtime-shape traces.
- `audit_partial.json`: crash-safe final row stream (same 32 audited variants).
- `audit_prunes.py`: reproducible read-only audit driver.

Protected submission and score files were not written by this audit.
