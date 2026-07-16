# Lane C17 — exact 7999.13 task233/task366 soundness audit

## Outcome

No candidate was promoted. Projected gain is **+0.000000**. The exact
`submission_base_7999.13.zip` remains unchanged.

| task | exact-base cost | best cheaper lead | result |
|---:|---:|---:|---|
| 233 | 7758 | none | every historical/spec-derived candidate is more expensive; the incumbent and compact descendants also fail default ORT |
| 366 | 7987 | 7646 | rejected: generator fresh 4685/4757 (72 failures) and 107 runtime shape mismatches |

The required winner gate was complete known correctness, independent fresh
100% in both ORT modes, full checker, strict shape/data propagation, runtime
shape agreement, standard-domain/static graph, no lookup/giant-Einsum cloak,
and exact Conv-family bias lengths. No audited graph passed all gates while
strictly reducing cost.

## task233

The exact member costs `7758 = 7439 memory + 319 parameters`. It is correct on
266/266 known examples with `ORT_DISABLE_ALL`, but default ORT produces only
49 correct, 215 wrong, and 2 runtime errors. Runtime tracing finds 25 declared
versus actual shape mismatches, including fixed-size input, rank, bbox, and
renderer cloaks.

The compact descendants cost 8992 and 9189, so they are already dominated by
7758; they inherit the default-ORT result and 19 runtime shape mismatches. The
cleaner specification-derived diagnostic costs 17007 and works on all known
examples in both ORT modes, but it is 9249 cost units above the exact baseline
and retains 9 nonfatal runtime shape mismatches. Its audited descriptor plus
renderer alone was previously measured above the baseline, so local shaving
cannot create a safe sub-7758 graph.

## task366

The three known-complete cheaper archive leads have actual costs 7646, 7916,
and 7985. All are 255/255 on the complete executable known corpus in both ORT
modes, and pass full checker/strict inference, but runtime tracing exposes
107, 92, and 100 declaration mismatches respectively. They therefore fail the
no-shape-cloak requirement.

The 7646 candidate additionally has direct generator evidence of 72 failures
among 4757 executable fresh cases (98.4864%). A separate 5000-case differential
run differs from the exact baseline on 177/4999 executable cases. The exact
7987 baseline itself fails 3/96 generator-fresh cases, consistent with the
documented unsound color-role shortcut.

The underlying generator allows repeated template colors in the sparse panel;
the measured collision rate inside that panel is 13.70%, so connected-component
template separation is required. The existing CC-only control already costs
13309 before matched filtering and output construction. The superficially
cheap 5246 graph avoids that floor but fails 43/255 known cases in both ORT
modes. No sound standard-ONNX sub-7987 path remains in the audited families.

## Reproduction and integrity

- `candidate_audit.json`: actual scorer costs, full/strict checks, dual-ORT
  known results, runtime shape traces, domains, banned ops, functions, sparse
  initializers, giant-Einsum flags, and Conv bias findings.
- `fresh_evidence.json`: retained independent fresh and differential evidence.
- `winner_manifest.json`: empty winner list.
- `rejected_manifest.json`: candidate-level rejection reasons.
- `validation/root_integrity.json`: exact archive hash, 400-entry ZIP test, and
  root-write declaration.

Exact archive SHA-256:
`a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`.
