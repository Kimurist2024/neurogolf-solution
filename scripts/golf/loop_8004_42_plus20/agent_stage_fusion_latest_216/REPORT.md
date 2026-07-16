# Current 71407 exact fusion / cleanup rescan

## Outcome

No additional strict-lower model was found. The current active stage contains
16 models, including the latest task066, task192, task319, and task349
descendants. No staged model, root submission, or score ledger was modified.

## Coverage

- Active-stage snapshot: 16 ONNX files, with a SHA-256 recorded for each file
  in `scan.json`.
- Original optimizer matrix: the same 21 pass sets used by
  `root_stage_fusion_202`, giving 336 task/pass profiles.
- Additional cleanup matrix: dead-code elimination, duplicate/unused
  initializer cleanup, no-op elimination, and a combined fixed-point cleanup,
  giving 64 profiles.
- Total: 400 profiles.
- Protobuf changed: 201 profiles.
- Changed and passed `onnx.checker.check_model(..., full_check=True)` plus
  strict/data-propagating shape inference: 169 profiles.
- Changed but rejected by the structural gate: 32 profiles. These were shape
  inference conflicts in existing custom/static shape annotations and were not
  emitted.
- Valid changed profiles with a different competition cost: 0.
- Strict-lower candidates emitted: 0.

The valid transformed models all retained exactly the same measured
`memory + params` cost as their staged parent. Consequently, there was no
strict-lower candidate on which to run known/fresh four-configuration raw
equality, error/non-finite, or inherited mismatch-policy gates.

## Cleanup observations

The independent structural scan found no unused initializer and no identical
initializer alias group in any of the 16 models. Several graphs contain
multi-output nodes with an unconsumed output name, but eliminating dead ends
did not reduce measured competition cost. The no-op pass family likewise
produced no strict-lower result.

Machine-readable inputs, pass lists, per-profile costs, failures, hashes, and
the zero-winner result are in `scan.json`. The reproducible scanner is
`scan.py`.
