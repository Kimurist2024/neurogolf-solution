# Incremental loose-ONNX rescreen against 8009.46

The current loop contains 2,188 loose ONNX observations and 1,224 unique
non-authority `(task, SHA-256)` payloads.  Every model was measured in an
isolated child process so malformed/profile-crashing graphs could not abort the
inventory.  Of 871 successfully profiled SHAs, 339 are strict-lower; 19 are
the already staged `others/71407` files and 320 are unstaged historical or
rejected variants.

This lane found no new adoption candidate.  Of the 320 unstaged lower SHAs,
305 resolve directly to an owning lane `REPORT.md`.  The remaining 15 are not
unreviewed winners:

- eleven task192 threshold variants are cost 1138 and are superseded by the
  staged, independently audited cost-1134 task192 payload;
- two task023 coordinate rankers are covered by the later 50,000-case
  POLICY90 audit and remain below 90%;
- the task205 cost-1038 and alternate cost-1041 SHAs are covered by the
  task205 private-zero proof.  The 1038 lineage lacks the staged model's
  all-valid-input proof, and the alternate 1041 is superseded by the staged
  exact SHA.

The 78 graphs whose official profiler returned a negative component are
recorded as measurement failures, not as lower-cost models.  No root archive,
ledger, or stage file was changed by this rescreen.

Evidence:

- `scan.json`: isolated official costs and all strict-lower rows;
- `classification.json`: nearest owning report and the 15 explicit exceptions;
- `scan.py`, `measure_one.py`, `classify.py`: reproducible inventory code.
