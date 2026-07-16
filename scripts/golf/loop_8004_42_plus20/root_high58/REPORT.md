# Mid-cost history pre-screen 58 and task323 exhaustive adoption

Eight additional 8005.16 members were screened. One candidate is admitted:
**task323 cost106→104**, projected gain **+0.01904819497069441**.

The archived task323 lead was initially rejected because it retains a
56-input final Einsum and placed one selector exactly on the zero boundary.
The admitted rebuild moves the sensitive coefficient ten float32 ULPs toward
the safe negative side without changing parameter count. This is accepted
under the user's guaranteed-private exception because the decoded generator
has a finite complete support: fixed size13 and all 13×13 seed placements.

- Candidate SHA-256:
  `db773b15ceea8c42fac7543b7b7e93e0fd56a73493c7a8122b587327544c5926`.
- Official-like full-known cost: 106→104 (memory8, params98→96).
- Known dual ORT: 172/172 in both modes, runtime errors0.
- Direct decoded-generator exhaustive audit: 169/169 under disabled/default
  ORT and 1/4 threads, nonfinite cells0.
- Minimum positive raw value: `1.0181422137066955e25`; minimum nonzero
  absolute raw value: `2.599362586050822e18`.
- Full checker, strict data propagation, static positive and truthful runtime
  shapes, standard domains, bias UB0, no lookup/scatter, no shape cloak.
- The 56-input contraction is inherited from the current 62-input design and
  is allowed only because all generator states are exhaustively proven and
  the former zero-boundary cancellation has been moved negative.

The other seven targets (288/217/257/207/246/335/253) have no retained
numeric lower history.

Evidence: `task323_robust_audit.json`, `history_lead_audit.json`,
`audit_task323_robust.py`, and `task323_cost104_robust_u10.onnx`.
