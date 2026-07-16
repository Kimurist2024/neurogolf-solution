# High-cost history pre-screen 51

Eight additional 8005.16 members and all retained lower history leads were
re-profiled and complete-known screened. Accepted: **0**; projected gain:
**+0.0**.

- task355: five actual249 candidates miss 3–6 of 267 known cases.
- task174: known-perfect histories profile at actual240–278, above base238.
- task325/task247: no retained lower history.
- task042: actual193/201 leads score at most6/266 known.
- task143: actual148 is known266/266 dual, but contains TfIdf lookup plus a
  17-input giant Einsum and independently scores only2/5000 and3/5000 fresh.
- task079: known-perfect history only ties actual209 or costs212/229.
- task065: known-perfect histories cost202–205 versus base199.

Evidence: `history_lead_audit.json` and the independent task143 fresh audit at
`scripts/golf/loop_7999_13/lane_c11/fresh_audit.json`.
