# High-cost history pre-screen 54

Eight additional members of the immutable 8005.16 baseline and every retained
numeric history lead were re-profiled with actual runtime shapes and screened
under both `ORT_DISABLE_ALL` and the default optimizer. Accepted: **0**;
projected gain: **+0.0**.

- task243: the only retained model is known-perfect, but actual cost is 626
  versus the baseline's 147.
- task162: all five retained models are known-perfect under
  `ORT_DISABLE_ALL`, but cost 828--839 versus the baseline's 146 and all fail
  the default-runtime known gate (0/266).
- task119: the only retained model costs 141 versus the baseline's 140 and is
  only 109/266 known.
- task180/task295/task074/task093: no retained numeric lower lead.
- task271: the cost-10 lead is 0/267 known. The cost-126 lead is cheaper than
  the baseline's 135 and is 267/267 under both known gates, but it uses nine
  `TfIdfVectorizer` lookup nodes. Its existing independent 5,000-case
  generator audit passed only 2/5,000 under each optimizer and produced 4,998
  runtime/output failures. It therefore fails the private-zero guarantee gate
  decisively and is rejected despite a theoretical +0.0689928715 score gain.

Evidence:

- `history_lead_audit.json`
- `task271_private_gate.json`
- `../../loop_7999_13/lane_archive_top200/task271_r02_dual5000.json`

