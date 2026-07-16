# High60 retained-history audit

Baseline: `submission_base_8005.16.zip`, SHA-256
`73073208ec8b5b296f1fc13300a7e4df871135f0a9edd4a682ac7b48077aba00`.

Targets: task353, task081, task111, task248, task266, task375, task231,
and task142. Accepted: **0**. Projected gain: **+0.0**.

The SHA-deduplicated retained archive inventory supplied eight candidates. Each
was repriced with the official-like runtime trace and checked on the complete
known corpus under ORT `DISABLE_ALL`; default-optimizer validation was also run
for every DISABLE_ALL-perfect model.

- task353: base actual cost 93. Both retained candidates are known-perfect in
  both modes (271/271), but reprice to 99 and 104. Neither is lower.
- task111: base actual cost 89. The five retained candidates reprice to
  96, 96, 108, 109, and 100. All are DISABLE_ALL 265/265, but all five fail
  every known case under the default optimizer because their declared/runtime
  shape assumptions are not stable. They are also all more expensive.
- task231: the only strict cost lead is 59 versus 64, but it is 0/266 known.
- task081, task248, task266, task375, and task142 have no retained numeric
  lower lead.

No candidate reached the strict-lower plus known-perfect-dual precondition, so
structural admission and independent fresh testing were intentionally not
entered. No private-zero exception was used.

Primary evidence: `history_lead_audit.json`.

