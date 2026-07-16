# task023 rank82 kernel optimization

## Result

No candidate met the requested empirical adoption threshold.  The authority
ZIP and all protected root files remain unchanged.

- Authority: `submission_base_8005.17.zip`
- Authority ZIP SHA-256:
  `c48fa65401a5bd26d3ed1c556eee8f85c0a2063db313be6b96c73e86159b0a04`
- Authority task023 SHA-256:
  `bd242d29ab9514b2432dce31e6df28dd67f00bf1bdcb54c8a00f28614f974fb0`
- Authority cost: 1622
- Candidate cost: 1541
- Potential score gain if eligible: `+0.051228399`
- Required fresh gate: two independent seeds, each at least 90%
- Winner: none
- Decision: `NO_ADOPTION`

The generator is non-injective, so this lane is explicitly an empirical
normal-policy search.  It does not claim a complete/private guarantee.

## Final independent comparison

The two final fresh streams use seeds `923023001` and `1023023001`, 5,000
examples each.  The rates below use exact QLinearConv uint8 saturation and
ORT TopK's deterministic index tie order.

| candidate | cost | known disabled/default | fresh seed 1 | fresh seed 2 | decision |
|---|---:|---:|---:|---:|---|
| clean1541 | 1541 | 266/266, 266/266 | 4403/5000 (88.06%) | 4410/5000 (88.20%) | reject |
| root coordinate2 | 1541 | 266/266, 266/266 | 4447/5000 (88.94%) | 4459/5000 (89.18%) | reject |
| rank82 integer | 1541 | 266/266, 266/266 | 4446/5000 (88.92%) | 4460/5000 (89.20%) | reject |

The existing root candidate was also measured on a separate three-seed set at
`4460/5000`, `4450/5000`, and `4462/5000` (89.20%, 89.00%, 89.24%).  The new
integer candidate's training-selection shards were 89.12%, 88.60%, 88.80%, and
88.60%.  The apparent one-shard improvements do not cross 90% robustly.

## Search methods

Only the 36-byte initializer `score_W_q` was changed.

1. Inspected the clean cost-1541 graph and both latest root coordinate models.
2. Implemented `structured_rank_search.py`, using real 0/255 QLinearConv
   clipping, TopK index tie order, structured worst-positive/hard-negative loss,
   all-positive/all-negative pair loss, three independent selection shards,
   and exact known gating after int8 quantization.
3. Continuous candidates that moved materially away from the known-safe basin
   failed the known 266/266 hard gate.  The only final gated model was the
   unchanged root control.
4. Implemented `ensemble_integer_search.py`, which chooses the globally best
   legal integer coordinate move across four generator shards.  Every proposed
   move preserves all known examples; the objective is lexicographic worst
   shard, aggregate right count, then low-tail rank margin.
5. The integer search produced three byte-distinct clean models but plateaued
   below 90%.  The best new file is
   `candidates/task023_rank82_integer_root_c2.onnx`, SHA-256
   `d9c9c5d471b34b6c35ffc6006d038b6e21ef3e91ae31145e85da8ca46ffff0e9`.

## Structural and cost evidence

All three audited files:

- pass full ONNX checking and strict shape inference with data propagation;
- have official-like cost 1541 (`memory=1103`, `params=438`);
- pass all 266 known examples with ORT optimizations disabled and default;
- have static positive shapes and truthful output shape `[1,10,30,30]`;
- are graph-byte-identical after zeroing only `score_W_q`;
- retain `score_W_q` shape `[1,1,6,6]`, exactly 36 bytes;
- use QLinearConv with eight inputs, hence no optional bias (`UB=0`);
- contain no banned op, lookup op, custom domain, giant Einsum, nested graph,
  PRIVATE_ZERO table, or shape/archive cloak;
- complete runtime smoke inference with zero errors.

Because the independent fresh screen already fails the required threshold,
the candidates were not advanced to the expensive four-configuration fresh
runtime gate.  This avoids presenting partial structural success as an
eligible adoption.

Evidence:

- `audit.json`: final cost, dual-known, two-seed fresh, shape, structure audit.
- `integer_search_report.json`: four-shard global integer search history.
- `search_report.json`: structured rank-search result.
- `audit_finalists.py`, `ensemble_integer_search.py`, and
  `structured_rank_search.py`: reproducible Codex-written tools.

## Adoption decision

`winner = null`.  Do not merge any task023 file from this lane.  The cheaper
graph remains promising, but its best independently measured rate is below the
user-authorized 90% threshold.
