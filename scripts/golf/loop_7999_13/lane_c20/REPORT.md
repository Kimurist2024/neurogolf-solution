# Lane C20 — task133 / task349 exact-7999.13 audit

## Outcome

No candidate was promoted. The lane projects **+0.000000** score and leaves the
exact archive unchanged.

Both exact-archive incumbents are generator-risky, but every clean repair costs
more than the exact member. The final fresh5000 gate was therefore not reached:
there is no candidate that is simultaneously cheaper, known-exact in both ORT
modes, generator-sound, and runtime/static-shape consistent.

| Task | Exact cost | Cheapest clean generator-correct candidate | Gap | Verdict |
|---:|---:|---:|---:|---|
| 133 | 4403 | 5570 | +1167 | reject |
| 349 | 3964 | 4572 | +608 | reject |

## Task 133

The generator supplies one unscaled template sprite and one to three partial
sprites. A shared signature color locates each sprite; the output stamps the
full template at each partial sprite's recovered scale and body color.

The exact cost-4403 graph is not safe. It has 26 runtime/static shape
contradictions. On seeds 0..99 it achieved only 94 correct, 3 wrong, and 3
runtime errors in **each** ORT mode; the first runtime error is a 3x1 versus 3x4
buffer-shape conflict.

Historical nominal fixes cost 5337..5416 but still retain six shape
contradictions. The independent ground-up `agent_clean_rank.onnx` is the honest
reference: both ORT modes 267/267, full checker and strict data propagation,
zero runtime/static mismatches, fresh3000 exact, and fresh100 exact in both
modes. Its cost is 5570. Its truthful 900-byte label grid, two 390-byte rank
factors, and 384-byte GatherND index already account for 2064 bytes before the
remaining rule engine, so the 1167-byte gap was not safely closed.

## Task 349

The generator creates one to six maroon squares of side `2r`. Output keeps the
maroon core, paints its radius-`r` green halo, and paints blue downward beams,
with maroon over green over blue.

All five archived sub-3964 leads were re-audited. Cost 3547 cannot run because
ORT 1.24 has no int8 TopK kernel. Costs 3698 and 3710 are wrong on 77 and 68 of
267 stored examples respectively. Costs 3956 and 3954 pass all stored and
structural gates, but preserve the incumbent's fixture-signature halo patch.

The known false positive was reproduced independently in both ORT modes at
seed 349101, sequential valid case 421. The exact member and both 395x variants
have 12 differing one-hot entries. The general six-object OR rewrite
`agent_alt_exact_opt.onnx` is exact on that case, passes known 267/267 in both
modes, has zero shape mismatches, and passes 3001 fresh including six objects;
it costs 4572. A further 608-byte reduction would be required merely to tie the
unsafe incumbent. The retained exact-overlap audit shows the 600-byte affine
terminal feature tensor and general overlap state are the dominant remaining
floors; deleting them reintroduces visible overlap failures or fixture lookup.

## Structural policy audit

`candidate_audit.json` records actual scorer cost, complete stored fixtures in
both ORT modes, full checker, strict shape/data propagation, runtime/static
shape agreement, domains, functions, sparse initializers, banned operations,
lookup red flags, and Conv-bias safety. No winner uses a giant Einsum, lookup
memorization, shape cloak, nonstandard domain, function, sparse initializer,
banned operation, or unsafe Conv bias—because no winner was admitted.

## Deliverables

- `winner_manifest.json`: empty, gain 0
- `rejected_manifest.json`: candidate-level rejection evidence
- `candidate_audit.json`: actual-cost and structural audit
- `fresh_counterexamples.json`: dual-ORT defect reproduction
- `fresh_evidence.json`: generator and fresh summary
- `historical_scan_summary.json`: historical-family conclusions
- `validation/root_integrity.json`: unchanged root evidence

No root ZIP, CSV, score ledger, or handcrafted ONNX was edited by C20.
