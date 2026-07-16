# Wave3 true-rule rebuild: tasks 005, 080, 101, 133

## Outcome

No candidate meets every policy-90 gate while being strictly cheaper than the
exact `submission_base_8004.50.zip` member. `winner_manifest.json` is therefore
empty and the projected score gain is `+0.0`.

The final policy is: known 100%, fresh >=90%, runtime errors 0 in both
`ORT_DISABLE_ALL` and default ORT, full checker, strict shape inference with
data propagation, runtime/declaration shape agreement, actual scorer cost below
the exact base, Conv-family bias UB 0, and no lookup/private-zero/shape cloak/
giant-Einsum construction.

| task | base cost | base fresh (each ORT) | clean generator-rule model | verdict |
|---:|---:|---:|---:|---|
| 005 | 2325 | 1983/2000, errors 0 | 2545, 2000/2000 | no cheaper clean candidate |
| 080 | 3051 | 2000/2000, errors 0 | 3051, 2000/2000 | structural tie/floor |
| 101 | 5712 | 1991/2000, errors 0 | 7264, 2000/2000 | clean model +1552 |
| 133 | 4403 | 1940/2000, errors 34 | 5570, 2000/2000 | clean model +1167 |

All clean models are known-exact in both runtime modes: task005 266/266,
task080 231/231 after the scorer-contract >30 skips, task101 266/266, and
task133 267/267. Their output margins had no values in `(0, 0.25)`.

## Generator references

The generators and `common.py` were read before model work. Existing
spec-derived NumPy references were independently rerun:

- task005 (`045e512c`): 266/266 stored plus 8000/8000 fresh;
- task080 (`39e1d7f9`): 231/231 valid stored plus 2514/2514 valid fresh
  (486 generated 31x31 cases skipped exactly as the scorer does);
- task101 (`447fd412`): all stored plus 2000/2000 fresh;
- task133 (`57aa92db`): 267/267 stored plus 3000/3000 fresh.

The rules are respectively stride-4 sprite rays, line-grid motif completion,
exact-cover completion of red-marker magnified creatures, and reconstruction of
partially shown signature/body-color magnified creatures.

## Attempts and stopping reasons

### task005

The exact 2325 model clears the 90% accuracy rule, but its Hardmax routing is not
eligible as a newly adopted clean rebuild. The stored cost-2389 sound model is
also rejected by the explicit Hardmax plus 9-input Einsum policy. The compact
cost-2534 generator-exact model retains a 9-input Einsum and is likewise
rejected. The clean cost-2545 model uses two bounded 7-input compactness Einsums,
has no Hardmax, is dual-ORT 2000/2000, and passes the official Conv-bias checker,
but is 220 cost above base. Prior exact-base work tested all 48 optimizer passes,
113 deduplicated model variants, direct-output, no-seed-pad, ArgMax-color,
zero-sharing, and scalar-paint alternatives; none is strict-cheaper and clean.
The 900-byte label plus dynamic selector/paint state is the stopping floor.

### task080

The exact 3051 graph is already the generator-rule compiler. Its irreducible
terms include the 900-byte output label, three static-stride float color decodes
(800 bytes total), and the line-grid renderer. A 77-model/929-source scan found
the best nonbase candidate at 3053. All 48 optimizer passes, the independent
full rebuild (4868), spacing-selection, gather-render, dtype, and branch-removal
variants failed to improve. Stop at the measured structural floor.

### task101

The sound transform requires extracting the complete blue/red template,
matching magnifications 1/2/3, and exact-covering all remaining red markers
without mixing copies. The clean exact-cover model costs 7264 and is dual-ORT
2000/2000. Archive costs 5672 and 5688 are nominally below 5712 but contain
explicit fixture-signature probe branches; cost 5703 additionally comes from a
quarantined private-zero lineage. Those remain prohibited even after lowering
fresh acceptance to 90%. Dense/all-match, signed inverse, TopK list, greedy
selector, packed renderer, dtype, initializer, and node-shave branches were
already exhausted; the 1552 gap cannot be closed without dropping exact-cover.

### task133

The cost-4403 exact member has 26 runtime/declaration shape contradictions. On
this independent stream it produced 26 wrong cases and 34 runtime errors per
2000 in both ORT modes, so the runtime-zero gate rejects it regardless of its
97% combined correctness. Historical nominal repairs at 5337--5416 retain shape
contradictions or other generator defects. The independent clean rank-factor
rebuild costs 5570, is known-exact, dual-ORT fresh 2000/2000, and has no runtime
shape mismatch. Its unavoidable 900-byte label, two 390-byte rank factors, and
384-byte GatherND indices already consume 2064 cost before the rule engine.
Per-channel warp, ConvTranspose stamping, workspace shrinking, hidden-shape
repair, optimizer, pcolor detection, and boundary-clipping attempts cannot
bridge the 1167 gap. Stop at the clean structural floor.

## Structural evidence

`model_audit.json` records actual costs, full checker, strict data-propagating
shape inference, runtime/declaration shape comparison, domains, functions,
sparse initializers, banned/nested ops, lookup flags, known correctness, and
Conv-family findings. `fresh_gate.py` is the lane-local reproducible dual-ORT
known/fresh gate. Calling `scripts/golf/check_conv_bias.py::check_model` on all
retained baseline and clean candidate models returned `[]`.

No submission ZIP, score CSV, or score ledger was modified. During initial
measurement `try_candidate.py` copied exact baseline task005/task080/task133
members into the ignored `artifacts/handcrafted/` cache; the parent was notified
immediately. All later checks used the non-promoting auditor.
