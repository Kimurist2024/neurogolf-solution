# task349 exact-equivalence golf on LB 8008.14

## Result

- Authority payload SHA-256: `f7ec94cbd38b44979a254b26f7cf670dbada27fffd954bf7e12554a6aa08cd7c`
- Candidate SHA-256: `179bbed5bd313a1f6ec62f573fd725ab71ff55a9509daaceff3f40274ac514c7`
- Official cost: `3564 -> 3556`
- Projected gain: `+0.0022471919569046644`
- Decision: `LB_PROBE_EXACT_EQUIVALENCE`; no protected root was modified.

## All-input proof

1. `radius_code` is the result of integer `Mod 11`, so its full legal index
   domain is exactly 0 through 10.
2. Exhaustive inspection of all 11 table rows proves
   `top_offset[i] = hstart_offset[i] + hend_offset[i] - 1`.
3. The candidate removes the 11-element top table, stores `hend-1`, derives
   top by addition, then restores the original hend contribution before every
   other consumer.  This is an integer identity, not a sampled approximation.
4. A `[1,1,1,1]` int32 zero used by `Where` is replaced by the existing scalar
   int32 zero.  ONNX multidirectional broadcasting makes the result identical.
5. `h_patch_sigs[4]` equals the removed special scalar `214431744`; the
   existing int64 `[4]` initializer gathers the same boolean from the already
   computed exhaustive equality mask.

These rewrites preserve raw model output for every input on which the authority
is defined.  They do not attempt to repair the authority's generator misses;
they preserve its LB-white behavior exactly.

## Runtime evidence

- Full ONNX checker: pass.
- Strict shape inference with data propagation: pass.
- Known examples: official scorer correct; authority/candidate masks equal.
- Fresh raw equality: two independent 5,000-case seeds in both disabled and
  default ORT modes, total `20,000/20,000`, runtime errors 0.
- Candidate gold counts exactly match the authority in every mode/seed.
- Margin stable, minimum nonzero absolute output `2.0`.
- Standard opset18 domain; functions 0; sparse initializers 0; nonfinite
  initializers 0; Conv-family short-bias mines 0.

## Evidence

- `relation_zero_sig_result.json`: cost, known, margin, and proof summary.
- `final_equiv_seed_800814349.json` and
  `final_equiv_seed_349800814.json`: dual-mode raw-equality audits.
- `build_relation.py`, `build_zero_reuse.py`, and `build_sig_reuse.py`:
  reproducible exact rewrites.
- `build_relation_sum.py`: rejected experiment; ONNX `Sum` does not support
  int8 in this runtime and emitted no candidate.
- `build_log_tables.py`: proves both remaining offset tables are derivable from
  the fixed power-of-two shift table and is known-exact, but runtime memory
  raises cost to3588.  It is a measured regression and is not the winner.
