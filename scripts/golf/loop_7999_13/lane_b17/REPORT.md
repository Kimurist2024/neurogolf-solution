# B17 — task280 / task396 strict sound audit

## Outcome

Winner count **0**; valid cost gain **0**; valid score gain **0.0**. B17 did
not modify any root ZIP, CSV, best-score state, promoted artifact, or
`artifacts/handcrafted` model.

## Generator truth

- task280 / `b527c5c6`: two red side dots identify the outward directions of
  two green rectangles. Each dot emits a red centreline and a green band of
  radius `short_side-1` to the grid boundary. Flip and transpose preserve the
  rule. This needs exact emitter classification and full-distance rendering.
- task396 / `fcb5c309`: select the uniquely widest-and-tallest box, crop it,
  and recolor its border plus retained interior/static pixels with the other
  nonzero color. Correctness needs same-color box geometry; global frequency
  and all-nonzero-run shortcuts are not equivalent under random static.

## Exact 7999.13 audit

- task280: cost **828**, SHA `19d38b7bff083fd7da14262714afa75345594b91e39d978ada0d25a912971793`. ORT_DISABLE_ALL is
  267/267, but default ORT cannot create the graph. The model has a 24-input
  Einsum and **22** declared/runtime
  shape mismatches. It is prohibited independently of its low cost.
- task396: cost **1019**, SHA `ce0bd7c49e11cbde341756993a71618c5c0bf8e086de6caf56ad93e8588e1d94`. Known examples are
  266/266 in both ORT modes and runtime shapes are truthful, but the same fresh
  5000 cases produce **4954/5000** in each mode. This reproduces the
  private-black processing risk; bitwise-equivalent shaving cannot repair it.

## Task280 truthful rebuild

The B17 rebuild changes only the four carrier declarations of the
generator-derived `cand_pad20` graph to their real `[1,4,30]` runtime shapes.
It is bitwise-equivalent, max Einsum input count 10, both known modes 267/267,
both fresh modes 5000/5000, and has no shape cloak. Its SHA is
`7922cea4b2789ed175357f6c5e3855b4b19521ea90bd6dbd2c24abd5f2373b7c`. Truthful actual cost is **2161**, which is **1333 above**
the cost-828 comparator. The apparent cost-1209 form still had four cloak
tensors; the cost-884 form had 22 cloak tensors and a 21-input Einsum.

## Task396 cheap candidates

Every strictly cheaper known-correct, truthful candidate failed the mandatory
fresh5000 gate:

- cost 947, SHA `95b41e2deca620c011cb5af28ccb0741bd78d3bd1033740d0e7e074ad873ff46`: 4890/5000, wrong 110, errors 0 (default ORT produced the identical count).
- cost 961, SHA `43bab65ff82dbbf377cad0160a7650ea2554ac6c3beb219ef1825f521b6fd55d`: 4861/5000, wrong 139, errors 0 (default ORT produced the identical count).
- cost 964, SHA `08bac1c184449b5438a2b68548fd74c901eac7fba66f5cd85c10fe050a124b25`: 4908/5000, wrong 92, errors 0 (default ORT produced the identical count).
- cost 965, SHA `13a3892a52d0553c038ced7cfb548cd5f2f62eedf5e34a6b44ffa1e9dee55b3d`: 4896/5000, wrong 104, errors 0 (default ORT produced the identical count).
- cost 965, SHA `cda9acdf5168445fca48926393f0f89516d800cf14f432da1e0d04b2193980dc`: 4893/5000, wrong 107, errors 0 (default ORT produced the identical count).
- cost 1014, SHA `60a16bc916be3fc38cc9312a3c01fcc1bd60343aa0731e9c33bc46f223a311a4`: 4963/5000, wrong 37, errors 0 (default ORT produced the identical count).

The best failure rate is the cost-1014 compact occupancy graph at 4963/5000;
it still has 37 deterministic wrong outputs per ORT mode and cannot be adopted.

## Task396 SOUND control

The generator-derived corner parser SHA `f1bddd36f0c0b943fe84d500bb629159b3639997bf7ea4b2e39eb2aa2bc9da2b` is truthful,
known 266/266 in both modes, and fresh 5000/5000 in both modes. Actual cost is
**1245**, or **226 above** the unsound cost-1019 baseline. This establishes the
measured safe floor reached by the existing same-color geometry formulation.

## Final gates

No strictly cheaper candidate survived fresh5000 dual ORT, so there was no
finalist for the external validator. Full checker, strict shape inference,
standard-domain, no sparse/nested/function, banned-op, finite-initializer,
Conv-bias, and max-Einsum checks are recorded per candidate in
`candidate_audit.json`.
