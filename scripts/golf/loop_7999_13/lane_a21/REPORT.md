# A21 — task285 / task286 strict sound search

## Result

No candidate is safe to adopt. Projected gain is **+0.000000**. The exact
`submission_base_7999.13.zip` remains unchanged; no root ZIP, CSV, score
ledger, or handcrafted artifact was modified.

Exact archive SHA-256:
`a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`.
Recomputed actual costs are task285 = **8717** and task286 = **7561**.

## True rules

The task285 generator creates one to three separated rooted creatures. Each
has a 2x2 color hub: input shows one complete quadrant and only the root cells
of the other nonzero quadrants. Output reflects the shown creature into all
four nonzero quadrants and recolors each copy with its hub color.

The task286 generator treats cyan as walls. All other cells are passable; the
non-black colors seed a cardinal flood. The reached component is filled using
the two seed colors selected by absolute `(row + column) % 2`; disconnected
black corridors remain black.

The independent NumPy references matched 265/265 stored cases and a new
3000/3000 generator stream for each task. Task286's new stream reached an
observed masked-flood depth of 60, confirming that a short sampled schedule is
not a sound replacement.

## task285

All eight archive SHAs were audited. Actual costs were 8767, 8732,
unsupported, 8845, 8862, 8887, 8896, and 9036, so none improves the exact
8717 member. The unsupported model uses TopK opset 11. The other history
models retain fixture-specific `target_rows/target_keep_bits` logic or
heuristic row beams and declared/runtime shape mismatches; several also fail
all 265 known cases at runtime under default ORT.

The clean specification-derived bitboard rebuild was independently remeasured:

- SHA `1f1bd10f2f2b5480468e73cdab2330f7f8447d5ddb294178601d364d0bbd2b15`;
- actual cost 14699;
- known 265/265 in both ORT modes, zero runtime errors;
- full checker, strict shape/data propagation, standard domain, no lookup,
  no Conv-bias issue, and zero declared/runtime shape mismatches.

It is the sound implementation, but is 5982 cost worse than the exact base.
The detector and rule have extensive prior exhaustive/fresh evidence; a new
dual5000 model run cannot make this cost regression adoptable.

## task286

The four cheaper history models are:

| candidate | actual cost | possible gain | decision |
|---|---:|---:|---|
| r01 | 7122 | +0.059814871721 | reject |
| r02 | 7263 | +0.040210489952 | reject |
| r03 | 7304 | +0.034581313283 | reject |
| r04 | 7497 | +0.008500516055 | reject |

All four pass 265/265 known examples in both ORT modes and have static runtime
shapes. Nevertheless, they all retain public-fixture `rcorr`/`ex_rcorr`
signature corrections around a finite-depth flood. This is a prohibited
lookup lineage, not a compiler of the generator rule. Decisive independent
fresh evidence already gives r01 **4294/5000 (85.88%)** and r02
**4612/5000 (92.24%)**. Per the no-burn instruction, r03/r04 were not sent
through another expensive 5000-case run merely to test the same invalid
family.

The lowest-cost correction-free specification implementation is the physical
row-bitset rebuild:

- SHA `a70c361b2583d65dbdecfec89707c9a94f8c35102159510bf1c006e99fcd334f`;
- actual cost 54552;
- known 265/265 in both ORT modes, zero runtime errors;
- full checker, strict static shapes, standard domain, no lookup/cloak/UB,
  zero declared/runtime shape mismatches.

Independent structurally different sound implementations cost 432364 and
4042393. Thus the verified sound floor found here is 46991 above the exact
base, while every sub-base graph is the rejected correction-table lineage.

## Fresh gate

Final fresh5000 under both ORT modes is required only after a candidate is
strictly cheaper, structurally clean, lookup-free, shape-honest, and correct
on all known examples in both modes. No model passed those prerequisites, so
there is no final fresh candidate and no winner.

Machine-readable evidence is in `model_manifest.json`, `audit_rows.json`,
`rule_reference_evidence.json`, `fresh_evidence.json`, `audit.json`, and
`winner_manifest.json`.
