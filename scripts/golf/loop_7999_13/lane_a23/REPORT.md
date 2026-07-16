# A23 task044/task205 strict audit

## Outcome

No candidate is adoptable. The exact `7999.13` score therefore remains
unchanged and this lane contributes `+0.0`.

| task | exact base cost | full-history coverage | best model reaching fresh | result |
|---:|---:|---:|---:|---|
| 044 | 1087 | 112 unique SHA | none | no cheaper valid model; non-injective rule |
| 205 | 1042 | 237 unique SHA | 937 | 4904/5000 in both ORT modes; reject |

The base archive is the exact immutable `submission_base_7999.13.zip`, SHA-256
`a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`.
Exact task payload hashes are `d4cd7b...54d4` for task044 and
`8a6acd...1468` for task205.

## task044

The generator creates two gray containers with connected black creatures in
their interiors, places translated copies of those creatures outside the
containers in two colors, and adds unrelated dust of a third color. The output
erases the two external copies and paints their colors into the matching holes.

The prior exact-base full-tree scan covered 112 unique task044 SHA values:
106 were rejected by a static cost floor at or above 1087, one was the exact
base, three failed structure, and two were unscorable. No model reached actual
scoring below the exact base.

The generator is also non-injective. Exhaustive latent parsing found legal
parameterizations with byte-identical input and different output. A rerun of
the exact translated-shape reference over per-seed 0..2999 passed visible data
but reported ambiguous legal inputs at seeds 1503 and 1506. Thus no deterministic
ONNX has a generator-entailed total tie rule. The ground-up autocorrelation
reference costs 1702, has truthful-shape mismatches, and is still not exact.

## task205

The rule is a solid 6..10 by 6..10 rectangle in a random 15..30 grid, with one
to three same-color interior markers in distinct rows and columns. The output
moves the rectangle to the origin and paints every marker row and column.
Sound localization needs 2-D perimeter/defect evidence; low-cost 1-D marginal
models confuse random background alignments with the rectangle.

All nine lower-cost history frontier models were reprofiled, full-checked,
strict-inferred, Conv-bias audited, and run on all 266 known cases under both
`ORT_DISABLE_ALL` and default optimization:

| cost | SHA prefix | terminal result |
|---:|---|---|
| 778 | `47c524f5` | 12 runtime/declaration shape mismatches |
| 937 | `bbfa8f5b` | reached fresh; 96/5000 wrong in each mode |
| 951 | `e49f95d9` | giant 20-input Einsum |
| 965 | `bfe47720` | giant 20-input Einsum |
| 997 | `af4ece09` | giant 20-input Einsum |
| 1010 | `a750a7b8` | documented quarantined private-zero lineage |
| 1015 | `14de50b9` | documented quarantined private-zero lineage |
| 1036 | `372d16d3` | 266/266 runtime errors in each ORT mode plus shape mismatch |
| 1038 | `43c963c4` | reached fresh; 72/5000 wrong in each mode |

The independent fresh run used seed `93023205`, generated exactly 5000 valid
instances with no generation/conversion failures, and compared directly to
generator gold. Cost 937 scored 4904/5000 in both modes. Cost 1038 scored
4928/5000 in both modes and was decoded-identical to the exact base on all
5000 cases, proving that its four-cost shave preserves every baseline error.
Both candidates had zero runtime errors but fail the required 100% correctness.

The explicitly excluded historical 97.8% model is SHA
`90ddee72b4e7d5e6b0dbf7f700ea719bf9ad882da261817da4f216873d4e3937`,
cost 1517, with prior fresh result 4891/5000. It is both more expensive than
the exact base and unsound, so it was not reconsidered for adoption.

The standard dense exact-rule references are sound but cost 83602
(`97b256...b4a`) and 484639 (`28105b...745c`). They fail the strict cost gate
before finalist validation. The compact cost-977 rebuild uses a forbidden
20-input Einsum and has a recorded 7/1000 fresh failure, so it is also rejected.

## Gate conclusion

There are zero finalists after full checker, strict inference, truthful runtime
shape, standard-domain, dense/no-lookup, no-giant-Einsum, no-UB, known-both-ORT,
and fresh-both-ORT gates. The external team validator was therefore not invoked;
its status is `not_applicable_no_finalist`.

No root ZIP, CSV, ledger, handcrafted model, or root artifact was modified.
Evidence is in `full_history_inventory.json`, `audit_rows.json`,
`fresh_dual_5000.json`, `task044_rule_evidence.json`, and
`winner_manifest.json`.
