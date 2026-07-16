# clean-mid9 audit — tasks 089 / 002 / 191 / 088

## Outcome

No candidate is admissible. Projected gain is **+0.000000** and no ZIP, root
CSV, `best_score.json`, or handcrafted artifact was modified.

The exact 8004.50 archive members used as baselines are:

| task | baseline SHA-256 | campaign cost | private-zero catalog |
|---:|---|---:|---|
| 002 | `20f024826fc233dd1a90e6e10052627cc6ea2847d337a7210c5173b84a6a6eb8` | 1286 | no |
| 088 | `40ce4f7849f7e5ea07f181b3e947d7a9528feef6b29ed7e9d3e836c9515be4b3` | 902 | no |
| 089 | `8084c194af3721712a2d0340a9779ac0cd60a1b333ce9fe02b49afd20118ca48` | 1361 | no |
| 191 | `109928c1f7ec9fd2ca497bcc538bd0a7065cc5fda572855b0ac7a12299c3c115` | 897 | **yes (highest-risk, ×3)** |

Loose history inventory covered 2,407 paths and 148 SHA-distinct models: 29 for
task002, 39 for task088, 42 for task089, and 38 for task191. The retained ZIP
lineages deduplicate into those loose families. No candidate survived the
cheaper + strict/truthful + policy gate, so fresh candidate validation was not
run merely to rehabilitate a structurally invalid model.

The completed 148-row audit in `audit_results.json` confirms `accepted_count=0`.
Across rejected rows, the dominant blockers were 86 non-cheaper static floors,
61 runtime-shape mismatches, 38 default-ORT failures, 23 non-cheaper actual
profiles, and 17 giant-Einsum graphs (categories can overlap).

## Task decisions

### task002 — no sound cheaper graph

The generator explicitly yellows hidden pot interiors and then makes one
row-major `is_surrounded` pass. Two valid generator parameterizations produce
the same input but different outputs at `(1,1)` and `(1,2)`, independently
reproduced in `reference_audit.json`. Therefore no deterministic one-input ONNX
can implement the literal generator exactly.

The closest retained control is SHA
`15123ddd97b84e6975b4a1ff5066455a965e034d368a535ecdb30429a65f2d6c`,
static cost 1319, already above 1286, and it uses a 66-input Einsum. It fails
both the strict cost and no-giant-contraction gates.

### task088 — sound rule is above the current floor

The true rule finds the four marker-color rectangle corners, crops the interior
sprite, and recolors its nonzero cells to the marker color. The independent
reference passed both seeds 50/50.

The sound spec-derived graph is SHA
`de02351ede2d755b715cf48cd17d2a8f7eae69d80fd7650351e92c82fe36b405`,
cost 5412, far above 902. The smallest retained micro probe, SHA
`2ebd9b4b16809924f89aedc98f910b3dd83f9e10e63e7b4fd258c32a7f40be3f`,
is fresh-exact but profiles at cost 1026 on the campaign zero input and emits
multiple declared-versus-runtime shape warnings. It is neither cheaper nor
truthful.

### task089 — all sub-baseline appearances are annotation artifacts

The true rule learns each complete 3×3 sprite from its red/green marker and
stamps omitted copies, mirroring only red. The independent reference passed
both seeds 50/50. The sound spec-derived rebuild SHA
`c97fff30f5fa41cf8345791fbcd78b6ad0c0af4e6b25d9aed38258b952b6a683`
costs 2620, above 1361.

The three apparent history reductions are:

| SHA-256 | claimed static cost | rejection |
|---|---:|---|
| `33db6c4a4422d8b070388418c755f6f30fcebd775436692ed5854feecc2bc85e` | 1184 | unresolved/nontruthful CenterCropPad shapes |
| `36d0bb3602d0ff1a2f5333e4bc429cc0149a2d5c98407b8eb290afb1853cd630` | 1184 | unresolved/nontruthful CenterCropPad shapes |
| `6b0a5edccfdc4748db93bbd571fa79695dd863d88b54cda9d48fbb6dfc8681dc` | 1298 | same shape issue; full profile is not lower |

Re-annotating these graphs truthfully exposes the genuine working tensors and
removes the nominal saving; the sound rebuild is the appropriate repaired
control and cannot beat the current campaign cost.

### task191 — private-zero guarantee is impossible below 897

task191 is in the private-zero highest-risk catalog, so the 90% exception does
not apply. The decoded true rule needs all eight dihedral orientations of the
reference pattern over the 23×23 grid. The independent reference passed both
seeds 50/50, consistent with the earlier 2000-case proof.

The true-rule rebuild SHA
`2bbcc9a818a1e7212cc8f2d2a012beaeaaf9377d7e9e1e0db9bd527ec0bccdba`
costs 10829. Its documented architectural core alone is above 8,800 bytes, so
a strict truthful model cannot beat cost 897. Aggressive SHA
`93b0fc544bf0e1e3f901228cc1923c96197677a83f77fd2ecf1cda3ac4825470`
has nonstatic shapes and dynamic Conv-bias evidence; SHA
`7432cc3f4b517aadc84d9b3ae0b4c0fdbc55005fd3cba17a9e015e9e77f3aa44`
profiles at 3430 and fails default-ORT session creation. Neither can satisfy the
user's private-zero guarantee condition.

## Final decision

Accepted models: **0**. Aggregate gain: **0.0**. This lane should not be merged.
Machine-readable details are in `result.json`; supporting inventory, static
audits, runtime warning capture, and independent rule checks are adjacent.
