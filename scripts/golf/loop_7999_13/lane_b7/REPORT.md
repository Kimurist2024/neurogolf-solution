# B7 exact one-node Einsum factor wave

## Outcome

No candidate was accepted. The exact `submission_base_7999.13.zip` baseline remains unchanged, with projected score **7999.13** and projected gain **+0.000000**.

All seven exact members are single output-direct Einsum graphs. Their charged intermediate memory is zero, so an improvement must strictly reduce unique initializer elements without introducing any counted node output. No such sound reduction was found.

## Immutable baseline

- Archive SHA-256: `a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`
- Root ZIP/CSV/ledger files modified by this lane: **no**
- All seven members pass ONNX full check, strict shape inference, and banned-op inspection.

| Task | ZIP index | Member SHA-256 | Memory | Params / cost |
|---:|---:|---|---:|---:|
| 030 | 27 | `a61cb73cc21fdfd78be445db81d4209060fcc61c8ed1a3f8f54fb48758ddf17d` | 0 | 162 |
| 132 | 118 | `2ac5f7927047f6a3670b172d2dd4d3bfc4d9c97e0aca34b5ef29ed3cfcf544ff` | 0 | 316 |
| 175 | 161 | `0979ba8969cdfd796f0c4e0c40c1ebf062d28093ab8866801bad9f504d537945` | 0 | 166 |
| 199 | 183 | `d236c732d0df80270154b8ee593e17768dd54fc8dcec4aac93e752474651383e` | 0 | 261 |
| 212 | 194 | `e3f20fe069499de6c8ab36eadb10e69802ab61c4c87eb98d9843c7a87869ad42` | 0 | 240 |
| 240 | 220 | `1ac586676b5ef226ead36bdaf92333f30b18a5ed53d9fdea1eb1036e3b692465` | 0 | 172 |
| 304 | 278 | `e395301e8b11cc06ce90b68e7ddfefd87ec003437431b484aa0c6b4f2f3b3f51` | 0 | 180 |

## Generator truth reviewed

- **030 / `1caeab9d`:** three colored copies share a mini-bitmap; output relocates every copy vertically to `megarows[0]` while retaining each horizontal placement and color.
- **132 / `56ff96f3`:** one or two non-overlapping rectangles are specified by opposite colored corners; output fills each complete rectangle with that color.
- **175 / `73251a56`:** a fixed 21×21 arithmetic color field is generated from `(row, col, mod, modset)`; rectangular black cutouts must be restored from the uncut symmetric field.
- **199 / `834ec97d`:** the unique non-yellow pixel moves down one row, while yellow is painted above it through the source row on columns matching the source-column parity.
- **212 / `8d510a79`:** blue/red pixels extend vertically toward or away from the gray horizon according to color and side, stopping at the grid edge or an occupied output cell.
- **240 / `9d9215db`:** the sparse seed bitmap is mirrored across both axes; qualifying diagonal/next-door pairs additionally generate symmetric dotted-square edges.
- **304 / `c3e719e8`:** every cell whose color equals the unique modal color selects a 3×3 output block containing the entire input pattern.

These rules were used to reject component removal or initializer sharing that only fits archived examples.

## Exact factor audit

`factor_audit.py` treats each float32 initializer value as an exact rational number, computes every tensor mode rank, enumerates all initializer-name subsets for exact static precontraction, and checks same-shape initializer identity.

- Positive-cost exact single-mode factorizations: **0**
- Positive-cost exact static precontractions: **0**
- Identical same-shape initializer pairs: **0**
- task030 `U` is exactly rank 2, but a 4×2 and 2×4 factorization costs 16 parameters, equal to the original 4×4 initializer.

The detailed reproducible inventory is in `factor_audit.json`.

## Historical lower-cost screen

The scan deduplicated 177 historical models by task and SHA-256. Static shape cost eliminated 143 models already at or above the exact baseline. All 34 genuinely cheaper models were then executed against complete known gold; none was correct.

| Task | Distinct models | Static-dominated | Cheaper executed | Cheaper known-complete |
|---:|---:|---:|---:|---:|
| 030 | 28 | 26 | 2 | 0 |
| 132 | 34 | 24 | 10 | 0 |
| 175 | 21 | 17 | 4 | 0 |
| 199 | 24 | 15 | 9 | 0 |
| 212 | 17 | 17 | 0 | 0 |
| 240 | 27 | 22 | 5 | 0 |
| 304 | 26 | 22 | 4 | 0 |

## Per-task floor findings

- **030:** the two cost-150 `AZ`/`Amap` sharing directions both fail known gold. These 4×3 tensors are non-identical and encode different coordinate maps. The only lower exact rank is `U`, whose factorization is cost-neutral.
- **132:** all ten lower-rank/component candidates fail known gold. `P`, `R`, and `S` are indispensable mixed-radix coordinate selectors. `PC` and `L` are related triangular bases but sharing them requires a larger common-basis representation than their current 36 total parameters.
- **175:** deleting any of the four polynomial/color features reduces cost to 129 but fails known gold. Every useful tensor mode is full exact rank; selector absorption duplicates more parameters than it removes.
- **199:** all five CP-component drops, all three latent drops, and the cost-243 rank-3 `P` approximation fail known gold. Exact rank of `P` is four, so the approximation is not eligible.
- **212:** no historical model has static cost below 240. The sound direct implementation with only 41 parameters costs 4431 because its counted intermediates dominate. All exact zero-memory tensor-network modes are already irreducible for positive cost.
- **240:** dropping any of the five shared CP components reduces cost to 154 but fails known gold. The singleton-broadcast factor is already represented by `A3`; the remaining rank-five components are all required.
- **304:** dropping any color-basis component reduces cost to 168 but fails known gold. `SF` and `SG` are different feature selectors (`constant+x` versus `constant+y`); sharing either one changes the generator rule, while adding a feature permutation costs more than the six saved parameters.

## Gate disposition

Strict admission requires known-complete correctness, fresh/domain 5000/5000, zero runtime errors, and zero structural/UB findings. No lower-cost candidate passed the prerequisite known and exact-factor gates, so fresh-5000 was intentionally not run on rejected models. `winner_manifest.json` is empty.
