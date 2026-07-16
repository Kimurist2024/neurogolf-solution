# Private-zero guarantee audit: tasks 035/066/090/377

## Outcome

No historical candidate or realistic minimal rebuild meets the user's
private-zero guarantee exception. **Accepted models: 0; verified gain: +0.0.**
No root ZIP, CSV, score pointer, or `artifacts/handcrafted` file was changed.

Authority is `submission_base_8004.50.zip`. The complete machine-readable audit
is in `result.json`; the non-promoting reproduction script is
`audit_candidates.py`.

## Decoded generator rules

- **035 (`task_1f642eb9`)**: project every non-cyan border marker to its nearest
  cell in the solid cyan rectangle. The input-only reference agrees with all
  266 known examples.
- **066 (`task_2dd70a9a`)**: recover the generator's hidden orthogonal S/U path
  between the red start pair and green end pair, and recolor the hidden blue
  path cells green. The generated family uses two cyan anti-ambiguity markers
  (and a third for U). The manually specified test is a rotated tight-turn
  special case, so the executable input-only decoder is intentionally not
  claimed as a passing proof; no candidate reaches the behavioral gate anyway.
- **090 (`task_3eda0437`)**: find the unique maximum-area all-zero axis-aligned
  rectangle and paint it pink (6). The input-only reference agrees with all
  267 known examples.
- **377 (`task_eb5a1d5d`)**: read the colors of nested filled rectangles and
  render the sequence as an odd-sized concentric square. The input-only
  reference agrees with all 266 known examples.

## Candidate audit

The table uses the actual DISABLE_ALL profiler cost, not the misleading static
floor embedded in filenames. Gain is `ln(base/candidate)` and is only potential;
every row is rejected before admission.

| task | base | candidate costs | decisive evidence | verdict |
|---:|---:|---|---|---|
| 035 | 545 | 493 | SHA `9adddc6da5f90bfb4b9cf3658c9b9cff96ca5aacbda1ae928c348087a5022433`; dual ORT known **0/266**, runtime errors 0; potential +0.100276621 | reject: known 0% |
| 066 | 677 | 368, 582, 583, 636 | all four use 61-input Einsum; 368/582 also use `TfIdfVectorizer`; potential gains +0.609588335, +0.151200825, +0.149484087, +0.062472710 | reject: lookup/giant contraction |
| 090 | 1050 | 304, 348, 382, 400, 1221, 1004, 1016, 1054 | 304–400 use lookup and have 2–6 declared/runtime shape mismatches; 1004/1016 cannot build the truthful trace because duplicate node names make the traced model invalid; 1221/1054 are not cheaper and have 12–13 shape mismatches | reject: lookup/shape cloak/not cheaper |
| 377 | 409 | 481, 473, 431, 451, **408, 408**, 409, 409 | both cost-408 files (`9a8ac1bd...`, `a27549f6...`) fail the runtime trace with `{1,5,1} != {1,5,10}` buffer reuse; other variants are not cheaper and/or have the same false-shape failure | reject: shape cloak/default-runtime failure |

All 21 unique candidates pass full ONNX checker and strict data-propagating
symbolic inference; that is insufficient when declarations disagree with real
runtime shapes, a lookup/giant contraction is present, known correctness fails,
or the model is not actually cheaper. Conv-family bias UB count is zero for all
21 candidates.

## Minimal-rebuild assessment

- **035**: the prior exact integer-separation audit exhaustively checked all 64
  subsets of the six scaled decoder lanes over 94 unique known states. No lane
  is removable; the tempting 493 fold destroys the modulo-256 code.
- **066**: the compact archive models achieve their price only with prohibited
  lookup or a 61-input contraction. A conventional exact S/U directional path
  reconstruction cannot realistically fit below the already compact cost 677.
- **090**: all cheap visible heuristics have known fresh failures; the existing
  analytic maximum-rectangle model is above the incumbent and has false runtime
  declarations. No sound below-1050 rebuild is available.
- **377**: the algebraically exact one-parameter shave to 408 inherits false
  shapes and fails Default ORT. Repairing the declarations exposes much larger
  intermediates and loses the cost reduction.

Because no model passes the prerequisite structural, truthful-cost, complete-
known, and dual-ORT gates, no model can enter the required multiple-independent-
seed fresh-100% confirmation stage. This is a hard rejection, not an untested
private-zero promotion.
