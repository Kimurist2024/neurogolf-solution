# Lane C39 — task343 95%-policy re-audit

## Outcome

The cost-172 r01 candidate is **adoption-eligible and recommended under the
user's explicit >=95% / runtime-error-0 policy**, but it is **not strict
100%-sound**.

| model | SHA-256 | cost | fresh correct | fresh wrong | runtime errors |
|---|---|---:|---:|---:|---:|
| baseline | `7d64c3eda1167f322d8981531e433e7195e54d48e16e29c771b52a379af17ab1` | 173 | 4963/5000 (99.26%) | 37 | 0 |
| r01 candidate | `6ada3c411cf90b4bcb42ff69e47eee35ed1c1b7d8b842c96c5c02c0eb06bec9e` | 172 | 4975/5000 (99.50%) | 25 | 0 |

The same generated corpus (generator `task_d8c310e9`, seed 343799445) was
used for both models and both ORT modes. ORT_DISABLE_ALL and default ORT gave
identical decisions on all 5000 cases for each model. This confirms the old
`4975/5000` record is exactly **25 ordinary wrong outputs and zero runtime
errors**, not a conflated execution-failure count.

The candidate improves observed generator accuracy by 0.24 percentage points
and reduces cost by one. The projected score gain is
`+0.005797117684327446`.

## Accuracy comparison

The models are not equivalent. In each ORT mode:

- both correct: 4938;
- baseline-only correct: 25;
- candidate-only correct: 37;
- both wrong: 0; and
- threshold outputs differ on 62/5000 cases.

Thus candidate errors do not merely form a subset of baseline errors, but its
net matched-corpus result is 12 more correct cases. Adoption is reasonable only
because the user explicitly permits >=95%; a strict generator-perfect policy
must still reject it.

## Structure and known cases

Both baseline and candidate pass full ONNX checker, strict inference, static
positive inferred shapes, standard-domain/no-banned-op checks, and actual
runtime-shape tracing. Candidate declared/actual shape mismatches are zero and
its measured intermediate memory exactly equals scored memory (140 bytes), so
there is no shape cloak.

The candidate has no functions, sparse initializers, nested graphs, giant
Einsum, or lookup-table operator. Its only Gather uses a Mod-generated periodic
coordinate vector to remap the input width; it is not an output lookup. Each
Conv reuses the input as its weight for scalar self-correlation and supplies no
bias input, so there is no short-bias or out-of-bounds bias dependency.

Both models pass all 266 known cases under both ORT modes with zero wrong and
zero errors.

## External validator

The moved validator was run for seeds 80004605, 80004606, and 80004607, 500
arbitrary cases each. All 1500 cases were executable; there were zero
both-failed or one-failed skips. Candidate and baseline differ on 838/1500
threshold outputs. These probes intentionally use arbitrary grids outside the
task generator domain, so the disagreement establishes non-equivalence and
runtime robustness, not generator-gold accuracy.

## Decision split

- Adoption-eligible under user >=95% policy: **yes**.
- Recommended under that user policy: **yes**; it is cheaper and higher on the
  matched generator sample, with no runtime errors.
- Recommended as a strict SOUND replacement: **no**; 25/5000 generator cases
  remain wrong.

No shared ZIP, score ledger, CSV, or aggregate file was modified.
