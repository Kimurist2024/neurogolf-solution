# Lane C18 — task070 exact scalar-fusion audit

## Outcome

**ACCEPT — relaxed95, base-equivalent tier.** The candidate reduces actual cost
from 83 to 75 for projected gain **+0.1013524942602875**. It is recorded as a
winner under the user's authorized >=95% criterion. No root ZIP, CSV, ledger,
or handcrafted model was modified.

| gate | ORT_DISABLE_ALL | default ORT |
|---|---:|---:|
| complete known | 266/266, 0 errors | 266/266, 0 errors |
| fresh | 4990/5000, 10 wrong, 0 errors | 4996/5000, 4 wrong, 0 errors |
| base/candidate decoded equal | 5000/5000 | 5000/5000 |
| base/candidate raw bitwise equal | 5000/5000 | 5000/5000 |

Fresh rates are 99.80% and 99.92%, both above the 95% threshold, with zero
runtime errors. The first generator-rule mismatches occur at zero-based case 64
for seed 7999130700 and case 209 for seed 7999130701. Their wrong raw logits
have magnitudes in the hundreds or thousands, so they are semantic blue/green
swaps rather than near-zero margin noise. This candidate is therefore not
classified as 100%-sound; its acceptance depends on the relaxed95 policy.

## Fusion and lookup assessment

The rewrite itself is algebraically exact:

```text
H[e] = sum(input[b,i,h,w] * R[r,i] * T[e,r])
```

The terminal `H[e]` operand is replaced with the same factors under fresh
dummy indices `input[b,n,u,v]`, `R[o,n]`, and `T[e,o]`. This is finite-sum
associativity/distributivity. The candidate equation and operand list match
that inline exactly, all five initializers are byte-identical to the base, and
only node `H` plus its stale `value_info` entry are removed.

Although the terminal Einsum has 17 operands, this mutation is not an
example-lookup expansion: it has only the original 75 scalar parameters, no
coordinate/example bank, and no Gather, Scatter, TopK, TfIdfVectorizer, or
Hardmax. The 10000/10000 raw-bitwise base/candidate equality independently
confirms exact runtime equivalence.

The inherited two-Einsum base is not a complete 100%-sound implementation of
the cyan-rectangle rule. Crucially, the fusion faithfully preserves it:
candidate and base are raw-bitwise identical on every one of 10000 fresh
evaluations, decoded-identical on all 10000, and have no one-sided runtime
errors. Thus the rewrite introduces no new semantic or runtime regression and
qualifies for the user-authorized >=95% tier.

## Cost and structure

- Base: `memory=8`, `params=75`, `cost=83`.
- Candidate: `memory=0`, `params=75`, `cost=75`.
- Accepted projected score gain: `ln(83/75) = 0.1013524942602875`.
- Full ONNX checker: PASS.
- Strict shape inference with `data_prop=True`: PASS.
- Static input/output: float32 `[1,10,30,30]`.
- Default-domain opset 18 only; no banned/Sequence/nested/function/sparse ops.
- No Conv-family nodes or bias findings; no nonfinite initializer or output.
- Known and 10000 fresh candidate executions produced zero runtime errors.

## Evidence

- `audit.json`: complete cost, structure, known, dual-ORT fresh, raw/decoded
  differential, margin, and algebraic-inline record.
- `counterexample_summary.json`: first fresh failures.
- `winner_manifest.json`: relaxed95 winner and complete gate evidence.
- `rejected_manifest.json`: empty rejection list with tier note.
- `validation/root_integrity.json`: immutable baseline and root-file hashes.

The exact baseline archive remained SHA-256
`a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`,
contains 400 entries, and passes `unzip -t`.
