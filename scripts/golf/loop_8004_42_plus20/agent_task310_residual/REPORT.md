# task310 residual exact audit

## Decision

**ADMIT as an authority raw-pass-through exact shave.**

The candidate reduces task310 from **501 to 491** without changing any audited
raw output bit:

- authority SHA-256:
  `4eed21efedf2b44e11d2bb748d383275d193144c3c0f8f9f55265c8639e6fdec`
- candidate SHA-256:
  `6ccf625a0dca41d5c9cb39ddb41c3756313f2a01ac95f38d70c880c677ccf858`
- authority cost: memory 194 + params 307 = **501**
- candidate cost: memory 194 + params 297 = **491**
- strict reduction: **10**
- projected score gain: `log(501 / 491) = 0.020161973290344318`

No root submission, ZIP, score ledger, staging area, or immutable authority
artifact was modified.

## Exact factor identity

The final contraction uses the same initializer `A2[2,2,2,2]` twice. Its
entries are exactly the four-bit even-parity indicator:

```text
A2[d,r,j,c] = 1  iff  d+r+j+c is even, else 0.
```

Let

```text
H = [[1,  1],
     [1, -1]]
W = [1/2, 1/2].
```

Then, for every binary index tuple,

```text
A2[d,r,j,c]
  = sum_k W[k] H[k,d] H[k,r] H[k,j] H[k,c]
  = (1 + (-1)^(d+r+j+c)) / 2.
```

The audit reconstructs all 16 `A2` entries and verifies bit-identical float32
values. Replacing both `A2` operands by the same shared `H` and `W` therefore
preserves the real multilinear expression for every possible model input.
All coefficients are exactly representable powers of two or signed integers.

The 16-element `A2` initializer is removed and replaced by `H` (4 elements)
plus `W` (2 elements), saving 10 parameters. Nodes and counted intermediate
tensors are unchanged; only the final Einsum expands from 31 to 39 operands.

## Generator and policy boundary

The generator `inputs/arc-gen-repo/tasks/task_c909285e.py` creates a 20–30
square periodic wire grid, draws a complete square perimeter of side 5–8, and
returns that square crop. The 501-cost authority uses a frequency heuristic,
six `TfIdfVectorizer` nodes, and a giant final contraction. Prior fresh evidence
found one legal failure among 10,000 cases: seed `202607149901`, case 4037,
with 132 differing cells.

This candidate does **not** claim to repair or approximate the true generator
rule. It is admissible only under the requested alternative of exact authority
raw pass-through. It preserves the same known generator failure and introduces
zero additional failures. The inherited lookup/giant-contraction lineage is
not presented as a complete-support implementation.

## Runtime equivalence gates

The authority and candidate were executed independently in all four required
CPU configurations:

1. `ORT_DISABLE_ALL`, threads 1
2. `ORT_DISABLE_ALL`, threads 4
3. `ORT_ENABLE_ALL`, threads 1
4. `ORT_ENABLE_ALL`, threads 4

Results in every configuration:

| Corpus | Raw bitwise equal | Candidate truth | Errors | Nonfinite | `(0,0.25)` | Max raw delta |
|---|---:|---:|---:|---:|---:|---:|
| known | 266/266 | 266/266 | 0 | 0 | 0 | 0 |
| seed 202607149901 | 5000/5000 | 4999/5000 | 0 | 0 | 0 | 0 |
| seed 202607149902 | 5000/5000 | 5000/5000 | 0 | 0 | 0 | 0 |

Thus the aggregate evidence is:

- known raw pass-through: **1,064/1,064 case-configs**;
- fresh raw pass-through: **40,000/40,000 case-configs**;
- maximum raw difference: exact `0.0`;
- candidate-only wrong cases: `0`;
- candidate runtime errors, nonfinite values, and near-positive values: `0`.

The one truth failure per seed-1 configuration is the authority's already
recorded case 4037, reproduced bit-for-bit by the candidate.

## Structural and cost gates

- full ONNX checker: pass;
- strict shape inference with data propagation: pass;
- direct runtime shape trace: 21/21 outputs traced, mismatches 0;
- truthful `[1,10,30,30]` input/output, no supplied `value_info`;
- finite initializers; standard domain only;
- banned ops, nested graphs, functions, sparse initializers: 0;
- Conv-family bias/UB findings: 0;
- official-compatible profile: memory 194, params 297, cost 491.

The pre-existing six `TfIdfVectorizer` nodes and giant Einsum remain. The
candidate adds no operator and no intermediate output, and its acceptance is
strictly scoped to the formal factor identity plus demonstrated authority raw
pass-through.

## Other residual paths

Relevant earlier and current searches were reconciled:

- all generic optimizer/fusion passes leave the 501 graph unchanged;
- the initializer-pair precontract scan finds no task310 pair;
- exact sparse reconstruction costs 553–711 for the useful sparse factors;
- absorbing `cap` globally into a shared `P0/P1/P2` costs 471 but changes the
  function; selective all-input-exact clones cost 531;
- the truthful selector rebuild costs 633;
- a single `count < 29` selector is disproved by 22 legal cases in a 50,000
  generator audit.

The parity factor is the only residual probe that is both exact and strictly
lower.

## Evidence

- `task310_exact_parity_factor.onnx`: isolated winner payload
- `build_parity_factor.py`: deterministic builder and tensor identity check
- `audit_candidate.py`: four-configuration known/fresh raw audit
- `audit.json`: complete machine-readable runtime evidence
- `result.json`: concise handoff record

