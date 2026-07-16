# task366 exact residual re-golf lane 217

## Outcome

**No candidate is admissible under the requested truthful-shape, error-free gate.**
The staged `others/71407/task366.onnx` and all root ledgers/submissions were left
unchanged. Projected admitted gain is `+0.0`.

The strongest diagnostic candidate is:

| model | memory | params | cost | gain vs staged | runtime errors | truthful runtime shapes | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| staged `8cdf40d1...` | 7622 | 363 | 7985 | — | 12 in the retained error-bearing fresh streams | no | immutable baseline |
| `task366_lowbit_no_oob.onnx` `0fb66f33...` | 7622 | 362 | 7984 | `ln(7985/7984)=+0.000125242658` | **0** | **no: 98 mismatches** | **REJECT** |

Thus even the strict-lower, error-free candidate still relies on the incumbent's
shape cloak and fails the explicit `truthful` requirement. The earlier independent
task366 rebuild audit established a metadata-truthful cost of **9465**, above both
7985 and 7984 (`agent_rebuild_high3/REPORT.md`). Local exact shaving cannot bridge
that 1481-cost gap.

## Exact reductions found

The rejected cost-7984 candidate is nevertheless generator-independent except for
the inherited shape metadata and the deliberate OOB repair:

1. Sixteen of the 21 log2 chains consume a fixed-width unsigned lowbit:
   14 are `x & (0-x)` and two are `(~x) & (x+1)`. Both expressions are always zero
   or a single power of two. `lowbit_structure_proof.json` checks all 21 chains and
   recognizes exactly those 16; the five unproved chains remain unchanged.
2. At those 16 sites, changing the private divisor from binary16
   `0.693359375` to `0.69287109375` permits the following `Round` to be removed.
   `exhaust_round_carrier.json` exhausts zero and every uint32 power of two under
   disable-all/default ORT with 1/4 threads. Forty-four binary16 divisors pass all
   four configurations; the candidate uses the closest passing value to ln(2).
   All powers that overflow binary16 to `+inf` retain the baseline integer carrier.
3. The scalar target `[16]` is reconstructed as the already-live
   `Shape(safe_name_105) == [15]` plus the already-live int64 one. This preserves the
   shape-inference barrier but removes one scalar parameter.
4. The two binary16 `Add(x,-1)` sites become `Sub(x,+1)`, reusing the existing
   positive-one initializer. `exhaust_add_sub_one.json` checks all 65,536 binary16
   bit patterns in four runtime configurations with zero raw-bit differences and
   removes another scalar parameter.
5. The incumbent throws when the first 8-wide Gather window contains index 16.
   A `Clip(indices,0,15)` using already-live bounds removes this inherited error.
   Its 32-byte output consumes the lowbit memory savings, leaving the final net
   reduction at one parameter.

The lower cost-7953 diagnostic without the Clip is exactly raw-equivalent to the
incumbent, but inherits the incumbent's 12 observed OOB errors and is also rejected.
Removing all 21 Round nodes is invalid: five sources are not structurally lowbits,
and the original-divisor no-Round carrier differs on 31 of the 32 nonzero uint32
powers in the exhaustive test.

## Validation of the cost-7984 diagnostic

- ONNX full checker, strict shape inference, strict `data_prop`, static-shape,
  banned-op, nested-graph, and standard-domain gates pass.
- Known scorer-convertible corpus: 255 cases x four configurations = 1,020 runs;
  all correct, zero errors, zero nonfinite outputs, raw-equal to the incumbent.
- Fresh generator: four independent seeds totaling 8,000 cases x four
  configurations = 32,000 candidate runs. Candidate errors/nonfinite outputs are
  zero; minimum truth rate is 97.866%, above policy90.
- The first two fresh streams intentionally retain three incumbent OOB cases.
  Across four configurations the incumbent has 12 errors; the candidate has zero
  and produces the correct grid on all three cases. On every case where the
  incumbent returns, candidate and incumbent raw outputs are identical.
- Official remeasurement: staged `memory=7622, params=363, cost=7985`; candidate
  `memory=7622, params=362, cost=7984`.
- Runtime value-info trace: 739 tensors inspected, 98 declared-vs-actual shape
  mismatches. This is the decisive rejection reason.

## Evidence

- `audit_no_oob.json`: cost, structure, known/fresh four-config results, error and
  runtime-shape audit.
- `lowbit_structure_proof.json`: fail-closed graph-pattern proof for the 16 edited
  log2 carriers.
- `exhaust_round_carrier.json`: uint32 lowbit carrier/divisor exhaustion.
- `exhaust_add_sub_one.json`: exhaustive binary16 Add/Sub identity.
- `build.json`, `costs_no_oob.json`: reproducible construction and scorer cost.
- `candidates/task366_lowbit_no_oob.onnx`: rejected diagnostic only; SHA-256
  `0fb66f3346ec31c0ed514fdbd4e0d6fa1f40e6c4affc61de40d065ef9208c388`.

