# task226 cost372 deep re-golf

## Verdict

`SAFE_PRIVATE_ZERO`: promote
`candidates/task226_greater_cost370.onnx` as the strict-lower replacement for
the 8009.46 task226 authority.

- authority: memory 333 + params 39 = cost 372
- candidate: memory 331 + params 39 = cost 370
- score gain: `log(372/370) = 0.005390848634876373`
- candidate SHA-256:
  `aebca4b2e7c3ce5cb5663a6f8b88e428e9bcf53b3e9f1161728c7dd9c502389f`

## Rewrite

The authority computes two row conditions through four Boolean scalar
intermediates:

```text
r3_and_nr8 = Cast(r3_f) AND NOT Cast(r8_f)
r6_and_nr1 = Cast(r6_f) AND NOT Cast(r1_f)
```

Every official and generator input is one-hot.  These gathered background
channel values are therefore exactly `0.0` or `1.0`, giving the exhaustive
two-bit identity:

```text
bool(a) AND NOT bool(b) == (a > b),  a,b in {0.0,1.0}
```

The candidate replaces each Not+And pair with one `Greater` over the existing
float Gather outputs.  Two one-byte Boolean intermediates disappear; no
initializer or output renderer changes.

## Full-support proof and runtime gates

The generator has exactly 17 valid width compositions and 8 valid height
compositions.  All 136 Cartesian cases were executed under ORT disabled,
basic, extended, and fully enabled:

- 136/136 correct in every mode
- 136/136 raw-byte equal to the authority in every mode
- runtime errors 0; nonfinite values 0

Additional gates:

- known: 133/133 in all four modes, raw-equal to authority
- fresh: two independent 5,000-case streams, all correct in all four modes
- full checker and strict shape inference with data propagation: pass
- all 63 node outputs exposed: declared/runtime shapes match in all four modes
- standard domains only; banned ops, nested tricks, lookup/scatter, giant
  Einsum, huge fan-in, shape cloak, and Conv-family bias UB: none
- official scorer: cost370, correct
- team validator: valid, 133/133, no warnings or failures

## Other reductions investigated

The 136-state cell-query set-cover has an exact HiGHS optimum of 10 probes
(gap 0), equal to the existing four row plus six column probes.  Thus no probe,
Gather, Cast, or index parameter can be removed while preserving all states.

The 17-row column truth table was also re-synthesized with shared mux helpers.
One helper, two cost-1 helpers, cost-1 plus cost-2, and two cost-2 helpers all
bottomed out at the incumbent 25 scalar `Where` outputs.  The row token mux
network bottoms out at 14 vector muxes (28 bytes); the direct `Greater`
identity is what lowers it to 26 bytes.

A signed-int8 sparse QLinearConv rewrite would leave only eight nonzero weight
values algebraically, but sparse QLinearConv weights fail ONNX full/strict type
inference and were rejected.  Initializer ties and tied zero-point variants
failed finite feature separation.

## Artifacts

- `build_candidate.py`: deterministic authority extraction and rewrite
- `audit_final.py`: complete four-mode audit
- `audit/build.json`: build manifest
- `audit/search_summary.json`: re-golf search evidence
- `audit/final_audit.json`: complete machine-readable gate evidence
- `result.json`: promotion handoff

No root submission, score table, or other active stage was modified.
