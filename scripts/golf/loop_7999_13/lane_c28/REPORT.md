# C28 strict audit — task190 / task195

## Outcome

- Exact aggregate source: `submission_7999.13_wave15_candidate_meta.zip`
- Aggregate SHA-256: `0f106fa0d9599d4853397e0f9310e3ae1bcf47d6f418c6b9dec31e4a4490bc36`
- Tasks: 190 and 195
- Admissible winners: 0
- Verified score gain: `+0.000000`

No root ZIP, CSV, score pointer, or shared artifact was changed.  This lane only
wrote under `scripts/golf/loop_7999_13/lane_c28`.

## Generator-rule proof

The readable references in `audit_lane.py` were derived independently from
`task_7ddcd7ec.py`, `task_80af3007.py`, `common.py`, and the Sakana reference
programs.

| Task | True rule | Known | Fresh generator |
|---:|---|---:|---:|
| 190 | Detect the optional same-color diagonal markers around the unique 2x2 block and extend each selected corner ray to the 10x10 edge. | 266/266 | 5000/5000 |
| 195 | Recover the translated 3x3 binary gray sprite and emit its 9x9 Kronecker self-product. | 265/265 | 5000/5000 |

Both references had zero mismatches.  This verifies the rules used for the
optimization decision; it is not an example-fit interpretation.

## task190

The exact member has SHA-256
`7f5f1cd6e9bb3158db6a4f15d25327c904e38a406710c7a28e7de58c1272a56e`
and cost 153 (memory 56, parameters 97).

It is not a source for a strict candidate.  The six-node graph first encodes
the whole input to an integer token, performs two `TfIdfVectorizer` table
lookups, and feeds the selected features to a 25-input final `Einsum`.  The
lookup and giant-Einsum gates reject this structure even though ONNX full
checking and strict shape inference accept its syntax.

The deduplicated repository history contains 36 exact-byte-distinct task190
models.  No screenable model is cheaper than 153.  After excluding lookup /
giant-Einsum models, the lowest static/runtime screen cost is 1656.  The
transparent generator-compiled Conv implementation is much higher still
(historically 8014), because even one truthful 10x10 spatial carrier consumes
100 bytes before direction detection, propagation, color recovery, and
parameters.  Thus there is no candidate below 153 that also removes the unsafe
lookup structure.

## task195

The exact member has SHA-256
`02ea0c97c9f63f58c7099c94a7bc2634eea9ea21bf69df9936a2b0f5f3e2d56c`
and cost 150 (memory 129, parameters 21).

All 21 initializer elements are live.  The known micro-shaves around the final
`ConvInteger` do not reduce cost: omitting its optional weight zero-point only
shrinks the file, while the bias-free `QLinearConv` form needs a new value-3
zero point and costs 151.  The minimum channel selector that reaches color 5 is
the existing `10 -> 2 -> 3 -> 1` chain; shorter chains select the wrong channel.
The deduplicated history contains 25 distinct models and no model below cost
150.

More importantly, the cost-150 member fails the current truthful-shape gate.
The mismatch is present in both ORT modes:

| Tensor | Declared | Runtime |
|---|---|---|
| `gn` | 1x1x1x1 | 1x10x30x30 |
| `q_u8` | 1x1x1x1 | 1x10x30x30 |
| `q2` | 1x2x1x1 | 1x2x30x30 |
| `q3` | 1x3x1x1 | 1x3x30x30 |
| `q1` | 1x1x1x1 | 1x1x30x30 |

Consequently a derived rewrite would retain a shape cloak unless rebuilt from
the generator rule.  The clean spec-derived build costs 2013, so it cannot beat
150.  No rewrite was admitted under the no-new-shape-cloak/no-UB rule.

## Admission disposition

Neither task produced a candidate that is simultaneously strictly cheaper,
generator-complete, and structurally admissible.  Candidate fresh 5000/5000
and external-cost admission were therefore not run: there is no eligible model
to test.  `winner_manifest.json` intentionally contains an empty `winners`
array.  Full machine-readable evidence is in `audit.json`.
