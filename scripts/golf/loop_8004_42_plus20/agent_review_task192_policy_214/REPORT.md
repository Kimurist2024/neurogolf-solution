# task192 lane 211 independent fail-closed review

## Decisions

### Active POLICY90 shave: **PASS**

`task192_policy90_center_direct.onnx` is an all-legal-input raw pass-through of
the staged POLICY90 parent. It reduces the measured profile from
`memory 200 + params 938 = cost 1138` to
`memory 200 + params 934 = cost 1134` (`-4`, projected score gain
`ln(1138/1134) = 0.0035211303985789606`).

- parent SHA-256:
  `e6515b2ddf32c2eb80581aa3267e24683d2aa53d9445483b2a2a0752f94072d5`
- candidate SHA-256:
  `1200fe8473c045ec89abaaf1860d1d0758316523855c9ff13d4c3fc092412047`
- classification: `PASS_POLICY90_INHERITED_RAW_PASS_THROUGH`

This is not a claim that the inherited count-33 HardSigmoid selector is exact
on the entire generator support. It is a proof that the `-4` coefficient shave
adds no approximation or counterexample beyond that already admitted parent.
The one fresh miss described below is shared by both byte-identical outputs.

### Exact ArgMax fallback: **PASS as fallback replacement**

`task192_center_direct_argmax.onnx` is an exact raw pass-through of the old
exact fallback and is strictly cheaper:

- old exact fallback: `memory 208 + params 941 = cost 1149`, SHA-256
  `19fbdce89a5c89f5ff376b2fbbdb630ead5535d5ed5ebe7d9914a4de89e5023c`
- reviewed exact fallback: `memory 208 + params 935 = cost 1143`, SHA-256
  `5c5eaefa81acce481dbc93855dbcc2f9ef821e055f8c982eadcd07f63c764a9d`
- reduction: `-6`, fallback-only gain `ln(1149/1143) = 0.005235614053944943`
- classification: `PASS_EXACT_FALLBACK_REPLACEMENT`

The cost-1143 fallback is suitable to replace the cost-1149 fallback file. It
does not supersede the active cost-1134 POLICY90 model.

## All-input algebraic proof

The parent basis is `[nonzero, background, selected]`. The candidate basis is
`[inside, nonzero, selected]`, with `inside = nonzero + background` for every
legal zero-hot/one-hot input cell. The changed factors are exact identities:

```text
center:
  [nonzero + background, nonzero] = [inside, nonzero]

neighbor:
  [nonzero + background, selected] = [inside, selected]

route:
  [background, -9*background + selected]
  = [inside - nonzero, -9*inside + 9*nonzero + selected]

histogram:
  input · nonzero
  = (input · [inside, nonzero]) · [0, 1]
```

The review enumerated all `2^10 = 1024` binary selected vectors, including
multi-hot vectors that the HardSigmoid can emit, against all 11 legal cell
states (zero-hot plus ten one-hot colors). All 33,792 factor/cell comparisons
were exact with zero mismatches. The identity is linear in `selected`, so the
enumeration is stronger than testing only the ten one-hot selections.

The active candidate preserves the parent HardSigmoid byte-for-byte at
`alpha=1, beta=-33`; the 30x30 adjacency initializer is protobuf-byte-identical;
and the final polynomial is unchanged. Histograms are integer counts at most
900, the selector is binary at integer inputs, and the remaining factors are
small integers, all exactly representable in float32. Consequently the basis
rewrite is a semantic pass-through for every legal one-hot/zero-hot input, not
an example-fit claim.

For the exact fallback, the histogram entries are the same ten values. The old
`ArgMax(axis=1)` on shape `[1,10]` and the new `ArgMax(axis=0)` on shape `[10]`
both use first-index tie breaking and produce shape `[1]`; their OneHot route,
depth 10, values `[0,1]`, adjacency, and final polynomial are unchanged. The
same coefficient proof therefore applies.

## Independent runtime verification

The review used different fresh seeds from lane 211:
`21419207` and `21419231`, 1,500 valid generated cases each. It tested both
model pairs under all four configurations:

- ORT optimizations disabled, threads 1
- ORT optimizations disabled, threads 4
- default ORT optimizations, threads 1
- default ORT optimizations, threads 4

Results:

| lane | known | fresh | raw pair comparisons | candidate truth | errors | nonfinite |
|---|---:|---:|---:|---:|---:|---:|
| active 1138 -> 1134 | 265/265 in every config | 2999/3000 in every config | 13,060/13,060 | 99.9667% fresh | 0 | 0 |
| exact 1149 -> 1143 | 265/265 in every config | 3000/3000 in every config | 13,060/13,060 | 100% fresh | 0 | 0 |

Both parents and candidates were raw-bitwise equal on every case in every
configuration. Thus there were 26,120 successful pair comparisons (52,240
model executions) and zero candidate-specific regressions. Active seed
`21419231` contains one inherited count-33 policy miss; parent and candidate
emit the same raw bytes for it. Active fresh accuracy remains above the user's
90% admission policy.

## Static, truthful-shape, and UB gates

Both reviewed candidates pass:

- ONNX full checker
- strict shape inference, with and without data propagation
- standard-domain opset 18 only
- static declared shapes
- functions / sparse initializers / nested graphs: `0 / 0 / 0`
- banned ops / Hardmax / unused initializers / nonfinite initializers: `0`
- Conv-family nodes: `0`, therefore short-bias UB findings: `0`
- typed runtime trace: active 4 outputs, exact 5 outputs; shape mismatches and
  nonfinite values: `0`

The immutable root submission ZIPs and `all_scores.csv` retained their expected
SHA-256 guards before and after the audit. This review did not edit the staged
model, lane 211, root submissions, or ledgers.

Machine-readable evidence: `audit.json`. Independent reproducer:
`audit_independent.py`.
