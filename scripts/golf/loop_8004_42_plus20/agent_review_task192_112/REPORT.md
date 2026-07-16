# task192 selected-mask independent review

## Verdict

**PASS** — `task192_selected_masks.onnx` is a strict-lower, all-input exact
replacement for the immutable `submission_base_8008.14.zip` task192 member.
This lane performed no promotion and did not modify any submission, score, or
`others/` artifact.

| model | SHA-256 | memory | params | official cost |
|---|---|---:|---:|---:|
| immutable authority task192 | `e7f9a11b93b611acfa4bba39e90e1ddf24223d50add4277fe9716f21f6ede10c` | 88 | 1521 | 1609 |
| existing exact-poly control | `c3cbaf44d962ca72e15514da1b32c121ee489d153ef39d38b7101f09576e92b6` | 168 | 1139 | 1307 |
| selected-mask candidate | `40244ab462644481407ebb7200984dfdff1475c0d8e6ff731ba2d588ec92ea09` | 248 | 949 | **1197** |

The candidate saves 412 cost versus authority and 110 versus exact-poly.
The immutable ZIP SHA-256 was independently confirmed as
`50b3215030cf506f692af50e41203d805256a250bd29882cc777749767a350c6`.

## Factorization proof

For selected one-hot vector `s`, the removed relation tensor has
`relation[0,d,a] = 1` and `relation[1,d,a] = I[d,a]`. Therefore:

- `sum_a relation[0,d,a] * s[a] = sum_a s[a] = 1`
- `sum_a relation[1,d,a] * s[a] = s[d]`

The contraction is exactly `Concat(all_colors, selected)` with shape `[2,10]`.
The audit exhaustively checked all ten possible one-hot selections with zero
numeric difference, verified both relation uses were replaced by the same
`selected_masks` tensor, and confirmed all seven unaffected initializers were
byte/value equivalent.

## Runtime evidence

All 265 known examples passed in all four configurations:

| configuration | candidate | candidate vs exact-poly raw | threshold | runtime/nonfinite |
|---|---:|---:|---:|---:|
| disable-all, 1 thread | 265/265 | 265/265 | 265/265 | 0 / 0 |
| disable-all, 4 threads | 265/265 | 265/265 | 265/265 | 0 / 0 |
| default, 1 thread | 265/265 | 265/265 | 265/265 | 0 / 0 |
| default, 4 threads | 265/265 | 265/265 | 265/265 | 0 / 0 |

Fresh streams were new to the prior audits and used 2000 examples per seed in
both ORT modes:

| seed | mode | candidate/reference | exact-poly raw/threshold | runtime/nonfinite |
|---:|---|---:|---:|---:|
| 112192071 | disable-all | 2000/2000 | 2000/2000 | 0 / 0 |
| 112192071 | default | 2000/2000 | 2000/2000 | 0 / 0 |
| 112192072 | disable-all | 2000/2000 | 2000/2000 | 0 / 0 |
| 112192072 | default | 2000/2000 | 2000/2000 | 0 / 0 |

Candidate margins were `min_positive=1.0`, `max_nonpositive=0.0` throughout.
The immediately preceding full run generated these rows; a quick second pass
only reclassified the recorded rows after immutable hashes and seed/count
configuration were rechecked.

The authority is an approximate prior implementation: it scored 1904/2000 and
1906/2000 on the two fresh streams in each mode. Its divergence is retained in
`review.json` as informational evidence. It is not a regression gate because
the candidate and independently established exact-poly rule both match the
generator and each other exactly.

## Structural gates

- ONNX checker `full_check=True`: pass
- shape inference `strict_mode=True, data_prop=True`: pass
- standard ONNX domain, finite initializers, no functions/sparse tensors: pass
- all node outputs static and positive-sized: pass
- runtime shape trace: 6/6 tensors truthful, 0 mismatches, 0 nonfinite values
- graph output: truthful `[1,10,30,30]`
- Conv-bias UB: 0 findings, including a standalone `check_conv_bias.py` run

The `neurogolf-onnx-golf` fail-closed workflow influenced the review by making
raw exact-control equivalence, dual-ORT execution, truthful runtime shapes, and
official profiled cost mandatory rather than relying on known accuracy alone.

## Artifacts

- `review.json`: complete machine-readable evidence
- `manifest.json`: hashes, costs, gates, and PASS verdict
- `review_task192.py`: reproducible independent audit

