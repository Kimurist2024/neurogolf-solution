# task066 cost-561 Selu candidate — independent review 206

## Verdict

**PASS / safe exact pass-through admission.**  I independently audited
`task066_selu_cost561.onnx` against the immutable 8009.46 task066 authority.
The candidate is one official cost unit cheaper, preserves the authority's
downstream `uint8 ti` carrier over the complete generator-reachable numeric
domain, and introduces no error/shape/UB condition.

- Authority: `/private/tmp/ng800946_rank/task066.onnx`
- Authority SHA256: `bb8cebc8d71d275f4ec3f542d6aefea238b6c36d1cec77c0f2c1d533bf04ab4e`
- Candidate: `../agent_task066_selu_200/task066_selu_cost561.onnx`
- Candidate SHA256: `2e3bd402f667062b32858d3a11182d3e8050d833d2974d1d37fbadd688f4648b`
- Official authority profile: memory 346, params 216, **cost 562**
- Official candidate profile: memory 346, params 215, **cost 561**
- Score gain: `ln(562/561) = +0.001780944370994692`

This review did not edit the root ZIP/CSV or `others/71407`.  At completion,
their relevant root hashes remained the expected immutable baseline values:
`submission.zip=4eb324d7...` and `all_scores.csv=8c99379c...`.

## Independently verified graph delta

The protobuf comparison found exactly one changed node and one removed scalar:

- node 64: `Div(selLog, ln2) -> selQ` becomes
  `Selu(selLog, alpha=1, gamma=1.4432698488235474) -> selQ`;
- float16 initializer `ln2=0.69287109375` is removed;
- the other 76 nodes, all 18 common initializers, model shell, opsets,
  inputs/outputs/value-info, and metadata are protobuf-identical.

`selQ` has exactly one consumer: node 65, `Cast(selQ -> UINT8) -> ti`.  Therefore
complete `ti` equality closes the replacement proof even if the two float16
arithmetic paths differ before that cast.

## Full generator-support proof (`selF >= 1`)

I did not rely on lane 200's prose.  `proof_geometry.py` independently contracts
the opaque authority `Einsum` constants and enumerates the generator geometry.

### Graph algebra

Contracting the two coordinate `Einsum`s to effective input weights yields:

- `rs[0] = sum(row * input[color=red])`;
- `rs[1] = sum(row * input[color=green])`;
- `cs[0] = sum(col * input[color=red])`;
- `cs[1] = sum(col * input[color=green])`;
- every background/cyan/other-color coefficient is exactly zero.

Each marker has two adjacent cells.  The graph's uint8 cast, right-shift by one,
and red-row parity therefore recover the minimum red/green path coordinate and
select row versus transposed-column orientation exactly.  The red marker is
always on the right of the green marker, so `cOut=cR+1` is exactly the outward
red cyan-guard column.

The independent channel contraction is one-hot cyan (color 8); the second
`G` input is one-hot green (color 3); and `pow2` is `2**0..2**19` followed by
zeros.  Since the two green marker cells occupy the same canonical column,
the graph computes:

```text
G = 2 * (cyan row bitmask in the green-marker column)
O =     (cyan row bitmask in the outward-red column)
```

### Exhaustive geometry

Exact inclusive bounds from `task_2dd70a9a.py` and `common.randint` were
enumerated:

| generator shape | base geometry tuples |
|---|---:|
| S | 15,336 |
| U | 449,928 |
| including both flip and xpose values | **1,861,056** |

For every tuple, the mandatory guard pair supplies one common bit:

| shape/orientation | mandatory result |
|---|---|
| S, unflipped | `G` shifts guard `mid-1` onto `O[mid]`; `mid<gr0`, so `aMask>0` |
| S, flipped | reflection plus `G>>2` aligns the pair at least two above `gr0`; `bMask>0` |
| U, unflipped | guard `base+1`, initial `G` shift, then `G>>2` align at `base>=gr0+2`; `bMask>0` |
| U, flipped | reflection aligns the pair below `gr0`; `aMask>0` |

Random cyan cannot invalidate this result: it is prepended, then the path and
mandatory guards overwrite it.  Any surviving extra cyan only ORs additional
distinct power-of-two bits into `G/O`; bitwise AND cannot clear the mandatory
common bit.  The complete `useB` truth table was also exhausted: if `aMask=0`,
the proof supplies `bMask>0`; if a positive `aMask` is overridden, `useB` can
only do so through `hasB`, hence `bPow=bMask & -bMask` is positive.

Consequently, every generator-valid selected uint32 mask is in
`[1, 2**20-1]`.  Its float16 cast is positive (finite or positive overflow),
so the input to the replacement is never negative or NaN.

## Complete numeric carrier proof

An independent isolated ONNX model evaluated every uint32 value in the strict
superset `[0, 2**20]`, **1,048,577 inputs**, in all four required ORT modes.

| mode | float16 `Div` vs `Selu` differences | `Cast(... -> uint8)` differences |
|---|---:|---:|
| disable-all, 1 thread | 51 | **0** |
| disable-all, 4 threads | 51 | **0** |
| default, 1 thread | 51 | **0** |
| default, 4 threads | 51 | **0** |

The candidate carrier is also bit-identical across all four modes.  The only
nonfinite isolated `Div` output is at sentinel input zero, which the support
proof excludes.  Thus every reachable integer, including every value where
float16 `Div` and `Selu` round differently, produces exactly the same `ti`.

## Whole-model audit

Independent fresh seeds were `66206001` and `66206002`, distinct from lane 200.
Every row below was repeated in disable-all/default ORT with 1/4 threads.

| suite | candidate/gold per mode | final raw equal to authority | `ti` raw equal | errors |
|---|---:|---:|---:|---:|
| known | 266/266 | 266/266 | 266/266 | 0 |
| fresh seed 66206001 | 1881/2000 | **2000/2000** | **2000/2000** | 0 |
| fresh seed 66206002 | 1888/2000 | **2000/2000** | **2000/2000** | 0 |

The authority itself is only `3769/4000 = 94.225%` on these arbitrary fresh
draws, but the candidate is an exact pass-through on all 4,000 inputs in all
four configurations.  The fresh misses are inherited authority behavior, not
a candidate regression.  Sampled `selQ` was also raw-identical in every case.

Runtime errors were zero and final nonfinite values were zero.  The existing
authority graph sometimes materializes `selF=+inf` (6 known cases and 48/34
fresh cases per configuration); the candidate inherits exactly the same
occurrences.  Neither model produced a nonfinite `selLog`, `selQ`, `ti`, or
final output in these audits.  This inherited intermediate is covered by the
complete carrier proof and is not a new failure mode.

## Structural gates

- ONNX full checker: pass;
- strict shape inference: pass;
- strict shape inference with data propagation: pass;
- all node outputs statically resolved: pass;
- runtime shape/dtype trace: **79/79 truthful**, zero mismatches in both
  disable-all and default ORT;
- standard ONNX domains only: pass;
- functions, nested graphs, sparse initializers, banned/Sequence ops: zero;
- Conv-family short-bias UB findings: **0**;
- official known correctness: authority and candidate both pass.

## Artifacts

- `proof_geometry.py` — independent effective-Einsum contraction, pinned graph
  algebra, exhaustive S/U/flip/xpose support proof, and selection truth table.
- `audit_review.py` — independent protobuf/static/cost audit, four-config
  known/fresh raw audit, exhaustive uint32 carrier check, and runtime-shape
  trace.
- `REPORT.md` — this decision record.

