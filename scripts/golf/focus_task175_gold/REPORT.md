# task175 exact gold-safe reduction

## Result

- Authority: `submission_base_8014.69.zip:task175.onnx`
- Pinned authority member SHA-256: `b6404486ccc1a74c36bab6031f11c54c7326f787a743f02dff77e63c782af343`
- Authority cost: **140** (`memory=0`, `params=140`)
- Accepted candidate: `candidates/task175_gauge_remove_w_v.onnx`
- Candidate SHA-256: `acead77ce6b60ae5d5dd88e5c2c006cecdac6c9c5fd56bc97b56b37b72df8a1a`
- Candidate cost: **134** (`memory=0`, `params=134`)
- Strict score gain: `ln(140/134) = +0.0438026227`

This candidate is an algebraic rewrite of the current LB-white authority, not
a retrained or policy-rate approximation.  It removes the four-parameter `W`
and two-parameter `V` initializers.

## Exact rewrite

The relevant authority contraction is

```text
TA[P,l] * TB[l,g,r] * W[r,L] * TB[L,m,D] * V[D]
```

Let `a = W[1,0]` and choose

```text
X     = [[1, 0], [1-a, 1]]
X^-1  = [[1, 0], [a-1, 1]]
Y     = [[1, 0], [1,  -1]]
```

For the pinned float32 authority, `Y @ X == W` exactly.  Define

```text
TB' = X * TB * Y
TA' = TA * X^-1
```

Also, `Y @ [1,1] == V`.  Therefore the rewritten contraction

```text
TA'[P,l] * TB'[l,g,r] * TB'[r,m,D]
```

with `D` left unpaired (implicit sum over `[1,1]`) is the same tensor network.
`W` and `V` are no longer model operands or initializers.

## Strict gates

`scripts/golf/try_candidate.py`:

- PASS file size: 1017 bytes
- PASS validator, allowed ops/opset, ONNX full checker, and static shapes
- PASS all visible train/test/arc-gen gold
- PASS margin: minimum positive raw value `0.25`
- PASS official profile: `memory=0`, `params=134`, `cost=134`

`scripts/verify_fix.py --k 2000 --min-fresh-rate 1`:

- `decision=ADOPT`
- `lib_gold=true`
- `official_gold=true`
- `margin_stable=true`, `margin_min=0.25`
- fresh seed `777175`: `2000/2000`, 0 failures

Independent audit in `audit.json`:

- seed `777175`: `2000/2000`, 0 failures
- seed `1775175`: `2000/2000`, 0 failures
- both seeds: minimum positive raw `0.25`, maximum non-positive raw `0.0`
- strict inferred output shape: `[1,10,30,30]`

## Rejected lower-cost experiment

`candidates/task175_gauge_remove_w_v_cp4.onnx` has cost 130 but replaces `R`
with a non-exact rank-4 CP approximation.  It failed visible gold at `train[1]`
and is **rejected**.  It must not be merged or probed.

## Files and mutation scope

- Reproducible builder: `build_candidates.py`
- Independent two-seed audit: `audit_candidate.py`
- Machine-readable evidence: `build.json`, `audit.json`
- `try_candidate.py` promoted the accepted candidate only to
  `artifacts/handcrafted/task175.onnx`.
- No submission archive, score CSV, or root authority file was modified by
  this lane.
