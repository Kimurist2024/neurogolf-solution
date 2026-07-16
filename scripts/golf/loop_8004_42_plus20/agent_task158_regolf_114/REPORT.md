# task158 exact regolf from cost 7529

## Outcome

One SOUND, strictly cheaper task158 replacement is accepted against the prior
cost-7529 winner. Official-like cost decreases from **7529 to 7525** with
unchanged truthful memory 6662 and parameters 867 -> 863. Projected incremental
score gain is **+0.000531420233045797**. No ZIP, root score file, submission, or
other lane was changed.

- candidate: `sound/task158_exact_regolf.onnx`
- SHA-256: `127984c6807d84559bbf74fd58e3b09a66459d142cef65a8635647e64f5e59fd`
- baseline SHA-256: `9d9a3ca8fb39856125925ea464ed1cc80f0301bd785ff7b60da37bd1c2b6b9d1`
- trusted cost-7612 SHA-256: `3bfa73410f489f0bc444a1f4567f95837e445cd940d10b7282bdc50a95dd2dba`
- known: 266/266 in both ORT modes, raw delta to trusted zero
- fresh: seeds 1581141 and 1581142, each 2000/2000 in both modes
- fresh safety: wrong 0, runtime 0, nonfinite 0, small-positive 0, raw delta 0

## All-reachable exact proof

Let `S` be the bias-free anchor convolution sum in the 7529 graph. The endpoint
codes are integers in {0,1,2}, the stencil is integral, and the input is exactly
one-hot, so `S` is integral. The stencil absolute sum is 179, hence
`|S| <= 358`; both `S+146` and `2*S` remain integer values below magnitude 1024
and are represented exactly in float16.

The winner removes Conv bias +146, doubles the anchor stencil, and maps every
decision threshold `t` to `2*(t-146)`:

- phase thresholds 149/157/177 become 6/22/62;
- role thresholds 148/154/170/218 become 4/16/48/144;
- strict validity `S+146 > 146` becomes exact integer test `2*S >= 2` using the
  existing float16 scalar two.

Positive scaling preserves every TopK order and tie. Therefore every phase,
role, validity, selected slot, coordinate, permutation, Scatter index/update,
and final output is unchanged for every reachable input, not merely the sampled
corpus. The transformed constants expose two typed initializer aliases:

- float16 `phase_cut_2=62` reuses `nm_d62`;
- uint8 `more_role_0=4` reuses `lutnp_shift4`.

Together with removal of `da_anchor_bias` and `anchor_floor`, this removes four
parameter elements without adding any runtime tensor.

## Audit and rejected shaves

The winner passes full checker, strict data-propagating shape inference, shared
structure gate, Conv-bias UB check, and runtime declared/actual shape trace with
zero mismatches. It uses standard ONNX only, no banned op, lookup, sparse or
external initializer, nested graph/function, giant initializer, or giant
Einsum. Maximum Einsum arity is three.

The 7529 baseline had no dead initializer, duplicate typed initializer, pure
CSE, or removable no-op. Its two unused TopK Values tensors are schema-required.
A uint8 positional-rank rewrite would reduce static cost but was rejected because
ORT CPU has no uint8 TopK kernel in either required mode. Scatter gating was
left intact: moving its compact [1,3] float16 gate to [1,18] indices or adding
Mod/Clip increases official memory. The full permutation table was retained
because narrower sampled support did not establish an all-generator proof.

Evidence:

- `evidence/audit.json`
- `evidence/fresh_dual_2x2000.json`
- `evidence/exact_proof.json`
- `evidence/search_summary.json`
- `result.json`
- `winner_manifest.json`
