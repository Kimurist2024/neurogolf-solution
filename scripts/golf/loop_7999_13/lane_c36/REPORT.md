# Lane C36 — task012 exact local-rule lower bound at 8000.46

## Decision

Retain the authoritative `task012.onnx`. No eligible cheaper model was found,
so this lane contributes **+0.0** and emits an empty winner manifest.

The authority is `submission_base_8000.46.zip`
(SHA-256 `74cb9c4a...6548f534`). Its task012 payload has SHA-256
`478a310e...76500500` and official-like cost **710 = 0 memory + 710
parameters**.

## Baseline gates

- Full ONNX checker and strict shape inference/data propagation pass.
- Canonical, static truthful `[1,10,30,30]` input/output; runtime shape cloak is
  false; no intermediate allocation is exposed.
- Standard ONNX domain, finite embedded initializers, one depthwise `Conv`, no
  lookup, subgraph, custom op, or Einsum.
- Known dual ORT: **265/265** in disabled and default modes, errors 0.
- Moved external validator: valid, preflight true, **265/265**, errors 0, cost
  710.
- Independent fresh control seed 93612: **100/100** in both ORT modes, errors
  0.

## Completed geometry gap

The task generator `0962bcdd` was reduced to its complete 392-case parameter
domain. A source point must influence both signed radius-2 offsets on each
axis, so both kernel sides must be at least five. For cost below 710 the kernel
area must be below 70, hence no side can exceed `floor(69/5)=13`.

Every such integer geometry and every padding alignment was checked:

- prior `kh>=7, kw>=7` region: 351 alignments;
- missing side-5/6 region through side 10: 869 alignments;
- missing long-side 11–13 region: 492 alignments.

All **1,712** alignments are infeasible with ordinary hard margins. The exact
decoder boundary was also solved separately with positives `>=1` and negatives
allowed to equal zero; all **1,712** remain infeasible. Bias-free boundary
searches likewise find no solution, including both 7x10 and 10x7 at area 70.
Thus there is no smaller exact one-node depthwise Conv, and the ten bias values
cannot be removed.

## Other exact reductions

The background and shared foreground 7x10 kernels both have matrix rank 7 and
no removable all-zero border. An exact separable factorization therefore needs
seven branches and exposes at least 63,000 intermediate elements, far above
cost 710.

Foreground kernels are byte-identical, but ONNX grouped Conv requires the ten
serialized output-channel slices. Batching foreground channels to reuse one
kernel requires separate background/foreground Conv results totaling at least
9,000 intermediate elements before one-output assembly. Group counts 1/2/5
permit kernel areas only 6/13/34 under cost 710; the radius lower bound and
legal cases with partner channels absent reduce these to already-rejected
depthwise cases. No exact factor, gauge, contraction, shared-axis, or active
input-column rewrite can beat the incumbent under the official memory metric.

## Evidence

`audit.json`, `exact_structure_audit.json`, all
`missing_biased_search*.json` / `homogeneous_search*.json`, both fresh-control
JSON files, `task012_baseline_external.json`, and `winner_manifest.json`.
