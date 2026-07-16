# task158 deep rebuild

## Outcome

One SOUND, strictly cheaper task158 candidate is accepted against the immutable
`submission_base_8005.16.zip` member. The actual cost decreases from **7615 to
7612**, for a projected score gain of **+0.00039403691322208417**. No ZIP or
protected score/CSV artifact was modified.

- candidate: `sound/task158_scatter_max_orientation_only.onnx`
- SHA-256: `3bfa73410f489f0bc444a1f4567f95837e445cd940d10b7282bdc50a95dd2dba`
- actual profile: memory 6739 + params 873 = cost 7612
- known corpus: 266/266 in both ORT modes
- independent fresh: 3000/3000 on each of seeds 1580461 and 1580462 in
  both ORT modes, with zero wrong and zero runtime errors
- raw margin: minimum positive 1.0, maximum nonpositive 0.0; no nonfinite or
  small-positive values

## True rule and exact rewrite

The generator places one complete 3x3 diagonally symmetric sprite and later
places two to four independently magnified/flipped copies which retain only the
two opposite endpoints. The output restores the hidden fill cells in every
copy. `reference_task158.py` implements this rule independently and reproduces
all 266 stored cases plus 3000 cases on each of two fresh seeds, covering all 33
reachable output shapes on each seed.

The incumbent orders up to three real target objects and then gathers their
vertical and horizontal orientation through an order vector. For every valid
slot that order is the identity, so those two three-element Boolean gathers are
exactly redundant. For invalid slots the candidate keeps the incumbent's
ordered top, left, and magnitude values, which point into the first real object
and therefore keep every ScatterElements index in range. It gates invalid
update codes to zero and uses standard opset-18 `ScatterElements(reduction=max)`;
the generator's restored colors are nonzero, so zero invalid duplicates cannot
erase a real update. This removes six counted Boolean bytes and adds three
counted update bytes, producing the exact three-byte reduction.

## Audit

The winner passes full ONNX checker, strict shape inference with data
propagation, the shared structure gate, and runtime declared/actual shape trace
with zero mismatches. It uses only the standard ONNX domain, has no banned op,
nested graph/function, external/sparse initializer, giant initializer/Einsum,
TfIdf/Hardmax lookup red flag, or Conv-bias UB finding.

The complete retained history scan found 60 unique task158 graphs and no actual
cost below 7615. Two misleading archive candidates advertised static costs
1844/1860 but reprofiled to 7838/7886, had 48--50 shape contradictions, and
failed fresh cases. A more aggressive new cost-7578 rewrite was also rejected:
its invalid slots could form out-of-range ScatterElements indices, producing
runtime failures on independent fresh seeds. Only the conservative 7612 rewrite
is listed in `winner_manifest.json`.

Authoritative evidence:

- `evidence/final_candidate_audit.json`
- `evidence/fresh_dual_7612.json`
- `evidence/reference_validation.json`
- `evidence/history_audit.json`
- `result.json`
- `winner_manifest.json`
