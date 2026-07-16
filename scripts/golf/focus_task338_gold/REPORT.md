# task338 gold-exact optimization

## Result

Admit `task338_row_winding_nand_ba865e4f333c.onnx` against the cost-403
task338 entry in `submission_base_8012.23.zip`.

- Cost: **403 -> 401** (memory 398, parameters 3)
- Score gain: **+0.004975134640113708**
- Official gold: **267/267 exact**
- Raw margin: **minimum positive 1.0**, no value in `(0, 0.25)`
- Fresh seed `338920001`: **2000/2000 exact**
- Fresh seed `338920002`: **2000/2000 exact**
- Fresh errors, non-finite values, shape mismatches, and small positives: **0**

Candidate:
`scripts/golf/focus_task338_gold/task338_row_winding_nand_ba865e4f333c.onnx`

SHA-256:
`ba865e4f333c2e098488356c62a47c6ad0a0d06bb1eb77ab21f21cd47cf42e5b`

Machine-readable verification is in `final_evidence.json`.

## Exact rule

The task generator draws pairwise-separated red rectangular frames.  On every
strict-interior row of a non-empty frame, the two vertical sides are exactly
the red cells whose immediate upper and lower neighbors are also red.  Frames
of height two have no interior and produce no such crossing.

The inclusive left-to-right parity of these crossings is therefore one only
between the paired sides of each frame.  Pairwise separation makes the parity
rule work for any number of frames sharing a row.  Masking parity by `not_red`
removes the left boundary crossing itself.  This is a specification-derived
rule, not a learned or private-zero approximation.

The graph computes the 30-column prefix XOR with five synchronous distances
`1, 2, 4, 8, 16`.  Each XOR uses four binary NAND gates.  NAND is implemented
with the same fp16 `Selu`/`PRelu`/`HardSigmoid` lineage as the accepted
authority graph, and shifts reuse its three-node `CenterCropPad` primitive.

## Rejected approaches

- Archived rank-5 cost-334 Einsum: visible sign-gold only; fresh correctness
  about 70% by case and real sign errors, so rejected.
- Trained rank-6 cost-394 Einsum: only 92.1% on 2,000 validation cases, so
  rejected.
- Cost-403 authority and cost-401 one-node bypass: both combine boundaries
  from different boxes on fresh layouts, so rejected as semantic bases.
- Direct exact `Conv/CumSum/Mod`: gold-exact but cost 10,959 because live
  30x30 intermediates are charged.
- Negative-pad prefix version: gold-exact but each `Pad` exposes 1,800 bytes;
  replaced by the profiled `CenterCropPad` shift lineage.

## Mutation boundary

This lane did not edit `submission_base_8012.23.zip`, `submission.zip`, or the
score ledger.  Running the repository's normal `try_candidate.py` gate copied
the admitted candidate to ignored `artifacts/handcrafted/task338.onnx`; the
verified standalone candidate and evidence remain under this focus directory.
