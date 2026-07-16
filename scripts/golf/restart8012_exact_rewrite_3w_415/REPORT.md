# Wave415 exact rewrite search (8012.15)

## Result

**New admissions: 0.** The root authority remains byte-identical to
`submission_base_8012.15.zip` (SHA-256
`1dce2bc9bd65497daf59575512972309a3d71b87ee406a97c21fe2ee16010231`).

Three internal workers partitioned 139 non-catalog authority members whose
current cost is 100–500 into 47/46/46 tasks. The scanner covered dead-code and
unused-initializer removal, typed initializer aliases, scalar broadcast/type
template reuse, no-op and neutral arithmetic elimination, CSE, optional
outputs, constant/shape folding, safe ONNX optimizer fusions, literal
CenterCropPad no-ops, and Einsum unit/precontract/same-term fusion.

- 81 authority members could not enter the strict-canonical candidate lane
  because of legacy strict-shape or declared canonical-I/O failures.
- 58 strict-canonical members were processed.
- 12 distinct structural variants were produced: six had no static cost gain,
  four failed strict structure, and two apparent one-byte CenterCropPad shaves
  changed raw output on known cases (tasks 162 and 279).
- No structural variant reached the fresh gate.

## History raw-equivalence rebase

The same three-worker contract rechecked 93 previously profiled actual-lower
history SHAs against the raw current authority. The only known-raw match was
task268, cost 420→327, SHA-256
`22ea97ffce8b14fbf923a89a0cda2233d83469201732ed3f4e914e5b2b1ced69`.
It is **rejected**: the pinned 8012.15 audit records fresh accuracy 41.35% and
45.10% and default-ORT session construction errors. It fails POLICY90 and is
not an exact authority-equivalent rewrite.

The retained task268 file under `candidates/history_worker_1/` is rejection
evidence only; it must not be packaged or merged.

## Protected state

`submission.zip`, `submission_base_8012.15.zip`, and `all_scores.csv` were not
modified. No file under `others/` was created or changed by this lane.
