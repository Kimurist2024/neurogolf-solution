# B19 factor/reuse audit — task340 and task361

## Outcome

No adoptable winner was found. The accepted list is empty and projected gain is
`+0.0`. Both task payloads are byte-identical between the exact
`submission_base_7999.13.zip` and Wave 12. No root ZIP, CSV, score ledger,
`best_score.json`, or shared handcrafted artifact was changed.

The generator sources were read directly:

- task340 / `task_d687bc17.py`: the four colors on the outer top/right/bottom/left
  borders define roles. Each same-colored interior marker is projected to the
  corresponding inner border lane; unrelated interior markers disappear.
- task361 / `task_e40b9e2f.py`: a size-10 partial two-color pinwheel is completed
  by placing every source wedge cell in all four 90-degree rotations. Legal
  length is 2--4 and length 2 uses the one-cell bump.

## task340 — sound floor retained

The exact cost-1173 model is structurally honest: zero `value_info`, no dead
node, no unused or byte-identical initializer, no banned/Sequence/nested graph,
and full checker plus strict shape/data propagation pass. An independent dual
runtime run is 5000/5000 under `ORT_DISABLE_ALL` and 5000/5000 under default ORT,
with zero generation or runtime/output errors.

All targeted exact-reuse scans were negative for task340: exact initializer
deduplication, model-wide factor reuse, Einsum factor reuse, low-rank
factorization, proportional reuse, initializer slicing, tensor-mode reuse,
shared-operand fusion, single-use inlining, signed-scale absorption, and signed
permutation absorption each produced zero task340 hits. Exact dead/no-op shaving
also produced no candidate.

The only large local table, `ACTable[6,9,1]`, contains 54 parameters and has
exact real rank 5. A dense rank factorization therefore needs
`5*(6+9)=75` parameters before accounting for any factor materialization; it is
strictly worse. Reversing the two Gather selections also raises the first table
intermediate from 90 to 180 bytes. The remaining reusable vectors are already
shared across every compatible Einsum. This agrees with the prior exhaustive
history/optimizer audit: 84 harvested rows bottom out at 1173 and all 48
single-pass optimizer variants fail to become cheaper.

Decision: retain the sound baseline; no fresh candidate exists to adopt.

## task361 — all compact lineages rejected

The nominal cost-968 baseline is not eligible for an exact-equivalence
exception. Although it is 5000/5000 under `ORT_DISABLE_ALL`, default ORT rejects
the session at `CenterCropPad` shape inference. More fundamentally, each of its
three `GroupNormalization(input, ...)` results is declared `[1,1,1,1]` even
though the operator preserves the fixed input shape `[1,10,30,30]`. The graph
output is likewise declared `[1,1,1,1]` while runtime emits the full one-hot
grid. Generic checker and strict inference acceptance do not remove these
explicit operator-shape witnesses.

Every unique harvested lineage has the same three GroupNormalization cloak
witnesses and the same default-ORT session error:

| actual cost | known result | disabled-ORT fresh100 | other terminal reason |
|---:|---:|---:|---|
| 810 | 198/266 | 79/100 | 68 known failures; cheaper but wrong |
| 854 | 218/266 | 79/100 | 48 known failures; cheaper but wrong |
| 1004 | 266/266 | 100/100 | cost is higher; shape-cloak |
| 1006 | 266/266 | 100/100 | cost is higher; shape-cloak |
| 1363 | 266/266 | 100/100 | cost is higher; shape-cloak |

The compact family depends on using a falsely tiny GroupNormalization result to
quantize the full `[1,10,30,30]` input before the pinwheel logic. Correcting that
first declaration alone exposes a 36,000-byte float32 intermediate, already far
above cost 968. A conventional honest crop/decode also cannot compete: cropping
the input first materializes 4,000 float bytes, while channel-decoding first
materializes 3,600 float bytes. No lower-rank or initializer-reuse scan found a
task361 alternative.

Decision: reject every archived task361 candidate. There is no both-ORT,
error-free, truthful-shape model below the exact cost.

## Evidence

Primary machine-readable evidence is `audit_evidence.json`, with the final
empty result in `winner_manifest.json`. Runtime evidence is in
`task340_dual5000.json`, `task361_base_dual5000.json`, and the per-lineage
`history/*_known.json` / `history/*_dual100.json` files. Scan manifests are
retained under this lane only.
