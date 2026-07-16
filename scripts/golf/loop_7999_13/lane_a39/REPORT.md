# A39 dead-node / dead-output audit

## Outcome

- Decision: **NO_ADOPTABLE_CANDIDATE**
- Authority score: **8000.46**
- Score gain: **+0.00**
- Authority SHA-256: `74cb9c4a96dcf9d1d1ca4ad1e0afdb67f5218c0a06ad8e896ad1654b6548f534` (unchanged)
- Scanned 400 models; found four single-output dead nodes and 46 partial multi-output sites.

## Requested dead nodes

| Task | Prune | Result | Safety lineage |
|---|---|---|---|
| 039 | `Equal keep_bg_equal` | 264/264 known runtime errors in both modes; Slice buffer-reuse mismatch | CenterCropPad |
| 089 | `ReduceMax keep_red_big` | default session creation fails; historical disabled probe fails all known cases | CenterCropPad |
| 122 | `GreaterOrEqual d_keep` | 266/266 known errors after pruning in both modes | CenterCropPad |
| 183 | `Min hold_u8` | 265/265 known runtime errors in both modes; Resize buffer-reuse mismatch | lookup/ScatterElements |

The apparently dead tensors are allocator/shape barriers. Removing them changes ORT buffer reuse and is therefore not a valid optimization.

## Partial multi-output nodes

- task019/task124 `Split`: replacing the unused variadic output with `""` passes ONNX checker but crashes ORT with SIGSEGV. This is rejected as undefined/unsafe behavior.
- task080/task400 `MaxPool`: the first Values output is schema-required even when only Indices is consumed.
- task131 `TopK`: Values is schema-required; the source also has forbidden CenterCropPad and lookup lineage.
- Replacing the two safe `Split` sites with individual `Slice` nodes is larger: it removes one singleton allocation but requires new int64 starts/ends/axes parameters, with no reusable initializer bank. `Gather` would add forbidden lookup lineage.

## Validation gate

No candidate survived the safety and known-runtime gates, so fresh dual 5000 and external 500 were correctly not run. No authority model or archive was modified.
