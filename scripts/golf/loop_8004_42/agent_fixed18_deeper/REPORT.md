# Fixed-18 deeper audit (8004.42 rebase)

## Result

No additional candidate is safe to adopt from this lane. The fixed 18 models
remain byte-for-byte authoritative in the parent baseline.

The lane inspected all 18 fixed tasks and screened the repository's retained
archive variants. The new 8004.42 members are already cheaper than the earlier
sound archive variants for the useful cases; no retained model beat the current
runtime-profiled cost while keeping exact correctness.

## Exact carrier-shave probes

Four graph-exact probes removed an initializer used only as `CastLike`'s dtype
carrier. All four pass ONNX full checking and strict shape inference, and all
four have Conv-family bias UB count 0. They are nevertheless rejected:

| task | baseline cost | probe cost | known / fresh | runtime | decision |
|---:|---:|---:|---|---|---|
| 031 | 224 | 411 | known PASS; fresh 100/100 | 0 errors | reject: cost increased |
| 071 | 188 | n/a | known FAIL; fresh 0/100 | Cast allocator shape mismatch | reject: runtime error |
| 088 | 218 | 316 | known PASS; fresh 100/100 | 0 errors | reject: cost increased |
| 302 | 151 | 52346 | known PASS; fresh 100/100 | 0 errors | reject: cost increased |

Although `CastLike(x, carrier)` and `Cast(x, to=...)` are value-equivalent,
they are not allocator-equivalent under ORT 1.24 with optimizations disabled
for these deliberately stale-shape graphs. The profiler therefore reports
larger truthful runtime shapes, or task071 fails buffer reuse outright.

## Other rejected avenue

- task183 has one apparently dead variadic `Min` (`hold_u8`). It is a known
  allocator barrier: pruning it makes `Resize` reuse a mismatched buffer. It was
  deliberately not promoted.
- Replacing task302's three dynamic `ConstantOfShape` scalars with constant
  initializers exposes the real `CenterCropPad` shapes and fails full strict
  checking against the existing value-info annotations. No candidate was
  emitted.

Evidence is in [candidate_screen.json](candidate_screen.json). No ZIP was
created or merged, and no root score/submission files were modified.
