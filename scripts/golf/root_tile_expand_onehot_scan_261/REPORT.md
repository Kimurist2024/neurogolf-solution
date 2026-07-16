# Tile / Expand / OneHot exact residual scan

## Outcome

`winner = null`. The complete current 400-model authority has four `Tile`, one
`Expand`, and four `OneHot` nodes. None has an algebraically exact rewrite with
a strictly negative official cost delta, so no ONNX candidate was created.

Authority: `submission.zip`, SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`.
This lane did not edit the ZIP, score files, root models, or any other lane.

## Complete census

| Task | Residual node(s) | Known official profile |
|---:|:---|:---|
| 066 | OneHot x1 | 266/266, cost 562 = 346 memory + 216 params |
| 133 | Expand x1 | 267/267, cost 4393 = 4176 + 217 |
| 200 | OneHot x1 | 84/84, cost 346 = 200 + 146 |
| 233 | Tile x3 | 266/266, cost 7308 = 6991 + 317 |
| 247 | OneHot x1 | 269/269, cost 212 = 165 + 47 |
| 300 | OneHot x1 | 267/267, cost 175 = 120 + 55 |
| 388 | Tile x1 | 266/266, cost 305 = 283 + 22 |

Every authority passes full ONNX checking, strict shape inference with
`data_prop=True`, standard-domain/banned/nested/function/sparse checks, and the
Conv-family bias-length checker. The known profiles have zero runtime errors,
nonfinite outputs, or final output-shape errors.

Pre-existing runtime/declaration contradictions remain in tasks 133, 233,
300, and 388. They are recorded rather than treated as permission for a new
shape cloak. No candidate inherited or added one.

## Tile results

No Tile has all-one repeats, and none repeats only runtime-singleton axes.

- task233 nodes 95/96/97 each take an actual `[5]` vector and repeats `[3]`,
  producing `[15]`. `Expand` would repeat each element, whereas Tile repeats
  the whole five-element sequence. A reshape/expand/reshape formulation adds
  counted activations and is not lower.
- task388 repeats `[1,1,2,2]`, but its actual input shapes span
  `[1,1,2,2]` through `[1,1,6,6]`; the outputs span 4x4 through 12x12.
  Its stored 1x1-looking declaration is not the runtime geometry. Expand
  would broadcast individual cells, not duplicate the complete tile, and is
  therefore not exact.

The shared one-element task233 repeats initializer is already shared across
all three Tiles. No initializer-alias saving is available.

## Expand result

Task133 expands an actual int8 `[1,1]` color coordinate to `[1,4,2,1]`, then
passes it to `Concat` beside row and column tensors of exactly that shape.
Concat does not broadcast. The four-element shape initializer is unique and
has no identical live alias.

Changing the preceding Add to produce the expanded tensor directly is not a
win: its one-element output and one-element constant become eight-element
objects. It removes the eight-element Expand output and four-element shape,
but raises the producer activation and initializer enough to be no cheaper.
Tile is merely an equal-output/equal-shape-parameter substitute. No candidate
was emitted.

## OneHot results

For each fixed-depth OneHot, the scanner constructs an optimistic lower bound
for `Equal(indices, range)`:

1. broadcast is assumed to require no Unsqueeze;
2. any unique depth/values initializers are removed;
3. an already-live exact range or identity basis would be reused if present;
4. because every downstream consumer requires numeric input, the original-size
   numeric Cast/Where result remains and Equal's bool selector is added.

No model contains a compatible live range or identity basis. The resulting
best-case cost deltas are strictly positive:

| Task | OneHot anatomy | Numeric consumer | Optimistic delta |
|---:|:---|:---|---:|
| 066 | depth 30, float `[1,30]`, values `[0,1]` | Einsum x2 | +59 |
| 200 | depth 10, fp16 `[1,1,10]`, values `[0,1]` | Conv | +17 |
| 247 | depth 10, fp16 `[1,10,1,3]`, values `[-64,64]` | RoiAlign | +39 |
| 300 | depth 10, actual fp16 `[10,1,1,1]`, values `[0,1]` | Einsum + CastLike | +17 |

Downstream absorption does not reverse those bounds:

- task066 shares one selector between two contractions; selected-profile
  materialization needs two 30-float tensors instead of one.
- task200's impulse selector feeds a padded Conv. A response-table Gather
  needs a 10x2x30 table, while coordinate/parity construction adds a
  30-element arithmetic intermediate.
- task247's graph-output RoiAlign jointly performs color selection and the
  variable-height/width spatial realization. Externalizing the selector adds
  an intermediate color mask and another spatial operation.
- task300 requires the selector as a numeric Einsum factor and as a dynamic
  weight source. Equal must be cast back to fp16, and no basis is reusable.

## Admission gates

The task brief requires full/strict/runtime-shape, four raw runtime
configurations, error/UB zero, and two fresh seeds x1000 for a lower survivor.
There is no strictly lower exact survivor before candidate creation, so those
candidate-only known4/fresh gates were correctly not run. Approximation,
lookup, fixture correction, and shape-cloak candidates were not considered.

## Reproduction

From the repository root:

```bash
.venv/bin/python scripts/golf/root_tile_expand_onehot_scan_261/scan.py
.venv/bin/python scripts/golf/root_tile_expand_onehot_scan_261/audit.py
```

Authoritative artifacts:

- `scan.py` — complete ZIP census, official known profiles, runtime shape
  tracing, exact rewrite classification, and cost accounting
- `scan.json` — per-node evidence
- `audit.py` / `audit.json` — fail-closed consistency result

Final decision: `NO_STRICT_LOWER_EXACT_REWRITE`; safe gain 0.
