# 71406/71409 residual rescan against 8009.46

## Outcome

No exact or true-rule winner was found. Nothing was promoted, merged, or copied
into a quarantine pool.

The immutable authority was `submission_base_8009.46.zip`, SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`.
The scan covered all 112 standalone ONNX files under `others/71406` including
`71409`, excluding the 400-member submission directory. They deduplicated to
110 `(task, SHA-256)` pairs. Two source files were byte duplicates.

The actual comparison found 75 pairs already byte-identical to the immutable
8009.46 member. This is larger than the 24 already-fixed pairs expected in the
handoff, but is a direct SHA-256 result over the supplied directory and archive.

## Inventory result

| Status | Count |
|---|---:|
| Already fixed, same SHA as 8009.46 | 75 |
| Known black task | 9 |
| Not lower after official profiling | 18 |
| Not lower after authority reprofile | 1 |
| Parameter floor not lower | 2 |
| Official profiler/runtime failure | 2 |
| Static UB or strict-shape failure | 2 |
| Strict-lower candidate requiring runtime audit | 1 |

The only residual strict-lower pair was task382:

- source: `others/71406/task382_further_improved.onnx`
- SHA-256: `848c2a7f1d9251bb579268db779a9bc784fea1327959f6d2f34a0a0a191ba029`
- immutable member SHA-256:
  `67a510b15c399a6bbe0edbf169c1a4d0dcba655241b720df7d279fdbc5ca28fd`
- official profile: `820 -> 813`, apparent reduction 7
- private-zero catalog flag: false

## task382 fail-closed audit

The candidate passes full ONNX checker, strict shape inference with data
propagation, finite-initializer/static checks, and the Conv-family bias UB
checker (`0` findings). Those static checks do not make the shapes truthful.

Direct ORT tracing found 20 declared/runtime shape mismatches. The model declares
its output as `[1,10,1,1]` but produces `[1,10,30,30]`; several QLinearConv
intermediates similarly claim singleton dimensions while executing with length
30 or 34.

Four known-corpus configurations were measured:

| ORT configuration | Candidate runtime errors | Authority runtime errors | Candidate correct | Authority correct | Raw equal |
|---|---:|---:|---:|---:|---:|
| DISABLE_ALL, 1 thread | 0/266 | 0/266 | 254/266 | 254/266 | 266/266 |
| DISABLE_ALL, 4 threads | 0/266 | 0/266 | 254/266 | 254/266 | 266/266 |
| Default optimization, 1 thread | 266/266 | 266/266 | 0/266 | 0/266 | 0/266 |
| Default optimization, 4 threads | 266/266 | 266/266 | 0/266 | 0/266 | 0/266 |

The default-ORT failure is a QLinearConv buffer-reuse shape mismatch
(`{1,1,1,34} != {1,1,30,5}`). Although the candidate is raw-equal to the
authority on every known example under `ORT_DISABLE_ALL`, finite sampling is not
an exact graph proof, both-ORT execution fails, and the shape declarations are
not truthful. There is no reproducible true-rule construction for this exact
byte payload.

Fresh sampling was not run: fresh accuracy cannot repair a mandatory truthful
shape/both-ORT failure. The candidate is rejected as shape-cloaked rather than
retained as an approximate quarantine candidate.

## Artifacts

- `inventory_screen.py` and `inventory_screen.json`: complete source inventory,
  deduplication, exclusions, official profiles, and residual set.
- `audit_residual.py` and `audit.json`: strict checker, UB, runtime-shape, and
  four-configuration ORT evidence.
- `winner_manifest.json`: empty promotion manifest.

No root submission, `others/`, score ledger, or authority archive was modified.
