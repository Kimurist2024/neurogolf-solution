# Lane C30 — task050 / task287

Result: **no promotable winner**. Both wave16 incumbents are retained unchanged.

## Incumbent verification

| Task | SHA-256 | Cost | Known disabled | Known default | Shape cloak | Conv UB |
|---|---|---:|---:|---:|---:|---:|
| 050 | `63dcf2dac0ab38c95afe890e5ba2decac3e7d35d622172a8bee9dc78ea23d45a` | 88 | 271/271, errors 0 | 271/271, errors 0 | false | 0 |
| 287 | `0c360bdd79e302da2f5ae1c45b3d7022870b497dea4a5f6da932864dbc7d2b7b` | 74 | 267/267, errors 0 | 267/267, errors 0 | false | 0 |

Both pass full ONNX checker, strict shape inference/data propagation, canonical/static truthful I/O, finite embedded initializers, and external-validator known-complete checks. Their pre-existing giant Einsums were not enlarged.

## Cheaper candidates rejected

- task050: built four cost-84 common-transition probes (K slice 0, slice 1, sum, and Hadamard). Every probe fails `train[0]` in both ORT modes, with runtime errors 0.
- task287: the only historical below-baseline model was cost 30. External validation gives 263/267, wrong 4, errors 0; all four training examples fail. It is also a prohibited fixed-index lookup.

Fresh-5000 was not run for these candidates because each failed the known-case gate. History scan covered 566 files / 18 unique task050 models and 575 files / 21 unique task287 models; no other below-baseline candidate exists.

Evidence: `audit.json`, `history_inventory.json`, `task050_probe_screen.json`, `task050_baseline_external.json`, `task287_gather_external.json`, and `winner_manifest.json`.
