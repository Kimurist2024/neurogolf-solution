# C10 strict SOUND wave report

## Outcome

No strict winner was found for tasks 014, 036, 075, 159, 218, 225, or 245.
The accepted manifest is empty, projected gain is `+0.0`, and no root ZIP,
CSV, score ledger, or shared handcrafted artifact was changed.

The immutable baseline is exactly `submission_base_7999.13.zip`, SHA-256
`a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`.
Every member below was extracted directly from that archive.

| task | exact cost | memory | params | true generator rule | decision |
|---:|---:|---:|---:|---|---|
| 014 | 370 | 322 | 48 | crop the least-frequent nonzero color's exact bounding box | shape-cloak and fresh reject |
| 036 | 325 | 255 | 70 | crop the connected celestial object color, preserving all cells inside its bbox | shape-cloak; sound rebuild costs 446 |
| 075 | 345 | 311 | 3x3 tile copied into blocks selected by blue markers | shape-cloak; no cheaper history |
| 159 | 293 | 146 | magnify the separate 3x3 sprite into the red square frame | shape-cloak; cheaper variants wrong/crashing |
| 218 | 329 | 260 | recover the compact color table from a 2x2/3x3 rectangular quilt | shape-cloak; cheaper variant wrong |
| 225 | 333 | 233 | copy the 2x2 source to the four clipped diagonal offsets | no cheaper known-correct graph |
| 245 | 387 | 304 | translate the shifted red sprite into the green-corner 7x7 frame | shape-cloak; de-cloaking costs more |

## Runtime shape audit

Full checking, standard domains, banned/nested graph checks, sparse/function
checks, and Conv-family bias-length checks pass for all seven exact members.
That is not sufficient: executing every node output on a known example exposes
the following declared-versus-runtime shape contradictions.

| task | mismatched declared tensors | undeclared intermediates | one-example runtime intermediate bytes | official-like memory |
|---:|---:|---:|---:|---:|
| 014 | 15 | 0 | 163,478 | 322 |
| 036 | 14 | 16 | 20,329 | 255 |
| 075 | 33 | 3 | 70,696 | 311 |
| 159 | 8 | 5 | 63,273 | 146 |
| 218 | 2 | 0 | 45,255 | 260 |
| 225 | 0 | 24 | 233 | 233 |
| 245 | 14 | 0 | 66,993 | 304 |

Representative hard witnesses are task014 `masked_f` declared
`[1,1,1,1]` but executed as `[1,10,30,30]`; task036 `inputh` declared
`[1,1,1,1]` but executed as `[1,10,20,20]`; task075 `h9_shape30_1`
declared `[1,1,1,1]` but executed as `[1,1,30,30]`; task159 `K` declared
`[1,1]` but executed as `[2,30]`; task218 `gn` declared `[1,1,1,1]` but
executed as `[1,10,30,30]`; and task245 `grid3_f16` declared
`[1,1,1,1,3]` but executed as `[2,2,5,5,3]`. Task245 additionally fails
strict shape inference with data propagation at `AffineGrid`.

Task225 is the only graph without a contradictory declaration. Its 24
intermediates omit optional `value_info`, but their independently measured
runtime byte total is exactly 233, equal to the official-like memory charge;
there is no hidden-memory discount. It is therefore the only exact member that
remained eligible for ordinary lowering work in this strict no-cloak lane.

## Candidate search and terminal rejections

- **task225:** the prior exact-baseline scan covered 26 byte-distinct models.
  Its only cheaper model is the cost-306 `micro4` carrier and it passes 0/265
  known examples. C10 additionally ran all 48 available single ONNX optimizer
  passes; they produced five unique graphs including the base, zero strictly
  cheaper known-correct graphs, and one cost-333 semantic tie.
- **task014:** the apparent cost-203 unused-initializer/shape-op result is a
  metadata accounting artifact. Independent strict validation remeasures it at
  the baseline cost and it misses 1/100 fresh. The exact cost-370 family also
  previously found its first seeded mismatch at fresh index 451/1000.
- **task036:** the exact graph scored 2986/3000 on the prior independent fresh
  audit. The generator-derived true-rule graph is 3000/3000 and structurally
  sound, but costs 446 instead of 325. A 48-pass exact-baseline optimizer sweep
  from the prior C3 lane found no cheaper visible-correct graph.
- **tasks075/159/218:** the exact-baseline B6 scan covered respectively 28, 35,
  and 29 byte-distinct models. It found zero cheaper models for task075; three
  cheaper but known-wrong models for task159; and one cheaper but known-wrong
  model for task218. Task218's exact fixed-seed fresh result was 4996/5000,
  below this lane's exactness gate. The compact task159 family also has a
  reproducible ORT segmentation fault on legal fresh input.
- **task245:** the exact graph has prior 5000/5000 fresh evidence, but the 14
  runtime shape contradictions are terminal under the no-cloak rule. Literal
  shape replacement exposes truthful tensors and raises cost; the shortened
  coordinate and factored-sign probes fail runtime/gold or cost more.

No graph reached the prerequisite intersection of strictly lower real cost,
complete known correctness, and structural/no-cloak safety. Consequently no
candidate was eligible for the final independent 5000/5000 run under both
`ORT_DISABLE_ALL` and default ORT, and no fresh result is claimed as an
adoption result.

Evidence is in `baseline_anatomy.json`, `runtime_shape_trace.json`,
`optimizer_sweep.json`, `rejected_manifest.json`, and the empty
`winner_manifest.json`.
