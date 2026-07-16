# Lane A4 — strict 7999.13 wave

## Result

One strict winner is retained: **task324 cost 442 -> 439**, projected score
gain **+0.006810469002527242**.

- Candidate: `candidates/task324_synth_quarter.onnx`
- SHA-256: `894be3a6ae4d93ce52a5bb0ec8b03fe3443de74b746cff7529dc4989cd73ac08`
- Exact baseline member: `baseline/task324.onnx`
- Baseline SHA-256: `e33ea69907733e417fb0c42b105b3f187f9ae65f49a29df03f7b75a9aa2f5c62`

No root submission ZIP, score pointer, CSV, or shared artifact was changed.

## Why the task324 rewrite is exact

The baseline stored `deg2 = [0, 0, 0.25]` (three parameters) and used it in
two small-signature Einsums. The candidate removes that initializer and
synthesizes its two roles entirely inside those same final contractions:

1. Selecting one `0.5` entry from each of two existing `base0` operands gives
   exactly `0.25`.
2. `refdiff[Z,A] * onehot_values[A] * seedsel[Z,B] * Emap[e,B]` contracts to
   exactly `[0, 0, 1]`, so multiplying it by the synthesized quarter recreates
   `[0, 0, 0.25]` for the degree-labelled use.

The machine audit proves both identities exactly. The graph retains 22 nodes
and memory 227; only parameter count changes, 215 -> 212. Graph I/O, opsets,
node outputs, and all value-info/shape annotations are unchanged, so the
rewrite introduces no new shape cloak or runtime carrier.

## Mandatory validation

- Independent team validator: candidate known **266/266**, errors 0, cost
  **439**; baseline known 266/266, cost 442.
- Repository dual-gold gate: lib gold pass and official gold pass.
- Fresh generator audit: **5000/5000**, fresh failures (including runtime
  exceptions) **0**, rate 100%.
- Margin gate: pass, minimum positive margin **3.75**.
- ONNX full checker and strict shape inference: pass.
- Functions, sparse initializers, nested graphs, foreign domains, banned ops,
  and Conv-family nodes/biases: all zero.

Evidence: `external_task324.json`, `fresh5000.json`, and
`structural_audit.json`.

## Other target outcomes

| task | exact base cost | outcome |
|---:|---:|---|
| 019 | 536 | No winner. Emptying unused Split output `ca0` passes the checker but ORT exits `-11`; rejected as an error candidate. |
| 034 | 511 | No safe parameter reuse or lower-memory algebra after full graph/initializer review; 14x14 shift template has exact matrix rank 7, so a rank factorization does not reduce its 196 parameters and adds memory. |
| 237 | 529 | No winner. Packed lookup kernel rank decomposition would add a charged spatial intermediate; final QLinearConv also supplies the required free 30x30 padded output. |
| 250 | 468 | No winner. Existing historical harvest already placed the closest other known-correct model at cost 473. fp16 Resize ROI is an ORT type error and shape-spoof variants are excluded. |
| 308 | 434 | No winner. The exact base already contains the newer 4x4-index plus reflect-pad reduction; prior exhaustive harvest found no lower known-correct model. |
| 324 | 442 | **Accepted at 439.** |
| 377 | 409 | No winner. Exact row/color equality CSE variants remain known-correct but actual costs are 431/451/473 (and 481 with prefix fusion), all worse. Empty TopK output is schema-invalid. |

The rejected task377 equivalences expose larger actual profiler shapes than
the incumbent's existing annotations, so their apparent node-count savings do
not translate to score savings. None is retained.
