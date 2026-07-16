# Changed-task resume report

## Outcome

- Baseline: `submission_base_8003.40.zip`
- Audited executable candidates: **34** (task073=5, task111=1, task122=1, task260=20, task271=1, task359=6)
- Accepted: **0**
- Fresh runs: **0** — every executable candidate failed known completeness first.
- task285/task289 were not retried because their known reductions depend on shape-cloak behavior.
- Sparse-initializer probes for task073/260/271 were rejected by full ONNX type/shape inference; sparse tensors cannot replace dense ConvTranspose/Einsum/QLinearConv inputs.
- No candidate was merged. Root ZIP, score JSON, CSV files, and `LOOP_STATUS.md` were not changed.

## Baseline identity and bias-UB gate

| Task | Cost | SHA-256 | Conv-family bias length |
|---:|---:|---|---|
| 073 | 12 | `d34dbd087d8d90c6b1f9d28be2786790b8eef44bccadd0a1225ec949f52b12a7` | PASS |
| 111 | 89 | `6af4011ea696c39460f4bbef5ead6ee6c7a80780ffca1a5b50e8e4534ad8fcf2` | PASS |
| 122 | 101 | `90c12806b6712f409a41b61b64a8b8256038a1c0b6707230b4d8bdc6cbce67e1` | PASS |
| 260 | 100 | `f334140ad26c60234be67f30da57e0974d185496246de46e10171cdc63453577` | PASS |
| 271 | 135 | `31394a849c0861a1dc2cc47e51aa287b9ab93cd5f64fafaee06b61651f4ad2df` | PASS |
| 285 | 8623 | `366212e29105fde0295030f3ec3bb014bd300f23aa8259ccd79da2eea720b9e2` | PASS |
| 289 | 32 | `d6098659b327d3f0c5f1fb8179686fb687d36fa52ced063ed45bb49561ae7408` | PASS |
| 359 | 24 | `0ac154453af9835deab6d1a8e9e217648b24dc3ec00f2f8e0c9b322a1ac5c0c4` | PASS |

All candidate files in `REJECTIONS.csv` also pass the Conv-family bias-length checker. They were rejected for correctness/runtime reasons before fresh validation.

## Rejection summary

- task073: FIR lengths 1–5 each score **0/15 known**, errors 0.
- task111: dead-node removal produces **265/265 runtime errors** from shape-buffer reuse.
- task122: dead-node removal produces **266/266 runtime errors** and does not reduce truthful cost.
- task260: all 20 singleton-broadcast candidates fail known completeness; best is only **23/266**.
- task271: archived cost-10 Gather candidate scores **0/267 known**.
- task359: four candidates score **0/266 known**; the two interrupted full profiles are independently rejected on train[0] with 1097 and 1381 differing cells.
- task285/task289: no new model was emitted; shape-cloak/dead-code variants were intentionally not retried.

Candidate-by-candidate paths, SHA-256, costs, checker/bias status, known counts, runtime errors, and rejection reasons are in `REJECTIONS.csv`.
