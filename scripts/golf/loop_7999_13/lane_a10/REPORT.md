# Lane A10 — exact 7999.13 standard/soundness wave

## Result

No safe strict winner was found for tasks 037, 048, 092, 222, 226, 297,
and 345. Projected accepted gain is `0.0`. No root ZIP, CSV, ledger, score
pointer, or shared handcrafted model was modified.

The sole authority is exact `submission_base_7999.13.zip`, SHA-256
`a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`.

## task048: cheaper exact fold, rejected by fresh soundness

The first Einsum used a dedicated `einsum_nm=float32[1,1]=1` operand only to
create two singleton output axes:

```text
before: bchw,c,w,nm->bnhm
after:  bchw,nc,wm->bnhm
```

Reshaping the existing 10- and 30-element coefficients adds those singleton
axes without changing their values or element counts. This removes one
parameter, adds no node or value-info, keeps memory 308, and scores **379 ->
378**.

It passes full checking, strict shape inference, standard-domain, static-shape,
Conv-bias, known-gold, official-gold, and margin gates. Known results are
270/270 with zero errors under both ORT modes. The rewrite is also raw-bitwise
equal to the exact baseline on every fresh case tested.

That final fact prevents adoption: on independent fresh seeds the candidate is
only **4467/5000** under ORT_DISABLE_ALL and **4521/5000** under default ORT,
with zero runtime errors. It reproduces all 5000 baseline raw tensors in each
mode, including the baseline's wrong outputs. The strict 5000/5000 requirement
therefore rejects the apparent +0.002642 gain.

## task297: real cost 361, but only via out-of-schema Conv padding

The baseline first Conv stores a `(1,10,1,2)` kernel whose second column is
entirely zero. Trimming it to one column gives cost **371 -> 361** and remains
265/265 known-correct in both ORT modes if end padding `-24` is used to retain
the six-column result.

It is not admissible. The ONNX Conv schema requires every `pads` value to be
non-negative. Full checker acceptance does not override that schema contract,
so the `-24` model is quarantined before fresh validation.

Two standard alternatives were measured:

- non-negative one-column Conv plus explicit Slice: cost 484;
- non-negative one-column Conv plus Split/re-concat: cost 511.

Both are known-correct, but the full-width carrier is charged by the profiler
and dominates the ten-parameter saving. Neither is a score improvement.

## Other exact-hash and history conclusions

- task037: true-rule fusion triggers a deterministic Slice allocator mismatch;
  the topology-preserving bool form costs 437,668. Harvested alternatives start
  at static floor 497 versus incumbent 374.
- task092: five harvested alternatives were screened; best actual cost is 393
  versus incumbent 367. No dtype/shape-compatible initializer reuse remains.
- task222: the memory-zero rank-8 analog matcher is itself only 2813/3000 on
  independent fresh data. The SOUND analysis identifies the true rule as a
  global noisy-rectangle selection; preserving or retuning the analog matcher
  cannot produce an admissible cheap model.
- task226: the prior exact-hash audit found no safe reuse; harvested static
  floors begin at 400 versus incumbent 399.
- task297: 30 historical models were actually scored. Eleven cost 370, but all
  eleven fail known gold.
- task345: the only cheaper historical model costs 365 and fails known gold.
  Dense Wpack factorization adds a charged full-grid intermediate.

Machine evidence is in `baseline_manifest.json`, `history_screen.json`,
`task048_audit.json`, `task297_audit.json`,
`task297_standard_profiles.json`, `failure_manifest.json`, and
`winner_manifest.json`.
