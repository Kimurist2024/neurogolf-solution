# Lane A5 — strict 7999.13 exact-local wave

## Result

No winner is retained. The projected aggregate gain from this lane is **0.0**.
All five experimental models were rejected before fresh validation because
none was both strictly cheaper and free of known/runtime errors. In particular,
the task338 rewrite that offered the largest apparent saving was not accepted:
it deterministically raises an ORT buffer-shape exception.

No root submission ZIP, best-score pointer, score CSV, or shared optimized model
was changed.

## Pinned baseline recheck

The seven members were extracted directly from
`submission_base_7999.13.zip` (SHA-256
`a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`).
Every exact member was rerun through `verify_candidate_timeout.py`; all seven
remain known-correct with zero runtime errors.

| task | member SHA-256 | memory | params | cost |
|---:|---|---:|---:|---:|
| 008 | `30abdd1f30f1aa88549edbf22c6e7a4af4fec3036fd8809812456ccb0df6e292` | 331 | 100 | 431 |
| 025 | `22a44063541ed6c696b535d4059abf5dd4844a0fafef2df6006b6f1e027ea595` | 396 | 78 | 474 |
| 062 | `ab49737eda12f816c79d80fa60f74cec3768ef723156b81420c5cf15a1a029f2` | 372 | 93 | 465 |
| 160 | `6300f4550400fc63391ee490cbb8635f468e571296dc84085c24e7aba85b8548` | 313 | 91 | 404 |
| 184 | `156fe12922d290876f63c210f9cec8252e308e1af0512e309cb2fc6fad8928fc` | 389 | 32 | 421 |
| 226 | `342ff4b0df090df3cb1fdea435049e05f9e317f4775af82a14ded63b2a490c13` | 346 | 53 | 399 |
| 338 | `edcac049616e90e42b848d1a719b3af7a4a078b5d1180a3cdf0ecf60e340a01d` | 424 | 2 | 426 |

## Candidate outcomes

- **task025 proportional initializer reuse:** both directions were tested.
  `sigV=[0,1]` and `negsigK=[0,-240]` are algebraically proportional, but
  compensating the K role needs a negative Attention `scale`. ORT produces NaNs
  for that setting, and both candidates fail the first known example with an
  all-off output.
- **task338 Boolean fusion:** the long scalar PRelu products implement exact
  Boolean AND/OR identities and can be written as variadic Min/Max. However,
  node removal changes ORT's reuse schedule for the incumbent's deliberately
  underspecified shape-cloak tensors. The candidate raises a deterministic
  `Slice` shape mismatch (`{1,1,29,30}` versus `{1,1,30,30}`), so it is an
  error candidate and is rejected.
- **task338 CastLike anchor removal:** known-correct, but cost rises from 426 to
  18,423 because the full fp16 cast tensor becomes profiled.
- **task008 int32 anchor removal:** known-correct, but cost rises from 431 to
  454 because two `Cast` outputs cost 24 more bytes than the incumbent profile.
- **task062/160/184/226:** full graph, initializer, shape, and historical-pool
  review found no parameter-only exact substitution. The apparent shared
  constants require shape expansion or charged intermediates and are strictly
  more expensive than their one-to-three-parameter savings.

The machine-readable disposition is in `winner_manifest.json`. Since there is
no cheaper known-pass candidate, no fresh/domain candidate gate was started;
this prevents a known-invalid or runtime-error model from entering the merge
pool.
