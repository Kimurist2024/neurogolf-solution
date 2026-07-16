# task192 focused gold rebuild

## Result

The strict true-rule winner set is empty.  No new task192 graph satisfies the
required combination of official gold, stable margin, lower cost, and
independent fresh 2000/2000 truth accuracy.

There is, however, a separate **LB-equivalent** cost reduction that preserves
the already LB-white cost-444 authority's decisions:

| model | memory | params | cost | projected gain |
|---|---:|---:|---:|---:|
| `submission_base_8014.69.zip::task192.onnx` | 200 | 244 | 444 | — |
| `candidates/task192_authority_equivalent_factored.onnx` | 160 | 243 | 403 | +0.096888 |

Candidate SHA-256 is
`65cd5ab8af8de6495d6507e4db826ffd55e9ba21419efd4a3aaecd48ed786164`.
It passes the full checker, static shapes, local and official gold, and stable
margin (`min positive = 7.953127861`).  On 2,000 generated grids at seed
777192 it has **0 sign-decision differences** from the LB-white authority; the
largest raw numerical difference is only `0.00048828125`, versus candidate
minimum nonzero absolute raw value `0.6487186551`.

The rewrite is algebraic.  It replaces `[I,N,S]` materialization with
`G=[I,S]`, reconstructs the same color routes from products of `[I,N]` and
`G`, and factors two float32-exact proportional route rows into a two-element
path coefficient and one shared 2x2 route.  The build script asserts exact
float32 reconstruction of the authority route initializer.

This cost-403 graph is **not** a true-rule admission: just like the retained
LB-white authority, it fails 62/2000 generator-truth cases (96.9%).  It should
only be used if behavior-preserving equivalence to the known LB-white member is
accepted as the safety proof.  Under the requested 2000/2000 truth gate it
remains excluded.

## True-rule attempt

`candidates/task192_rank8_gap_exact_argmax.onnx` compiles an exact ArgMax
selector and a trained rank-8 spatial code at cost 442.  It passes official
gold and margin.  Counterexample-guided training made the fixed verifier seed
pass 2000/2000, but untouched seeds continued to expose rare failures:

- seed 192433001: 5/2000 failures before adding its counterexamples;
- seed 192433002: 2/2000 failures;
- seed 192433003: 3/2000 failures;
- seed 192433004: 5/2000 failures.

It is therefore rejected; passing a trained verifier seed is not treated as a
generator guarantee.

## Authority safety

This lane did not edit `submission.zip`, `submission_base_8014.69.zip`,
`all_scores.csv`, or `best_score.json`.  `try_candidate.py` promoted the
behavior-preserving cost-403 graph only to `artifacts/handcrafted/task192.onnx`.

Machine-readable evidence is in `authority_equivalence.json`, `result.json`,
`build.json`, `rank8_gap_build.json`, and the CEGIS audit JSON files.
