# Exact micro-optimization Wave 2

## Outcome

- Baseline: `submission_base_8004.50.zip`
- Baseline SHA-256: `63cb4c2abf794bb3cc0ceb531db907625c82638656e7d1ab29865d39b42a6cac`
- Scanned: **400/400 tasks**
- Accepted: **0**
- Projected gain: **+0.0**
- ZIP integration: **not performed**
- Final verdict: **NO_SAFE_EXACT_CANDIDATE**

## Static scan

The scan covered byte-identical initializer aliases, output-unreachable code,
internal no-op Identity/Cast/Reshape nodes, duplicate deterministic producers,
unused optional secondary outputs, and truthful annotation-only shape
reductions.  It found:

| Class | Opportunities |
|---|---:|
| Initializer alias | 1 |
| Dead code | 5 |
| Identity/Cast/Reshape no-op | 16 |
| Duplicate producer | 2 |
| Dead optional output | 4 |
| Annotation reduction | 0 |

Thirteen baseline members do not support strict data-propagating shape
inference and were ineligible for this lane.  Most opportunities were either
catalogued private-risk/shape-cloak lineage or failed the candidate structural
gate.  Only tasks 124 and 165 reached runtime validation.

## Runtime decisions

| Task | Rewrite | Static / actual result | Validation | Decision |
|---:|---|---|---|---|
| 124 | omit unused Split output `r3` | static `-1` cost | validator process SIGSEGV, exit 139 | **REJECT** |
| 165 | reuse duplicate CastLike producer | static `-1`; profiled 592 -> 551 | candidate known 0/265, runtime errors 265 | **REJECT** |

Both failures are allocator/liveness effects.  They confirm that computational
equivalence at the ONNX operator level is insufficient for these incumbent
graphs: removing or aliasing a value changes ORT buffer reuse and makes runtime
shapes collide.  Neither candidate was run on fresh generator cases after the
mandatory runtime/known gate failed.

## Past failures intentionally not repeated

- Dead-node removal on tasks 039/089/111/122/183 previously caused complete
  known-set runtime failure.
- task048 missed the 95% fresh gate (1818/2000).
- task233 is a repeated private-zero dust-gain lineage.
- task333 changes a giant floating Einsum contraction.
- task285/289 shape-cloak paths were excluded.

The binary files under `candidates/` are **rejected evidence only**.  The only
authoritative integration input is `winner_manifest.json`, which is empty.
