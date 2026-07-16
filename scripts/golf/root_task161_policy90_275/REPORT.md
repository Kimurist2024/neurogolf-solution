# task161 cost186 normal-POLICY90 reaudit (lane 275)

## Decision

**FAIL_CLOSED_MARGIN; do not promote.**

The historical candidate clears the requested 90% known/fresh accuracy,
structure, runtime-shape, finite-output, and four-configuration stability
checks.  It nevertheless violates the repository's mandatory clean-margin
rule: actual fresh outputs contain values in the forbidden open interval
`(0, 0.25)`.  This is the sole false gate in `evidence.json`.

No root submission, score ledger, stage directory, `others/71407`, or authority
archive was modified.  Because the candidate failed, this lane did not make a
promotion copy.

## Immutable inputs and actual profiles

| artifact | SHA-256 | memory | params | cost | exact known verifier |
|:---|:---|---:|---:|---:|:---|
| `submission_base_8009.46.zip` | `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927` | - | - | - | immutable container |
| authority `task161.onnx` | `5dc274d8515f1ac2a5c58583197984cd60fa2ede69fbe8206992f98940a38fbe` | 120 | 70 | 190 | true |
| historical candidate | `6752eeea166c8111cda053c3cc36f54b1409d81c7553d672201792f646b31e3a` | 120 | 66 | 186 | false |

Both profiles were independently rerun through `score_and_verify` against the
immutable authority.  The strict reduction is 4 cost units and its projected
score gain would be `ln(190/186) = 0.0212773984473` if the candidate were
admissible.  It is not admissible.

The audit did not consume the past 265/266 or fresh result as evidence.  It
used only the candidate binary, the pinned ZIP, the known corpus, and the exact
task-map-selected generator `task_6cdd2623`.

## Known and fresh results

Every case was executed through eight sessions: candidate and authority under
`ORT_DISABLE_ALL/default × threads 1/4`.  Results were identical across the
four configurations.

| dataset | authority | candidate | candidate accuracy | candidate-added failures |
|:---|---:|---:|---:|---:|
| all known | 266/266 | 265/266 | 99.6241% | 1 |
| fresh seed `275161001` | 9948/10000 | 9935/10000 | 99.35% | 13 |
| fresh seed `275261001` | 9941/10000 | 9932/10000 | 99.32% | 9 |

The fresh streams contain 20,000 distinct accepted input/output pairs, have no
overlap with the known set or each other, and required exactly 10,000 generator
attempts per seed (zero skips).  Each candidate run is comfortably above the
normal-POLICY90 90% threshold.

Failure attribution is explicit:

- known: authority wrong 0; candidate adds the single `arc-gen[67]` regression,
  changing 374 thresholded cells;
- seed `275161001`: authority has 52 generator-rule failures, the candidate
  shares all 52 and adds 13 more;
- seed `275261001`: authority has 59 generator-rule failures, the candidate
  shares all 59 and adds 9 more;
- the candidate repairs no authority failure in these streams.

The authority's 111 fresh failures are its already-described least-frequency
tie weakness.  They are not charged as candidate-added regression.  Conversely,
the candidate's extra 22 failures are separately counted and disclosed.

## Margin blocker

| dataset | authority minimum positive | candidate minimum positive | candidate values in `(0,0.25)` |
|:---|---:|---:|---:|
| known | 0.9762964 | 1.0 | 0 |
| fresh `275161001` | 0.9738151 | **0.0583090** | **13** |
| fresh `275261001` | 0.9738151 | **0.2244898** | **9** |

The counts and minima repeat exactly in all four runtime configurations; the
maximum nonpositive output is 0.0.  Therefore this is deterministic semantic
evidence, not a thread/optimizer fluctuation.  The project playbook requires
on-values to be clearly positive and forbids every raw value in `(0, 0.25)`.
The candidate fails that clean-margin requirement even though the observed
signs are configuration-stable.

## Structural and runtime audit

Both authority and candidate pass full ONNX checking and strict shape
inference with `data_prop=True`.  The candidate is a standard-domain opset-18
graph with three live nodes: two `Einsum` and one `Add`.  It has:

- canonical float32 `[1,10,30,30]` input and output;
- 66 finite, live parameters; largest initializer 60 elements;
- no custom domain, banned/lookup op, nested graph, function, sparse/external
  initializer, dead node, unused initializer, or giant Einsum (15+ inputs);
- no Conv-family node, hence Conv-bias UB0 is vacuous;
- traced runtime shapes `score=[1,10]`, `feat=[2,10]`, and
  `output=[1,10,30,30]`, with zero declared/actual mismatch;
- runtime errors 0, nonfinite values 0, output-shape mismatches 0;
- sign and raw-output mismatches across optimizer/thread configurations 0.

Its two small coefficient tensors and one 60-element spatial coefficient
tensor feed live arithmetic contractions.  There is no indexing, table,
branch, fixture correction, or shape cloak.  task161 is also absent from the
public private-zero catalog; this was evaluated as a normal POLICY90 candidate,
not under any private-zero exception.

## Reproduction and artifacts

From repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  scripts/golf/root_task161_policy90_275/audit_task161_policy90.py
```

Artifacts:

- `audit_task161_policy90.py`: independent audit implementation;
- `evidence.json`: complete machine-readable structure, profile, known, fresh,
  comparison, margin, and gate evidence;
- `REPORT.md`: this disposition.
