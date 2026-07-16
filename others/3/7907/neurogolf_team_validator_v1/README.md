# NeuroGolf Team Validator v1

A portable validator and submission builder designed to keep local results aligned with Kaggle across teammates and GPT sessions.

## What it does

- Audits a 400-model submission ZIP and the 1.44 MiB per-model limit.
- Sanitizes ONNX names exactly before profiling.
- Checks prohibited operators, custom domains, functions, subgraphs, sequences, duplicate type entries, and non-static shapes.
- Runs every train, test, and arc-gen example with ONNX Runtime graph optimization disabled.
- Measures runtime tensor memory from the ONNX Runtime profiler.
- Counts parameters by initializer/Constant elements, including scalar unit cost.
- Computes the projected competition score.
- Performs randomized differential testing against the current best model.
- Compares changed tasks between two submission ZIPs.
- Builds deterministic delta and isolated A/B submission ZIPs.
- Generates a GPT handoff file so a new session starts from the correct baseline and rules.

## Install

```bash
python -m pip install -r requirements.txt
```

## 1. Audit a submission ZIP

```bash
python ngolf_validator.py audit-zip \
  --zip submission_best.zip \
  --out-json zip_audit.json
```

## 2. Validate one candidate ONNX against the current best task

```bash
python ngolf_validator.py validate-task \
  --task 101 \
  --candidate-model task101_candidate.onnx \
  --baseline-zip submission_best.zip \
  --data-dir neurogolf_2026_data \
  --data-zip neurogolf-2026.zip \
  --random-cases 2000 \
  --out-json task101_audit.json
```

A strict candidate should report:

- `candidate.valid = true`
- `known.wrong = 0`
- positive `decision.cost_reduction`
- `decision.verdict = ACCEPT_STRICT`
- zero random threshold mismatches

## 3. Compare two submission ZIPs

By default, only changed tasks are profiled:

```bash
python ngolf_validator.py compare-zips \
  --baseline submission_best.zip \
  --candidate submission_candidate.zip \
  --data-dir neurogolf_2026_data \
  --data-zip neurogolf-2026.zip \
  --random-cases 500 \
  --out-json comparison.json \
  --out-csv comparison.csv
```

Limit the run to selected tasks:

```bash
--tasks 36,101,132,233-235
```

## 4. Build a submission from individual ONNX files

```bash
python ngolf_validator.py build-submission \
  --baseline submission_best.zip \
  --replace 101=task101_candidate.onnx \
  --replace 319=task319_candidate.onnx \
  --output submission_candidate.zip \
  --out-json build_audit.json
```

The builder preserves archive ordering and changes only the named tasks.

## 5. Create an isolated A/B submission

Copy task101 from another ZIP into the current best baseline:

```bash
python ngolf_validator.py isolate-task \
  --baseline submission_best.zip \
  --source submission_with_candidate.zip \
  --task 101 \
  --output submission_task101_only.zip
```

## 6. Generate a handoff for a new GPT session

```bash
python ngolf_validator.py write-handoff \
  --baseline submission_best.zip \
  --score 7982.53 \
  --output GPT_HANDOFF_CURRENT.md
```

Attach the handoff, current best ZIP, comparison reports, and signal ledger to the next GPT session.

## Acceptance tiers

### `ACCEPT_STRICT`

- Baseline and candidate pass all known examples.
- Candidate official-like cost is lower.
- Random executable behavior and threshold output match.

This is the preferred batchable tier.

### `ACCEPT_KNOWN_ONLY`

- Candidate passes all known examples and lowers cost.
- Random differential testing was not run or had no executable cases.

Submit in isolation first.

### `REJECT`

Any known error, invalid graph, non-decreasing cost, random threshold mismatch, or differential executability mismatch.

## Important interpretation

Local profiling is a close estimator, not a guarantee of hidden-test behavior. Even a mathematically exact-looking rewrite can change allocator behavior or hidden generalization. Maintain a leaderboard ledger and isolate changes with uncertain hidden-test impact.
