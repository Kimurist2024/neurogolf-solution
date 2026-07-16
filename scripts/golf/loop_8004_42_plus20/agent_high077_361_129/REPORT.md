# Lane 129 — task077/task361 sound strict-lower audit

## Outcome

No safe strictly cheaper candidate was found. Winner count is **0**, cost delta
is **0**, and projected score gain is **+0.0**. No candidate file is admitted.

The 8009.46 authority archive remained SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
throughout the run. This lane did not edit `submission.zip`,
`all_scores.csv`, `others/`, or shared `artifacts/`.

## Independent generator rules

`run_lane.py` reimplements both rules independently of all candidate ONNX
graphs and past candidate files:

- **task077 (`36fdfd69`)**: find the separated red-supported rectangles of
  height 2–3 and width 2–7. A valid rectangle has red support on every boundary
  row/column and no adjacent pair of red-empty columns. Restore every
  static-colored cell covered by such a rectangle to yellow.
- **task361 (`e40b9e2f`)**: infer the rotation center from the
  generator-mandated full 3x3 core, or the full 2x2 core in the half-cell-center
  case, and complete every colored pixel's fourfold rotation orbit.

Both references pass all known cases and two independent fresh streams:

| task | known | fresh seed A | fresh seed B | errors |
|---:|---:|---:|---:|---:|
| 077 | 266/266 | 1500/1500 | 1500/1500 | 0 |
| 361 | 266/266 | 1500/1500 | 1500/1500 | 0 |

This is 6000 fresh reference cases in total. Exact seeds and counters are in
`audit/reference_audit.json`.

## Authority members

| task | authority member SHA-256 | scorer cost | known disabled/default | truthful runtime | authority fresh |
|---:|---|---:|---|---|---|
| 077 | `db46560f4e633e057f960fe2db040f62b118051fc1bbbfc6e29871fcd0e84d56` | **3364 = 3274 memory + 90 params** | 266/266 both | **10** declared/actual shape contradictions; 10,000 measured intermediate bytes | **2966/3000** per ORT mode, 34 wrong, 0 errors |
| 361 | `d606fcf6e11548c562db31cd942be67071b5932b9510a3858f87dd9ca4f315e4` | **844 = 820 memory + 24 params** | disabled 266/266; default session failure | **11** declared/actual contradictions; 48,442 measured intermediate bytes | not run because both ORT sessions are mandatory |

The task361 default failure is the malformed `CenterCropPad` shape contract:
the supplied shape has one element while the operation declares two axes.
Static full checker and strict data-propagating inference pass the unmodified
member, but direct runtime shape tracing is decisive and fails the truthful
shape requirement.

The task077 authority is also not a SOUND reference implementation. Both ORT
configurations make the exact same 16 and 18 mistakes on independent 1500-case
streams. Therefore algebraic equivalence to this incumbent cannot establish
private safety.

## Freshly regenerated mechanical scan

No past candidate ONNX was reused. All eight byte-distinct variants were
regenerated directly from the current authority members with transformation
proof records:

| task | transform | result |
|---:|---|---|
| 077 | Identity bypass | full checker and strict inference fail at `cp_1`: inferred dimension 21 conflicts with declared 1 |
| 077 | constant fold | ten `CenterCropPad` contracts fail with inferred dimensions 21–30 versus declared 1 |
| 077 | combined | same ten truthful-shape failures |
| 077 | metadata normalization | unscorable |
| 077 | normalized combined | valid/scorable only at **9945**, above 3364 |
| 361 | constant fold | full/strict failure: malformed `CenterCropPad` axes plus `idx2_src` 2-vs-1 contradiction |
| 361 | metadata normalization | unscorable |
| 361 | normalized combined | unscorable |

Stage totals are
`REJECT_CHECKER_OR_STRICT_SHAPE=4`,
`REJECT_UNSCORABLE=3`, and
`REJECT_NOT_STRICTLY_LOWER=1`.

The apparent task077 Identity is not a free semantic node in this scored graph:
its declared shape is part of the incumbent's shape-accounting construction.
Removing it exposes the 21-vs-1 contradiction. It is therefore rejected before
known/fresh testing rather than inheriting the incumbent's LB behavior.

## Independent history boundary

History was read only as negative evidence; no historical payload was copied
or submitted.

- **task077:** the independently audited spec-derived exact rebuild has SHA-256
  `e78b782afb2428019d59949562692ba164bbe5b93e70853581c56ba594da8992`,
  known 266/266, two fresh streams of 2000/2000, and scorer cost **17653**.
  It is sound but 14289 cost above the current authority, so it cannot satisfy
  strict-lower.
- **task361:** the retained cost-810 history is genuinely lower but only
  198/266 known. The cost-854 history is not lower and only 218/266. Known
  perfect retained variants reprofile at 1004/1006/1363 and fail default ORT
  on malformed `CenterCropPad` shapes. None reaches the required intersection.

Thus there is no candidate with a meaningful new cost/SHA/gain to report:
**new cost = none, admitted SHA = none, gain = +0.0**.

## Evidence

- `audit/results.json`: baseline official costs, full/strict checks,
  dual-ORT known results, direct runtime-shape traces, baseline fresh runs, all
  regenerated variants, and fail-closed stages.
- `audit/reference_audit.json`: independent known/fresh true-rule proof.
- `audit/history_evidence.json`: selected read-only history boundary.
- `manifest.json`: exact authority hashes and empty winner list.
- `baseline/`: byte-exact snapshots of the two authority members.
- `candidates/`: rejected regenerated intermediates only.
- `candidate/README.md`: explicit no-winner marker.
- `run_lane.py`: reproducible end-to-end audit.
