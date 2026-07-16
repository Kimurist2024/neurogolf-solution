# A14 strict cost audit

## Outcome

- Exact source: `submission_base_7999.13.zip`
- Source SHA-256: `a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`
- Tasks: 119, 153, 161, 174, 183, 190, 243
- Retained candidates: 25
- Exact-byte-distinct loose historical models: 206
- Safe winners: 0
- Verified gain: `+0.000000`

No root ZIP, CSV, score pointer, or ledger was modified by this lane.

## Candidate result

| Task | Exact cost | Retained result |
|---:|---:|---|
| 119 | 142 | The only lead retains an 18-22 input giant Einsum. |
| 153 | 237 | Cost 227 fails known data; cost 236 is QLinearConv bias UB. |
| 161 | 190 | Cost 186 fails complete known data. |
| 174 | 240 | Two giant-Einsum candidates; standard candidates cost 259 and 278. |
| 183 | 162 | Cost 91 fails known/runtime; known-correct alternatives cost 188-9221. |
| 190 | 153 | All eight leads retain a 25-input giant Einsum. |
| 243 | 177 | The sole lead retains an 85-input giant Einsum and actually costs 626. |

No candidate is simultaneously cheaper, complete-known correct, and
structurally eligible. Fresh validation was therefore not run.

## Required task153 UB audit

The apparent `237 -> 236` saving is exactly the removed tenth QLinearConv bias
element. The node produces ten output channels but the candidate supplies bias
`B` with only nine elements. The strengthened Conv-family checker reports:

`('QLinearConv', 'B', 9, 10)`

This depends on an out-of-bounds/uninitialized read and is rejected before
fresh testing. The independent C11 audit reached the same conclusion. Restoring
the tenth bias removes the nominal gain.

## Full loose history

The targeted repository-wide pass deduplicated 206 models by task and exact
SHA-256:

| Task | Unique | Result |
|---:|---:|---|
| 119 | 36 | 20 sound screens are not cheaper; 15 are structural rejects. |
| 153 | 25 | One cheaper full profile is known-wrong; the UB model is a structural reject. |
| 161 | 20 | The sole cheaper full profile is known-wrong. |
| 174 | 32 | 28 sound screens are not cheaper; three are structural rejects. |
| 183 | 31 | 27 sound screens are not cheaper; one is unscorable and two are structural rejects. |
| 190 | 36 | Ten sound screens are not cheaper; 25 are structural rejects. |
| 243 | 26 | Fifteen sound screens are not cheaper; ten are structural rejects. |

The archive inventory additionally covers 1,195 ZIPs, 224,111 ZIP members,
and 118,938 loose observations globally. The 25 lowest retained leads for these
tasks were independently reprofiled in this lane.

## Current-model analysis

All exact members pass ONNX full checking, use standard domains, have no Conv
bias finding, and contain no unused or identical same-shape initializer pair.
Tasks 153, 161, and 183 are non-giant standard graphs; their only cheaper leads
fail known correctness or UB safety. Tasks 119, 174, 190, and 243 retain giant
Einsums in the current baseline, so a new candidate must remove that structure
rather than inherit it.

With strict shape inference plus data propagation, exact task243 reports a
pre-existing contradiction at Reshape `r_u`: inferred dimension 30 conflicts
with declared dimension 1. The baseline is left untouched, and the retained
task243 candidate is independently ineligible due to its 85-input Einsum and
cost increase.

## Admission disposition

Fresh 5000/5000 can only confirm a candidate that is already cheaper,
known-complete, structurally standard, bias-safe, and free of shape/value-info
cloaks. That prerequisite set is empty. `winners` is therefore empty in
`final_manifest.json`.
