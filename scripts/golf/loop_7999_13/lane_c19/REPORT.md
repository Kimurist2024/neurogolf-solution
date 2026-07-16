# Lane C19 — task018 / task145 exact-7999.13 re-audit

## Outcome

No candidate was promoted. The exact archive remains unchanged and this lane
adds **0.000000** projected score.

The lane deliberately stopped before fresh5000 because no candidate passed the
cheaper + stored-correct-in-both-ORT-modes + shape-clean prerequisites. Running
the expensive final gate on an already disqualified candidate would not change
the adoption decision.

## Exact baseline

| Task | Exact-archive SHA-256 | Actual cost | Stored result |
|---|---|---:|---:|
| 018 | `3e2fd5348df3b6e64a1367c8d1a20161ff0e13b098dcedd31bc07cc8b92eb4ea` | 4818 | 266/266 under `ORT_DISABLE_ALL` |
| 145 | `fe83ed6befda07db144dd4d565fae8e2dd92444b0f3b0d52bdbb2778b45a8fef` | 5132 | 267/267 in both modes |

The baseline files came directly from `submission_base_7999.13.zip`, not from
the dirty handcrafted tree.

## Task 018

The authoritative generator is `task_0e206a2e.py`. A visible complete creature
and three transformed clone markers do not always uniquely determine which
generator rotation was used. The retained witness was rerun in this lane and
constructed two legal generator calls with byte-identical inputs and different
outputs:

```text
inputs_equal=True
outputs_equal=False
both_inputs_match_fresh=True
```

Therefore a deterministic ONNX cannot be 100% exact on the unrestricted
generator. This is not just theoretical: the exact-archive baseline is 96/100
on the retained fresh100 audit.

Two cheaper stored-correct graphs were re-audited. Cost 4733 has 57 declared / 
runtime shape mismatches; cost 4791 has 59. Both fail default-ORT session
creation, strict shape inference, and retained fresh audits. The fully
shape-clean rebuild passes both ORT modes but costs 10857 and is still 14/1000
wrong. None is a sound replacement for cost 4818.

## Task 145

The authoritative generator is `task_6455b5f5.py`: red guillotine separators
partition the grid into black rectangles; every global minimum-area leaf becomes
cyan and every global maximum-area leaf becomes blue, including ties.

The closest historical file re-audits at cost 5147, is wrong on 11/267 stored
examples under `ORT_DISABLE_ALL`, fails all 267 under default ORT, and has 64
runtime/static shape mismatches. Nine new crop-fusion variants also failed to
produce a valid score.

The independent spec-derived numeric rebuild is the sound reference: both ORT
modes 267/267, full checker and strict data propagation pass, zero runtime shape
mismatches, and retained fresh3000 is exact. Its actual cost is 10175, however,
5043 above the exact-archive incumbent. The retained architecture analysis gives
a conditional honest memory floor of about 8400 before scalars/parameters, also
above 5132. No safe sub-5132 graph was found.

## Structural / policy audit

`candidate_audit.json` records actual scorer cost, complete stored fixtures in
both ORT modes, full checker, strict shape/data propagation, runtime/static shape
agreement, domains, functions, sparse initializers, banned operations, lookup
red flags, and Conv bias consistency. The audited candidates use no nonstandard
domain, nested graph, function, sparse initializer, banned operation, giant
Einsum, TfIdfVectorizer, Hardmax, or unsafe Conv bias.

## Deliverables

- `winner_manifest.json`: empty, gain 0
- `rejected_manifest.json`: candidate-level rejection evidence
- `candidate_audit.json`: current actual-cost and structural audit
- `fresh_evidence.json`: generator / fresh / ambiguity evidence
- `historical_scan_summary.json`: prior candidate-family scan
- `validation/root_integrity.json`: exact root archive integrity

No root ZIP, CSV, score ledger, or handcrafted ONNX was edited by C19.
