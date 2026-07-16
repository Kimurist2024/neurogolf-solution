# Wave4 true-rule rebuild — tasks 018/233/286/366

## Outcome

No model is eligible for promotion. The exact `submission_base_8004.50.zip`
remains unchanged. **Projected gain: +0.000000**. This lane audited
21 bounded attempts and did not build a ZIP.

## True-rule references

The generator sources and `common.py` were read before model work. Independent
multi-seed fresh checks produced:

| task | stored | fresh | classification |
|---:|---:|---:|---|
| 018 | 266/266 | 1976/2000 (98.8%) | canonical deterministic policy; generator is non-injective |
| 233 | 266/266 | 2000/2000 | exact NumPy reference |
| 286 | 265/265 | 2000/2000 | exact NumPy reference |
| 366 | 266/266 | 2000/2000 | exact NumPy reference |

Task018 cannot have a universally exact deterministic input-only solver: retained
legal generator calls have byte-identical inputs and different outputs. The
canonical spec-derived policy still clears the user's 90% threshold, but every
cheaper ONNX must independently clear the model gate.

## Decisive candidate results

| task | baseline -> candidate | dual-ORT fresh | decision |
|---:|---:|---:|---|
| 018 | 4754 -> 4682 | 0/32, 0 errors | reject: 24-node TfIdf/private-zero lookup, not true rule |
| 233 | 7432 -> 4936 | 0/32, 0 errors | reject: private-zero candidate fails the required fresh100 guarantee |
| 286 | 7481 -> 7122 | 1727/2000 = 86.35%, 0 errors | reject: below policy90 and rcorr lookup lineage |
| 286 | 7481 -> 7263 | retained 4612/5000 = 92.24% | reject: public-fixture correction lookup is prohibited |
| 366 | 7987 -> 7646 | retained 4685/4757 = 98.4864% | reject: 107 shape mismatches; truthful cost is 9465 |
| 366 | 7987 -> 7839 | known dual100 | reject: 100 shape mismatches |

The task018 full 2000 run was not continued after independent 0/32 in both ORT
modes, because private-zero admission requires 100%; one mismatch is already a
terminal rejection. The result is also consistent with the retained 0/5 audit.

## Sound floors reached

- task018's shape-clean, lookup-free known-dual rebuild costs **10857**, above 4754.
- task233's closest spec diagnostic costs **17007**, above 7432 (and still has nine
  nonfatal declaration mismatches); the cheap 4936 graph is fresh-zero.
- task286's correction-free full-row implementation costs **54552**, above 7481.
- task366's metadata-only truthful repair costs **9465**, above 7987. Every
  known-perfect sub-7987 graph in this pool relies on false declared shapes.

These establish the stop condition: additional local shaving cannot cross the
incumbents without reintroducing lookup tables, finite-depth policy failures, or
shape cloaking.

## Evidence

- `reference_evidence.json`: stored and 8-seed fresh NumPy reference results.
- `attempt_audit.json`: actual scorer costs, dual-ORT known results, checker,
  strict data propagation, banned/nested/domain checks, runtime shape traces,
  and Conv-family bias findings for all attempts.
- `audits/`: independent dual-ORT fresh results for tasks 018, 233, and 286.
- `winner_manifest.json`: empty; `rejected_manifest.json`: decisive rejections.

No root ZIP, CSV, score ledger, or baseline model was modified.
