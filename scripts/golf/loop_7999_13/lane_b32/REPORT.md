# B32 task219 strict cost reduction

## Outcome

The exact `task219.onnx` member of `submission_base_8002.63.zip` was reduced
from cost **1479** to **1445** without changing its raw output. The cost saving
is 34 and the official-like projected log-score gain is
`ln(1479 / 1445) = +0.023256862164267183`, giving a nominal total of
`8002.653256862164` from the 8002.63 authority.

| Item | SHA-256 |
|---|---|
| Authority ZIP | `a2da30657f3798e861f369ac896f36722ff658ed3e468c4d55db9a04eefbccfc` |
| Exact current task219 | `7a2ead58107803948d316fb8e00c4fd3ff601769309f9ad99661976f1a51bd67` |
| B32 winner | `e6b9793c6c54db7c795c355d82853b6d2100c2992f5c887e7ee34a2ae07a172c` |

No root ZIP, CSV, score pointer, or shared artifact was written by this lane.

## Exact reductions

- Folded eight singleton-only initializer dimensions into broadcast shapes and
  removed six `Unsqueeze` intermediates plus `axes12`.
- Replaced `Cast(bool, uint8) * 98` with `Where(bool, 98, 0)`.
- Replaced `(x & (3*s)) / s` with `(x / s) & 3` for the seven power-of-two
  shifts, reusing the existing constant 3 and removing `cmaskv8`.
- Exhaustively checked both scalar identities over all uint8 input values.
- All remaining initializer flat values are unchanged.

The graph changed from 113 to 105 nodes, 43 to 41 initializers, memory
1253 to 1228, and parameters 226 to 217. A proposed variadic uint8 `Sum` was
discarded because the ONNX full checker rejects that type combination.

## Verification

| Gate | Result |
|---|---:|
| ONNX full checker | pass |
| Strict shape inference with data propagation | pass |
| Runtime-traced node shapes | 105/105 truthful |
| Canonical input/output `[1,10,30,30]` | pass |
| Banned ops / Conv bias / nested graph issues | 0 / 0 / 0 |
| Known corpus, ORT disabled | 265/265, errors 0, raw-equal 265 |
| Known corpus, ORT default | 265/265, errors 0, raw-equal 265 |
| External validator random differential | 500/500 raw-equal, max diff 0, `ACCEPT_STRICT` |
| Fresh 5000, ORT disabled | 4327/5000, runtime errors 0, generation errors 0 |
| Fresh 5000, ORT default | 4327/5000, runtime errors 0, generation errors 0 |

The fresh accuracy is 86.54%, so this is deliberately **not** presented as a
fresh rule reconstruction and does not pass the separate 95% fresh gate. Its
acceptance basis is the permitted current-raw-equivalence branch: complete
known dual-mode equality, external random500 equality, and exhaustive algebraic
identities show that the candidate preserves the exact current member's behavior.

## Historical private-zero exclusion

The historical cost 1081, 1103, 1174, 1191, and 1467 candidates were audited
and excluded. They are different hashes from both the current member and the
B32 winner; their failures include fresh results of 32/500, 9/500, 7/500, and
417/500, plus one known-corpus failure. The cost-1081/1174 models use large
`TfIdfVectorizer` structures, and the cost-1467 model is the archived
`nonadopt` private-zero candidate. None of their lookup weights or shapes were
copied into the winner.

This distinction matters: task219 is listed in the project private-zero ledger
because of those historical cheap candidates. B32 only simplifies the exact
current SHA algebraically. Therefore it preserves whatever leaderboard
contribution the current member has; it does not claim that historical
private-zero evidence has been overturned.

Machine-readable evidence is in `winner_manifest.json`, `winner_audit.json`,
`external500_summary.json`, and the two `fresh5000_*_explicit.json` files.
