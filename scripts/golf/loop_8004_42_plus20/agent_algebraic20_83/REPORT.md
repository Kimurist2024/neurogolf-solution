# Algebraic initializer-absorption audit (8005.17 authority)

## Outcome

No candidate is eligible under the requested zero-nonfinite and zero-near-
positive gates.  This lane performs no merge and contributes `+0.000000`.

The authority is `submission_base_8005.17.zip`, SHA-256
`c48fa65401a5bd26d3ed1c556eee8f85c0a2063db313be6b96c73e86159b0a04`.
No protected root file or ZIP was modified.

## Search breadth

- 30 eligible task files in task150--400 were audited, selected by maximum
  Einsum operand count (39--135 operands).
- The audit classified sign/`x^2=1`, one-hot, diagonal, identity, signed
  permutation, duplicate operand, duplicate initializer, and zero-slice
  structures.
- 34 latent-component deletion candidates across tasks
  163/175/199/229/232/304/315 were tested.  Every candidate has a concrete
  deterministic real input whose raw output differs from the authority, so
  every one is rejected as non-equivalent.
- All broad exact scanners produced only two in-scope strict-lower constant
  contraction candidates: task328 and task379.

The 30 audited source tasks and all per-initializer classifications are in
`result.json` under `inventory.rows`.  The 34 explicit counterexamples are in
`latent_prune_audit.rows`.

## Exact candidates

| task | actual cost | exact real algebra | known four-config result | numeric gate | decision |
|---:|---:|:---:|:---:|:---:|:---|
| 328 | 558 -> 554 | yes | retained 267/267 in both ORT modes | minimum positive `7.316870026530253e-11` | reject: near-positive |
| 379 | 1949 -> 1947 | yes | 266/266 in each of four configs; raw identical to authority | near-positive 0, but 12,896 `-inf` values per config | reject: nonfinite |

### task328

The removed vector `e[4]` is exactly
`einsum("baa->a", J[2,4,4])`.  All four `e` occurrences in the 58-operand
Einsum are replaced with corresponding contractions of `J`; all other graph
topology and constants are unchanged.  This proves equality term by term for
all real dynamic inputs.  Actual cost is 554 (`memory=200`, `params=354`),
strictly below 558.

The exact candidate SHA is
`4d0fc5264833fbf46609fde690ad8635e208a2cec381e749b5707ef828866cb2`.
Retained SHA-bound evidence reports 267/267 known cases in both ORT modes but
positive raw values down to `7.316870026530253e-11`; the candidate therefore
cannot meet the open-interval `(0,0.25)` zero-count gate.

### task379

The removed vector `QRow1_2[2]` is exactly
`einsum("bca->a", WBasis[2,2,2])`.  Its only occurrence is substituted inside
the final Einsum, and every other operand, term, node, and initializer is
unchanged.  This is an all-input real-algebra identity.  Actual cost is 1947
(`memory=1570`, `params=377`), strictly below 1949, for a potential gain of
`0.0010266941353610753`.

The exact candidate SHA is
`854c63d966310949803391cf4c019b02a9c0f2a53578257fee5898386e53cf64`.
It passes full checker, strict shape inference with data propagation, all
runtime/declaration shape checks in disabled and default ORT, standard-domain
checks, and Conv-bias UB=0.  Complete known results are identical in all four
configurations (`ORT_DISABLE_ALL`/`ORT_ENABLE_ALL`, threads 1/4):

- 266/266 correct, runtime errors 0;
- raw output bit-identical to the authority for 266/266 cases;
- near-positive values in `(0,0.25)`: 0; minimum positive: 0.25;
- NaN: 0, `+inf`: 0, `-inf`: 12,896.

Because the stated gate requires zero nonfinite values rather than merely no
new nonfinite values, task379 is not listed as a winner.

## Artifacts

- `audit_algebraic20.py`: reproducible Codex-written audit.
- `result.json`: complete inventory, counterexamples, proofs, costs, structure,
  and four-configuration known evidence.
- `winner_manifest.json`: empty winner list and explicit no-merge status.
- `candidates/`: SHA-bound diagnostic exact candidates; neither is approved.

