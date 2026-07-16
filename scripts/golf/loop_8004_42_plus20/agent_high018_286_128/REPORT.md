# high018_286_128 — task018 / task286 strict SOUND audit

## Outcome

**Safe winners: 0. Verified gain: +0.000000.** Neither authority member has a
strictly cheaper candidate that passes the requested SOUND gates. No root ZIP,
score ledger, `others/`, or `artifacts/` file was changed.

Authority: `submission.zip`, SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
(the immutable 8009.46 lineage).

| task | authority SHA-256 | measured cost | full/strict | runtime shape | four-config known | decision |
|---:|---|---:|:---:|:---:|:---:|---|
| 018 | `36eef8bf93f0e79ea8e730594a986add071ee30ea92fc8f662c61663bd6c52c3` | 4753 = 4520 memory + 233 params | full yes / strict **no** | **false** | disable-all 266/266; default session fails | reject shape cloak/default ORT failure |
| 286 | `06fd64e351d4332c0419908456002128730921a4dd80e019befc482ba2ee600d` | 7481 = 7116 memory + 365 params | yes | true | 265/265 in disable-all/default, threads 1/4 | not a SOUND source: finite-depth core plus fixture-signature corrections |

## task018

The exact authority member fails strict data-propagating shape inference at
`px_mv_i8` (`Gather` axis/rank error), has 61 independently recorded
declared/runtime shape contradictions, and cannot create a default-optimized
ORT session because `safe_name_72` requests a TopK larger than the inferred
axis. Its disable-all behavior is 266/266, but the explicit gate rejects shape
cloaks and default-ORT failures before any cost shave can be admitted.

The graph inventory found no dead node, unused initializer, identical
initializer alias, duplicate node, or unused optional output. Therefore there
is no local exact removal even before the mandatory truthful-shape gate.

The generator also has a stronger obstruction to a deterministic true-rule
rebuild: legal generator calls with byte-identical inputs and different outputs
exist because the three visible marker colors can leave the clone rotation
ambiguous. The retained generator-derived canonical policy is 1976/2000 fresh,
not 100%. A lookup-free, known-dual, shape-truthful control has SHA
`efc964b3627445dd60e7df6c6a3ea1837866014cc05d1d712b5c5e4633b3da8e`
and cost 10857, which is 6104 above the authority and still cannot remove the
non-injective mapping fact. The only sub-authority shape-clean history model
was a 24-`TfIdfVectorizer` fixture lookup and scored 0/32 fresh in both ORT
modes; it was not reused.

## task286

The authority graph is structurally and runtime-shape clean, but nodes
991--1005 contain explicit `ex_rcorr_*` signature comparisons and an
`ex_add_table[5,1,16,1]` public-fixture correction around a finite-depth packed
flood. Thus a decision-identical shave would inherit the prohibited
private-zero lineage. This is exactly the family already shown black on fresh
data; no prior approximate candidate was reused.

The exact local scan found no dead node, unused initializer, initializer alias,
duplicate node, scalar-broadcast reduction, or semantic no-op. The only
algebraic leads were the unused trailing outputs `V_12` and `S_12` of two Split
nodes. Omitting either output or both leaves every reachable tensor unchanged
at protobuf graph level, and all three probes pass full checker and strict
data-propagating inference. However, each probe exits with native signal 11 in
all three isolated gates:

- official scorer profile;
- ORT_DISABLE_ALL zero-input execution;
- default-optimized ORT zero-input execution.

The unmodified authority is the control and exits normally in all three. These
probes are therefore runtime-invalid, have no valid measured cost, and are not
candidates. Their SHA values and isolated crash traces are in
`optional_probe_build.json` and `optional_probe_audit.json`.

The correction-free true-rule implementation remains the physical-row bitset
rebuild, SHA
`a70c361b2583d65dbdecfec89707c9a94f8c35102159510bf1c006e99fcd334f`.
It implements unrestricted cardinal flood through every non-cyan cell,
dynamic pair-color recovery, and absolute checkerboard parity; it passed all
265 stored cases, fresh 1000/1000, full/strict checks, and has truthful runtime
shapes. Its measured cost is 54552 (53400 memory + 1152 params), 47071 above
the authority. The exact-rewrite audit found no reduction in that sound graph.

## Evidence and stop condition

- `authority_inventory.json`: authority SHA, scorer cost, structural audit,
  runtime trace, full node/initializer inventory.
- `current_known_four_configs.json`: known behavior in both ORT modes and
  threads 1/4.
- `exact_variant_scan.json`: fresh local exact scan from the authority member.
- `optional_probe_build.json`, `optional_probe_audit.json`: formal unused-output
  proof and isolated native-runtime rejection.
- `manifest.json`: machine-readable final decision and provenance.

The stop condition is reached: task018 cannot pass the mandatory runtime-shape
and default-ORT gates below 4753, while task286 has no executable exact local
shave and every known sub-7481 rule approximation is correction-table/private-
zero lineage. The only verified general task286 rule engine is far above the
cost threshold.
