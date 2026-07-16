# mid20c_87 — third 20-task expansion audit

## Result

No safe score improvement was found. The accepted manifest is empty and the
projected gain is `0.0`. No submission ZIP, CSV, score ledger, or protected
root artifact was modified.

The only authority is `submission_base_8005.17.zip`, SHA-256
`c48fa65401a5bd26d3ed1c556eee8f85c0a2063db313be6b96c73e86159b0a04`.
The 20 targets were tasks 029, 051, 064, 091, 123, 124, 137, 148, 153, 169,
174, 178, 199, 212, 301, 316, 325, 341, 355, and 357.

## Exhaustive history inventory

Every loose ONNX and every matching member of every repository ZIP was
inventoried and SHA-deduplicated. The scan found 800 non-authority SHAs:

- static cost reject: 431
- policy reject: 193
- structural reject: 111
- actual runtime-cost reject: 5
- complete-known reject: 59
- reproducible runtime-crash reject: 1

Sixty-five candidates reached isolated runtime profiling. Fifty-nine reached
complete known-example profiling; none was both known-correct and strictly
cheaper under the official scorer. Therefore no candidate was eligible for the
two-seed fresh gate.

## Important findings

The old helper table listed task091 at cost 126. Profiling the exact authority
member with `score_and_verify` gives cost 265. The scan was run against 265,
not the stale table value. Ten historical task091 models were known-correct,
but their official costs start at 266, so none improves the exact authority.

The apparent one-byte task124 reduction, SHA-256
`2df7617db09373acc416cbf505fff79823fabb22194d7f5c554c56975f43625a`,
omits an unused variadic Split output. It passes static checking but
reproducibly terminates isolated validation with SIGSEGV/exit 139 due to an ORT
allocator/liveness shape mismatch. It is classified as runtime-unsafe.

Tasks169, 174, 178, and 325 are documented private-zero/unsound tasks. Their
harvested SHAs were fail-closed because no complete true-rule/all-input proof
accompanied them. Lookup, giant-initializer/Einsum, private lineage, shape
cloak, non-static shape, schema-invalid negative Conv pads, and Conv-family
bias UB were likewise rejected before admission.

## Exact/no-op reduction audit

The exact authority members were separately checked for unused initializers,
bit-identical initializer reuse, removable zero Conv-family biases,
bypassable Identity nodes, neutral Add/Mul/Sub/Div, single-input Concat,
identity Transpose/Cast/Reshape, and zero Pad. No candidate transformation was
available across these 20 members.

## Evidence

- `result.json`: final decision and authority costs
- `winner_manifest.json`: empty promotion manifest
- `rescreen.json`: all 800 SHA rows
- `inventory/raw.json`: raw inventory metadata
- `inventory/summary.json`: per-task terminal results
- `audit/known_rejections.json`: all 59 known-stage decisions
- `audit/task124_runtime_crash.json`: isolated crash evidence
- `authority_official_profiles.json`: exact authority member profiles
- `mechanical_reductions.json`: exact/no-op audit
