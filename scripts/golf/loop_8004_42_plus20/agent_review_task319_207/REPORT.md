# task319 independent fail-closed review 207

## Decision

**PASS — inherited-cloak exact pass-through.**

The reviewed candidate is exactly the requested file:

- authority: immutable `submission_base_8009.46.zip::task319.onnx`, SHA-256
  `29d5bfe25f86b18e0b5938d85e4f38cca72c34d8aad6390bff43579124d0e391`
- candidate: `agent_task319_exact_201/candidates/task319_combined_runnable.onnx`,
  SHA-256
  `ade6b708b4ee6a0ba65d19e4182748750514435b3b8a005289582154b7208fd4`

The candidate is not metadata-truthful: it retains the authority's 26
declared/runtime shape mismatches.  However, its mismatch signature is exactly
the same 26 tensors under both `ORT_DISABLE_ALL` and default optimization, with
zero newly introduced mismatch.  This review therefore approves it only under
the explicit inherited-cloak policy and the complete-support pass-through
proof below.

Official scoring was independently rerun with the repository's faithful scorer:

| model | memory | params | cost | task score |
|---|---:|---:|---:|---:|
| authority | 863 | 140 | **1003** | 18.089249212038062 |
| candidate | 840 | 138 | **978** | 18.114490329965182 |

The exact improvement is `ln(1003/978) = 0.025241117927118195`.

## Independent runtime audit

Raw, pre-threshold output arrays were compared, not merely their signs.
Independent fresh seeds were `319207011` and `319207029` (1,500 generated cases
each).

| ORT configuration | known raw equal | fresh raw equal | errors | nonfinite |
|---|---:|---:|---:|---:|
| disable-all, threads=1 | 267/267 | 3000/3000 | 0 | 0 |
| disable-all, threads=4 | 267/267 | 3000/3000 | 0 | 0 |
| default, threads=1 | 267/267 | 3000/3000 | 0 | 0 |
| default, threads=4 | 267/267 | 3000/3000 | 0 | 0 |

Both models were 267/267 on known cases and 2919/3000 on this independent fresh
set.  All 81 fresh misses are inherited authority behavior; the candidate is
raw-identical on every one.  Candidate outputs had minimum positive value 1,
maximum 127, and no runtime error or nonfinite value.

An additional two-mode intermediate trace covered 512 cases.  It checked each
rewrite boundary directly: transposed correlation operands, transposed/halved
QLinearConv map, halved ReduceMax, unchanged selected count and predicate,
equal alternate index, equal selected bits, equal target color, equal terminal
weights, and equal final output.  Every relation had 0 failures under both
disable-all and default optimization.

## Complete generator-support proof of the five rewrites

### 1. Transpose both correlation operands

For the square 5x5 operands and pads `[0,0,2,2]`, which are symmetric under
row/column exchange,

`Corr(A.T, B.T)[i,j] = Corr(A,B)[j,i]`.

This follows by renaming the row and column summation indices.  The immediately
following `ReduceMax` reduces both spatial axes, so transposing the correlation
map cannot change its result.  The independent checker also evaluated all
`25 × 25 = 625` input/kernel basis-position pairs with zero failure.

### 2. Replace `3 - Where(cond,1,2)`

The complete boolean truth table is:

| cond | authority | candidate |
|---|---:|---:|
| false | 1 | 1 |
| true | 2 | 2 |

The candidate's scalar Gather removes one singleton rank, but the subsequent
singleton-condition broadcasting restores the same selected values before the
final transpose.  Both conditions were also evaluated with distinguishable
symbolic row values.

### 3. Absorb the QLinearConv factor

The two correlation operands are clipped to binary and are 5x5, hence their
overlap is `S ∈ [0,25]`.

Every non-background generator color occupies at most `4 × 5 × 5 = 100`
cells.  The graph zeros the ArgMax channel before TopK.  If ArgMax is the
background, only non-background counts remain; if it is not, the background
count is smaller than a non-background maximum and is therefore also at most
100.  Thus the selected count is always `C ∈ [0,100]` on complete generator
support.  Float16 represents these integer counts exactly and the uint8 cast is
lossless.

Therefore:

- authority QLinearConv value `8S ≤ 200` does not saturate;
- authority left shift `2C ≤ 200` does not overflow;
- candidate QLinearConv value `4S ≤ 100` does not saturate;
- `8S ≥ 2C` iff `4S ≥ C`.

All 2,626 pairs in `[0,25] × [0,100]` were exhaustively checked with the actual
uint8 saturation/wrap formulas and had zero predicate difference.

### 4. Keep the base predicate rank

The candidate keeps the singleton predicate as `[1,1,1,1]`.  Broadcasting it
over the original row `[1,1,5]` and scalar-Gather row `[1,5]` yields
`[1,1,1,5]`; transposing the last two axes gives `[1,1,5,1]`, exactly the
authority's Unsqueeze result.  The target-color Where similarly preserves its
single selected value while restoring singleton rank.  The observed
`safe_name_113` and `safe_name_117` arrays were exact in all 1,024 two-mode
traces.

### 5. Replace terminal Equal/Where with ScatterElements

Both indices are in `[0,9]`: one comes from ArgMax across exactly 10 channels,
and the other selects an index returned by TopK across the same 10 channels.
Each Scatter update/index tensor contains one element and uses axis 0.

Starting from ten ones, the first Scatter writes 0 at the ArgMax index and the
second writes 2 at the selected target index.  This exactly matches
`Where(target, 2, Where(background, 0, 1))`.  Importantly, it remains exact even
if the indices coincide because the second Scatter reproduces the outer
target-Where priority.  All 100 possible index pairs were exhaustively checked
with zero failure.

These five arguments are local algebraic identities on the complete generator
support.  They do not assume that the authority chooses the truthful answer in
non-injective cases.  Consequently the candidate preserves the authority's raw
output for every generated input.

## Structural and mismatch gates

Both authority and candidate pass:

- full ONNX checker;
- strict shape inference, including `data_prop=True`;
- canonical single input/output and positive static metadata;
- standard-domain, no banned/Sequence op, no function, sparse initializer,
  nested graph, or external data;
- finite initializers and Conv/QLinearConv bias UB0;
- official filesize limit.

The runtime-shape audit used 64 cases in each optimization mode.  Authority and
candidate each had the same 26 inherited mismatches, no runtime errors, and no
nonfinite intermediate values.  Candidate-minus-authority mismatch set size is
zero in both modes.

Machine-readable evidence and the reproducible independent checker are in
`audit.json` and `audit.py` in this directory.  No root submission, ledger,
`others/71407`, or lane-201 file was modified.
