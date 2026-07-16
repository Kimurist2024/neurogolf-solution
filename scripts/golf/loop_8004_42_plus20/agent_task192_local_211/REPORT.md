# task192 local exact / POLICY90 pass-through lane 211

## Outcome

**ACCEPT `POLICY90_INHERITED_RAW_PASS_THROUGH`**

- staged source: `others/71407/task192.onnx`
  - SHA-256: `e6515b2ddf32c2eb80581aa3267e24683d2aa53d9445483b2a2a0752f94072d5`
  - profile: `memory 200 + params 938 = cost 1138`
- candidate: `candidates/task192_policy90_center_direct.onnx`
  - SHA-256: `1200fe8473c045ec89abaaf1860d1d0758316523855c9ff13d4c3fc092412047`
  - profile: `memory 200 + params 934 = cost 1134`
- strict reduction: **-4 cost**
- projected incremental gain: **+0.0035211303985789606**

This candidate does not introduce a new approximation. It keeps the staged
`HardSigmoid(alpha=1,beta=-33)` selector and the 30x30 adjacency byte-identical,
and only changes an exact coefficient basis. Its all-support status is therefore
the same disclosed `POLICY90 / not all-support exact` status as the admitted
source. Root submissions, ledgers, and `others/71407` were not edited by this lane.

## Exact basis identity

The source shares this primitive basis:

```text
[nonzero, background, selected]
```

and recovers:

```text
center   = [inside, nonzero]
neighbor = [inside, selected]
route    = [background, -9*background + selected]
```

The candidate stores the fixed pair `[inside, nonzero]` directly as
`center_basis`, then concatenates the unchanged selected vector:

```text
basis = [inside, nonzero, selected]
```

The center consumes `center_basis` directly. The two remaining coefficient
maps recover:

```text
neighbor = [inside, selected]
route[0] = inside - nonzero = background
route[1] = -9*inside + 9*nonzero + selected
         = -9*background + selected
```

The histogram selects the nonzero row with `[0,1]`; it is identical to the
source histogram. The identity was enumerated for all ten possible selected
one-hot vectors. Center, neighbor, and route factors matched exactly in every
case. The final polynomial and adjacency are unchanged.

This removes six map elements and adds a two-element histogram selector, for a
net four-parameter reduction with no memory increase.

## Fail-closed verification

Static gates:

- full ONNX checker: pass
- strict shape inference: pass
- strict inference with data propagation: pass
- standard-domain opset 18 only
- functions / sparse initializers / nested graphs: 0 / 0 / 0
- banned ops, `Hardmax`, unused initializers, nonfinite initializers: 0
- Conv-family nodes and short-bias UB: 0
- runtime typed trace: 4 tensors, shape mismatch 0, nonfinite 0

Known 265 cases, optimization disabled/default and threads 1/4:

- source right: 265/265 in every configuration
- candidate right: 265/265 in every configuration
- raw-bitwise equality: 265/265 in every configuration
- errors / nonfinite: 0 / 0

Independent fresh streams:

| seed | cases | source/candidate right | raw equality across four configs |
|---:|---:|---:|---:|
| 21119271 | 2500 | 2498 (99.92%) | 10,000 / 10,000 |
| 21119289 | 2500 | 2498 (99.92%) | 10,000 / 10,000 |

Across known plus fresh, all **21,060 case-config comparisons** are raw-bitwise
equal. Runtime errors and nonfinite values are zero. The four fresh misses are
identical source misses caused by the inherited count-33 policy; candidate-specific
regressions are zero. Both independent streams remain far above the user's 90%
admission threshold.

Machine-readable evidence: `audit_policy.json`. Reproducer: `audit_policy.py`.

## Exact ArgMax investigation

The exact fallback was reduced from 1149 to **1143**:

- candidate: `candidates/task192_center_direct_argmax.onnx`
- SHA-256: `5c5eaefa81acce481dbc93855dbcc2f9ef821e055f8c982eadcd07f63c764a9d`
- profile: `memory 208 + params 935 = cost 1143`

It uses exact first-tie ArgMax+OneHot and the same coefficient identity. It is
raw-bitwise equal to the cost-1149 exact fallback on known 265/265 in all four
ORT configurations, with errors/nonfinite zero. It is rejected because 1143 is
five cost above the active 1138 source; fresh was intentionally cost-gated.
Evidence: `audit_exact_control.json`.

The algebraic Hardmax control reaches cost1134, but is explicitly rejected.
`SOUND_REBUILD_PROMPT.md` bans Hardmax lookup carriers, so it is not an adoption
candidate even though its local coefficient rewrite is exact.

## Adjacency and structural floor investigation

The exact ArgMax candidate's complete parameter breakdown is:

```text
adj 900 + center_basis 20 + neighbor_map 6 + route_out 6
+ onehot_values 2 + depth 1 = 935
```

The 30x30 radius-one adjacency has 88 nonzero entries but exact matrix rank 30.
Consequences of the investigated replacement families:

- Exact two-factor Einsum reconstruction needs inner rank at least 30, hence
  at least `30*30 + 30*30 = 1800` dense factor elements, worse than 900.
- Eye/Scatter/Constant/Trilu-style construction materializes a float30x30
  output costing 3600 memory. Even the impossible one-output idealized floor
  would be `1143 - 900 + 3600 = 3843` before construction parameters.
- Conv or float Slice/Pad/shift must first materialize at least one
  single-channel 30x30 float tensor, also 3600 memory. Dynamic selected-color
  contraction prevents making Conv the only graph output.
- The exact bitset route needs at least two 15-bit halves: float packing 240
  bytes, int32 cast 240, then horizontal/vertical/intersection outputs at least
  720 bytes. Its 1200-byte subtotal already exceeds the 900 parameters removed,
  before any bit decoding or output routing.
- Direct sparse initializers are already known to fail full checker/strict
  Einsum inference; supported sparse Constant materialization costs 3600 memory
  (`agent_task192_sparse_173/REPORT.md`).

Within the measured shared-three-row-basis ArgMax+OneHot class, 1143 is the
floor found here. The two remaining 2x3 maps are rank 2. Materializing either
dynamic 2x10 factor pair costs 80 bytes to remove only six parameters, a net
regression of 74. This is an architecture-class floor, not a universal proof
over every possible ONNX graph, but all requested Conv/Slice/Pad/shift and
output-free Einsum routes fail the cost test before behavioral admission.

Reproducible measurements: `floor_analysis.json` and `analyze_floor.py`.

