# target-mid19 strict eight-task report

## Outcome

One candidate is admissible: **task226 cost 375 -> 372**, projected score gain
`ln(375/372) = +0.008032171697262669`. The other seven tasks have no model at
the intersection of strict lower measured cost, complete known correctness,
true generator semantics, truthful runtime shapes, allowed graph structure,
and zero runtime errors.

The requested authority was `submission_base_8004.50.zip`. During the run the
authority advanced to `submission_base_8005.16.zip`; all eight member SHA-256
values are byte-identical between the two archives. The task226 result is
therefore **rebase-compatible with 8005.16**. No ZIP, CSV, root score pointer,
shared artifact, or handcrafted model was modified.

| task | baseline cost | current policy audit | result |
|---:|---:|---|---|
| 037 | 374 | shape cloak; default ORT 0/266 with 266 runtime errors | stop |
| 092 | 366 | shape cloak; cheaper archive lead only 221/265 known | stop |
| 132 | 312 | prohibited 47-input giant Einsum | stop |
| 159 | 293 | shape cloak; 22-input terminal Einsum; legal-fresh segfault lineage | stop |
| 218 | 329 | shape cloak; only cheaper archive model known-wrong | stop |
| 226 | 375 | truthful standard-domain incumbent | **accept 372** |
| 228 | 302 | shape-cloaked coordinate carrier | stop |
| 297 | 371 | truthful structural floor; cheaper histories all wrong/invalid | stop |

All pinned members pass full ONNX checking, strict shape inference with
`data_prop=True`, static-shape syntax, standard-domain, banned/nested/function,
and Conv-bias UB checks. That static acceptance is not treated as proof of
truthful execution. The independent runtime audits cited below expose the
shape-cloaked members, and the local dual-ORT audit finds task037 unusable under
default optimization.

## Accepted task226

Candidate:
`scripts/golf/loop_8004_42_plus20/agent_target_mid19/task226_sixbit.onnx`

- SHA-256: `852b6091385d97df6899e21304bf194440fb5cd3343385693093c24be0cb8203`
- measured baseline: memory 334, params 41, cost **375**
- measured candidate: memory 333, params 39, cost **372**
- serialized size: 2,889 bytes
- graph: 65 nodes; no Einsum, CenterCropPad, lookup/vectorizer, custom domain,
  function, nested graph, sequence, banned op, or Conv bias
- ops: 10 GatherElements, 10 Cast, 2 Not, 2 And, 37 Where, 2 Concat,
  1 BitwiseAnd, and 1 terminal QLinearConv

### Exact generator derivation

Task226 has a fixed 10x10 grid. The generator permits exactly 17 width
sequences and eight height sequences. Its eight possible interior separator
positions are not all informationally necessary: columns `1,2,3,5,6,8`
uniquely identify every one of the 17 legal width sequences. This is the
minimum possible probe count; no five-position subset separates all 17.

The candidate removes probes 4 and 7 and compiles the incumbent's column-code
semantics into reduced Boolean `Where` decisions over those six observed
separator/background bits. It contains no table initializer and does not fit
visible examples. The row decoder and truthful `[1,2,10,10]` terminal renderer
are unchanged. Twenty-five scalar decision outputs replace the removed probe
pairs, lowering memory by one byte, while the two removed index initializers
lower parameters by two.

### Admission evidence

- known100, ORT_DISABLE_ALL: **100/100**, runtime errors 0
- known100, default ORT: **100/100**, runtime errors 0
- all known, ORT_DISABLE_ALL: **133/133**, runtime errors 0
- all known, default ORT: **133/133**, runtime errors 0
- fresh seed `22650001`: **5000/5000** in each ORT mode, generation/runtime
  errors 0
- fresh seed `22650002`: **5000/5000** in each ORT mode, generation/runtime
  errors 0
- exhaustive private-domain proof: all `17 x 8 = 136` legal generator states
  are correct in both ORT modes and raw-bitwise equal to the pinned member
- full checker and strict `data_prop=True`: PASS
- truthful runtime shapes: all 65 exposed node outputs equal their inferred
  static shapes under both ORT modes
- margin: minimum positive 1.0; cells in `(0,0.25)`: 0
- standard domain, banned ops, functions, nested graphs, Conv bias UB: clean

The external validator's arbitrary 500-case differential is not an admission
gate for this true-rule candidate: those mutations include invalid task226
shapes such as 10x21. The complete 136-state generator proof is authoritative
and stronger than sampled off-schema equivalence.

Primary machine evidence is in `task226_strict_audit.json` and
`task226_known_validator.json`. The reproducible builder is
`build_task226_sixbit.py`.

## Rejected task audits

### task037 / `1f876c06`

The true rule completes length-3..7 diagonals between matching colored
endpoints. The cost-374 member uses 76 CenterCropPad nodes with deliberately
compact false declarations. It is 266/266 only with optimizations disabled;
default ORT produces 266 runtime errors. Prior true-rule fusion changes the
allocator schedule and fails, topology-preserving truthful Boolean exposure
costs 437,668, and harvested truthful/static models begin at cost 497. There
is no lower no-cloak reconstruction.

### task092 / `40853293`

The decoded rule completes unique-colored horizontal or vertical sticks. The
random generator contains a condition bug that makes fresh generation accept
only all-horizontal cases, while visible/archived known cases include vertical
sticks. The cost-366 member declares its output as `[1,10,1,1]` while ORT
returns `[1,10,30,30]` and similarly hides full input/index tensors. The
cheapest archived sub-baseline graph costs 340 but is only 221/265 known; it
cannot qualify for either known safety or the private true-rule exception.
Known-complete truthful history begins above the incumbent floor.

### task132 / `56ff96f3`

The true rule fills the rectangle between each same-color opposite-corner
pair. The current cost is parameter-only 312, but the entire solution is one
47-input giant Einsum, explicitly disallowed in this lane. Eight earlier
factor deletions at costs 282/287 score 0/267 in seven cases and 14/267 in one.
There is no intermediate memory to shave and no non-giant lower-cost rebuild.

### task159 / `6b9890af`

The true rule extracts the separate diagonally connected 3x3 sprite and
magnifies it inside the red frame. The cost-293 graph has eight runtime shape
contradictions and a 22-input terminal Einsum. Its compact lineage also has a
reproducible ORT segmentation fault on legal fresh input. The 35-model archive
screen finds three cheaper costs, 236/291/291, all known-wrong. Fresh testing
cannot rehabilitate wrong or crashing candidates.

### task218 / `90c28cc7`

The true rule compresses a rectangular quilt into its compact 2x2/2x3/3x2/3x3
color table. The incumbent GroupNormalization output is declared
`[1,1,1,1]` but executes as `[1,10,30,30]`; it is not a truthful parent. Its
prior exact fresh result is 4996/5000. The 29-model history screen has exactly
one cheaper member, cost 314, and it is known-wrong.

### task228 / `952a094c`

The true rule removes four inner-corner colors and places them at the
diagonally opposite outer corners. The cost-302 coordinate carrier hides four
to eight 16-way tensors behind false `[1,1]` CenterCropPad declarations. The
26-model history screen's only cheaper member costs 294 and is 0/266 known;
the complete-known alternative ties 302 and retains the cloak.

### task297 / `bd4472b8`

The true rule repeats the top-row color sequence twice below the gray row. The
incumbent is truthful and already costs 320 memory + 51 parameters = 371.
Thirty historical models were reprofiled; all eleven cost-370 scale-sharing
variants fail known gold. Trimming the zero Conv kernel column reaches 361
only with schema-invalid negative padding. Standard one-column alternatives
cost 484 (Slice) or 511 (Split/re-concat).

## Evidence index

- `baseline_known_static.json`: both archive member hashes, strict static
  structure, Conv UB, and dual-ORT known100/all-known results for all eight
- `task226_strict_audit.json`: exhaustive/fresh/runtime-shape proof
- `task226_known_validator.json`: independent cost and known validation
- `task092_archive340_validator.json`: explicit rejection of the cheapest
  task092 archive lead
- `result.json`: full eight-task machine disposition
- `winner_manifest.json`: the single mergeable model reference
- historical all-model/floor evidence:
  `scripts/golf/loop_7999_13/lane_a10/REPORT.md`,
  `lane_b5/REPORT.md`, `lane_c10/REPORT.md`, `lane_c12/REPORT.md`, and
  `lane_a33/REPORT.md`
