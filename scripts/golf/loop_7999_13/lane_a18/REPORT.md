# A18 archive-all400 strict retest

## Result

Two candidates survive every assigned gate:

| Task | Exact cost | Winner cost | Reduction | Projected score gain |
|---:|---:|---:|---:|---:|
| 063 | 26 | 24 | 2 | +0.08004270767353816 |
| 139 | 52 | 50 | 2 | +0.039220713153280684 |

Combined projected gain is **+0.11926342082681884**. No root ZIP, CSV,
score pointer, handcrafted model, or ledger was modified.

## Admission evidence

Both winners are exact one-node tensor-factor graphs with no charged runtime
memory; their actual and static costs agree. They pass:

- complete known examples under ORT_DISABLE_ALL and default ORT: task063
  266/266 in each mode, task139 265/265 in each mode;
- independent generator seed 71418063/71418139: 5000/5000 in each ORT mode,
  wrong 0, runtime errors 0, decoded output equal to the exact baseline on all
  5000 cases;
- full ONNX checker and strict shape inference with data propagation;
- standard-domain, function, sparse/external initializer, nested-graph,
  banned-op, Conv-bias, non-static-shape, shape/value-cloak, UB, and lookup
  gates.

The candidate files are byte-identical to their sole recorded loose sources:

- task063: `others/2/7908/task063_improved_cost24.onnx`
- task139: `others/2/7907/task139_improved.onnx`

Neither source is quarantine/private-zero named. The graphs contain only one
Einsum and small factor initializers (24 and 50 elements), no value_info, no
spatial/output lookup tensor, and no runtime carrier. task063 reduces the
incumbent contraction from 29 to 26 operands; task139 changes its exact factor
representation from 13 to 16 operands while removing the two-element `N`
initializer. These operand counts are recorded explicitly because they exceed
the generic archive-harvest cutoff, but they are not lookup tables and were not
forbidden by the A18 assignment.

## Rejections

### task073 r01/r02

Both cost-15 candidates pass known examples, but their `ConvTranspose` output
has ten channels while the bias initializer has only nine elements. This is
the host-confirmed short-bias out-of-bounds condition and is a terminal UB
reject even when a particular process happens to produce correct output.

The two candidate hashes differ only because r01 stores initializers in
`raw_data` and uses graph name `g`, while r02 uses typed float fields and graph
name `task073_cost15`. After normalizing metadata and TensorProto storage, the
executable graphs are identical.

### task202 r02/r03

Both cost-28 candidates pass all 230 known examples and the structural gates,
but fail the independent fresh run:

| Variant | ORT_DISABLE_ALL | Default ORT | Runtime errors |
|---|---:|---:|---:|
| r02 | 4862/5000 | 4862/5000 | 0 / 0 |
| r03 | 4993/5000 | 4993/5000 | 0 / 0 |

r02 has 138 counterexamples per mode; r03 has seven. The generator required
5817 attempts to yield 5000 convertible valid examples, with 817 conversion
skips and no generator exceptions. r02 also has one byte-identical source
whose filename is quarantine/private0-marked, while r03 has clean lineage;
the clean lineage does not override its observed counterexamples. Neither is
eligible.

## Files

- `winner_manifest.json`: exact winner/rejection disposition
- `candidate_audit.json`: costs, complete known, structure, UB, cloak, lookup,
  and executable differences
- `fresh_dual_5000.json`: independent same-case dual-ORT differential
- `lineage_audit.json`: source resolution and byte hashes
- `model_manifest.json`: exact archive and candidate identities
