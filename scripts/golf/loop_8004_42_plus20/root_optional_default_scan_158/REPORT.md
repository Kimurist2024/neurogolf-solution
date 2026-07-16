# Optional-default initializer scan 158

All400 immutable 8009.46 members were scanned for exact omission of ONNX
optional inputs already equal to their defaults:

- equal-size `Split` input -> `num_outputs` attribute (opset >=18);
- default `Slice` axes and/or all-one steps;
- zero `Pad` constant value and/or all-axis list.

Forty candidate profiles were generated. Twenty-four fail full checker or
strict data propagation after the omission, and none of the remaining sixteen
has a lower official-like profile.  No candidate reached runtime admission.

Safe adoptees/probes: **0**. Projected gain: **+0.0**. Root ZIP, score ledger,
and staged candidate directory were not modified.

Evidence: `scan.py`, `scan.json`.

