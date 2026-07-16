# task023 two-stage 1x1 morphology experiment

The cost-1541 clean ranker was rebuilt conceptually as a 4x4 two-channel
`QLinearConv` followed by a 1x1 `QLinearConv`.  With 34 weights, three biases,
and a 72-byte hidden activation, the intended cost is below the immutable
task023 threshold 1622.

No model was emitted as an adoption candidate.  Across the retained search,
no quantization passed all 266 known examples.  The best integer network was
252/266 known and 2678/3000 (89.27%) on its disjoint generator holdout.  This
is below both mandatory gates, so ONNX profiling and large independent fresh
admission were correctly skipped.

The result also explains the limitation of the unimplemented proposal in the
prior task023 POLICY90 report: a 1x1 second layer adds nonlinearity but does not
expand the first layer's 4x4 receptive field, so it still cannot reliably
separate adjacent-stick false 2x2 anchors.  The next lane therefore tests a
spatial 2x3/3x2 second layer while retaining cost below 1622.

`search.py` and `search.json` contain the reproducible training, exact integer
simulation, stored-example gate, and best failed weights.  Root, stage, and
authority files were not changed.
