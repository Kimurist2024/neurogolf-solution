# Wave422 — cost21–50 image-guided low-cost transfer

Scope: all authority tasks with cost 21–50 (44 tasks). The task-viz PNGs were inspected for the available members; the visual rules were classified as random-pixel placement, line/grid transforms, overlap/object recoloring, and connected-component transforms. Low-cost (1–20) templates were considered only where the input/output geometry and color-role rule matched.

No candidate is admitted in this wave. The broad historical archive scan was stopped after it began producing repeated ONNX Runtime shape-cloak/shape-mismatch diagnostics and a non-finite positive-array failure; no artifact passed the required canonical/full-checker, strict-shape, runtime, UB, fresh-accuracy, and small-positive gates. In particular, no unverified private-zero candidate is promoted.

Excluded throughout: known LB-black task070/task134/task202/task343; catalog/private-zero entries; existing adopted tasks task012/task023/task161/task175/task354/task355; and prior rejects.

Image reasoning summary: low-cost templates encode exact geometry and color-role changes, while the 21–50 group is dominated by random-pixel or component-specific rules. Direct template transfer would change semantics; therefore candidates are rejected unless fresh validation proves >=90% and strict lower cost. Root authority and scores were not modified.

