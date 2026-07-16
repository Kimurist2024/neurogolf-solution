# Rejected artifacts

Files named `*_cost-1.onnx` were emitted by the first optimizer census when
the scorer returned its `-1` rejection sentinel.  They are invalid, are not
referenced as winners by the final `optimizer_scan.json`, and must not be
admitted or merged.
