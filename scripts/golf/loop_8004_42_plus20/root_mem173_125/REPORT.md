# task173 TopK dtype/memory audit

Authority SHA-256:
`a23d2448c52fe24e949b7758aa754feddfb93012b430fbb1dec10c3e5ce183bf`;
official cost 3525 (memory 3360, parameters 165).

The three TopK score tensors contain exact integral values.  Replacing the
float16 score domain with uint8/int8/int16/uint16 preserves mathematical
ordering and ties, and the uint8 graph passes ONNX full checker and strict
shape inference.  ORT 1.24, however, has no TopK(11) CPU implementation for
any of those four dtypes under either disabled or default optimization.  The
fully cast-elided uint8 candidate fails for the same reason.  These candidates
are runtime-error risks and are rejected.

Historical truthful reconstruction is known to cost 18494.  The authority is
private-zero lineage and fresh 99/100 rather than perfect, so behavior-changing
rebuilds are not admitted without LB proof.  Safe winner count is zero; no root
or staged model was changed.
