# Sparse initializer scan 149

Sixty-five zero-heavy, non-Einsum dense initializers with at least ten
potential stored-value savings were converted exactly to standard COO sparse
initializers.  Dense reconstruction was bit-identical, but every candidate
failed full/strict ONNX inference because ordinary operator tensor inputs do
not accept `sparse_tensor` types (or lose rank).  Safe adoptees: 0; gain
`+0.0`.  Evidence: `scan.json` and `scan.py`.
