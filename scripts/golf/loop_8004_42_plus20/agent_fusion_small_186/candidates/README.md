# Rejected candidates

The three ONNX files in this directory are audit controls only. Each removes a
one-byte dead output and is strict-lower, but each fails every known example in
`ORT_DISABLE_ALL` due to runtime buffer-shape errors. Do not merge them.

See `../REPORT.md` and `../audit/result.json`.
