# A34 exact initializer-sharing report

Against `submission_base_8000.46.zip` SHA-256
`74cb9c4a96dcf9d1d1ca4ad1e0afdb67f5218c0a06ad8e896ad1654b6548f534`,
task398 has one strict winner: cost **350 -> 347**, projected score gain
**+0.0086083745366**.

All five `Q0..Q4` vectors and every `K` occurrence share the same three-state
Einsum axis.  Applying the common diagonal gauge
`G=[1/Q4[0],-1,-1]` to every Q and `G^-1` to K leaves all contractions
unchanged.  The transformed Q4 is exactly `[1,-1,-1]`, so it replaces every D
operand and deletes D's three parameters.  Node count, Einsum input count,
indices, Resize topology, and intermediate shapes are unchanged.

The candidate passes known 268/268 under both ORT modes, fresh generator
5000/5000 under both modes with zero errors, official-like cost/correctness,
full checking, strict shape inference/data propagation, truthful runtime
shapes, and margin stability (minimum 21.39396286).  The external validator
reports `ACCEPT_STRICT`; its four differences on 500 arbitrary out-of-generator
grids are allowed and do not occur in the generator-domain 10,000-run gate.

The lane ZIP SHA-256 is
`80b4bfc09e505b6a1073895ed575fd501b3ffb63f0d9be49d41dfa486da20ced`.
It is a valid 400-member archive, changes only `task398.onnx`, and has zero Conv
bias-length UB findings.

task099's proposed DBc/ST_v gauge reached nominal cost 392 but only 21/100
fresh in each ORT mode.  Float32 cancellation changed intended zero logits, so
it is rejected and task099 has no safe winner.
