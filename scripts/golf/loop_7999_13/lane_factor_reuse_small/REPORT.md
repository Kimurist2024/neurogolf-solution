# Exact factor-reuse lane

The strict winner is task137.  Its cost decreases from `260` to `258`, for a
predicted gain of `+0.007722046093910251`.

The target float16 initializer `CA_h[2,3]` lies exactly in the row basis of the
already-stored `CM_h[2,3]`.  The candidate replaces it inside the existing
Einsum with a `2x2` transform and the existing source factor.  Reconstruction
in the serialized float16 dtype has zero error, removes two parameter elements,
adds no runtime node, and leaves runtime memory at 196.

The candidate passes all 266 known cases, both repository gold paths, 5000 of
5000 fresh generator cases, and the independent team validator.  It has no
candidate runtime errors and a minimum decision margin of `1.7275390625`.
The exact baseline/root submission and score ledgers were not changed.
