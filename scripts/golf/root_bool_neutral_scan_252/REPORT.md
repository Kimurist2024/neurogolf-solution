# Boolean neutral-constant scan 252

All 400 members of the immutable 8009.46 archive were scanned for exact
Boolean identities involving uniform initializer tensors:

- `And(x, true) -> x`;
- `Or(x, false) -> x`;
- `Xor(x, false/true) -> x/Not(x)`; and
- `Equal(x, true/false) -> x/Not(x)` for Boolean `x`.

Broadcasted outputs are preserved with `Expand` and a static shape vector.
Only one site exists: task101's previously admitted
`And(scalar, true[1,1,3,6]) -> Expand(scalar, [1,1,3,6])`, cost 5655 -> 5641.
It is algebraically identical to the task101 candidate already staged in
`others/71407`; its different SHA is serialization/name-only and it contributes
no additional gain.

New safe adoptees: **0**.  `scan.json` records the all-400 operator census,
strict inference result, costs, and SHA.
