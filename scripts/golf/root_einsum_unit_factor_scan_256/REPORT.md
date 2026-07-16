# Einsum all-one factor scan 256

The immutable 8009.46 archive contains 605 `Einsum` nodes and 277 all-one
initializers.  Every use of each all-one initializer was checked for removal
from the corresponding equation.  A factor is removable only when every
output label remains supplied by another operand and every removed-only
contracted label has extent one; the initializer must be removable from all
graph uses, not merely one occurrence.

No initializer satisfies those global conditions.  Therefore no candidate was
emitted and no numerical-order gate was needed.

Safe adoptees: **0**; projected gain: **+0.0**.  Reproduction and machine
evidence are in `scan.py` and `scan.json`.
