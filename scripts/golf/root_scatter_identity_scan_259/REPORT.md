# ScatterElements full-overwrite scan 259

All 400 authority graphs contain 117 `ScatterElements` nodes. Ten use constant
indices. None has a full, duplicate-free identity index tensor matching
data/updates/output shape, so no node is equivalent to `Identity(updates)` by
the conservative full-overwrite proof.

Safe adoptees: **0**; projected gain: **+0.0**. Reproduction and the complete
census are in `scan.py` and `scan.json`.
