# Split/Concat round-trip scan 255

All 400 authority graphs were checked for single-input `Concat` and exact
`Concat(Split(x))` round trips using every Split output once, in order, on the
same axis.  No such site exists in the current 8009.46 payloads.

Safe adoptees: **0**; projected gain: **+0.0**.  The reproducible scan and
machine result are `scan.py` and `scan.json`.
