# task086 shared negative-PRelu scan 161

The five task086 `PRelu` nodes sharing scalar float16 slope `-1` were
exhausted over all31 nonempty subsets, using both exact `Abs` and
`LeakyRelu(alpha=-1)` replacements (62 profiles total). Every candidate passes
full/strict checks, but none is cheaper than the authority under the profiler;
even replacing all five and dropping the shared initializer increases enough
runtime memory to erase the one-parameter saving.

Safe adoptees/probes: **0**; projected gain: **+0.0**. Evidence: `scan.py`,
`scan.json`.

