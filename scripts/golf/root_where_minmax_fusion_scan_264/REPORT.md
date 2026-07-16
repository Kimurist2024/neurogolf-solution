# Comparison/Where Min-Max fusion scan 264

Among 1,039 `Where` nodes, 822 consume a direct comparison.  Exact branch
matching found one selector identity: task255
`Where(a > b, a, b) -> Max(a,b)`.  Its comparison output is shared elsewhere,
so the producer and all initializers remain live; official cost stays 1133.

No `Equal`-selector identity or strict-lower Min/Max fusion exists. Safe
adoptees: **0**; projected gain: **+0.0**. Evidence is in `scan.py` and
`scan.json`.
