# 8005.16 changed-member exact scan

**Safe adoptees: 0.**  The remaining 15 changed members were scanned for
byte-identical initializer aliases, output-unreachable payload, and isolated
internal `Identity` removal.

Only task187 produced a strict-static candidate.  Removing internal Identity
node 7 reduces the strict static estimate from 1566 to 1558, but the sanitized
candidate cannot create either the official cost-profiling session or the
default optimized ORT session: its `TopK` shape inference reports that the axis
has fewer than `k` elements.  The 8005.16 incumbent itself also fails the
mandatory default-ORT gate and contains runtime/declaration shape mismatches.
The candidate is therefore rejected before known or fresh scoring.

- Candidate: `candidates/task187_identity_007.onnx`
- SHA-256: `0e719feb02b141c481043f2397d7cac1a8f0058779276b287c3fba0c54e72110`
- Evidence: `scan_report.json`
- Gain counted: `+0.0`
