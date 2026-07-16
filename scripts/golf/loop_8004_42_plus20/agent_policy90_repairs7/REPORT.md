# Policy-90 repair lane: tasks 096, 205, 328, 343

## Outcome

All four candidates are terminal rejects under the active gates. This lane
contributes **+0.000000**, builds no ZIP, and modifies no protected root file.

| task | current | lead / repaired | optimistic gain | decision |
|---:|---:|---:|---:|---|
| 096 | 1198 | 2000 | negative | reject: truthful repair is more expensive |
| 205 | 1042 | 937 | +0.106214 | reject: Hardmax/contraction and no private guarantee |
| 328 | 558 | 554 | +0.007194 | reject: 58-input Einsum and unstable margin |
| 343 | 173 | 172 | +0.005797 | reject: exact private-zero lineage, not guaranteed |

## task096

The requested “10 output channels with length-1 bias” diagnosis was reversed.
The cost-1111 SHA `cf777f93...` already has `B2[10]`; its static declaration
claims one output channel while runtime produces ten. It also has ten declared
versus runtime shape contradictions and cannot create a default-ORT session.

I independently repaired the cleaner cost-1128 SHA `d0260677...` by making
every two-axis `CenterCropPad` shape a real two-element vector and retaining a
defined `B2[10]`. The result is SHA `72e481ee...`, passes all 266 known cases
under both ORT modes with zero errors, passes strict inference, and has a safe
10/10 QLinearConv bias. Once its actual tensors are declared and remeasured,
however, the cost is **2000** (`memory=1643`, `params=357`), worse than the
current 1198. It is diagnostic only and is rejected.

## task205

The cost-937 SHA `bbfa8f5b...` remains known-complete and shape-truthful. Two
new independent 5,000-case seeds give identical decisions in both ORT modes:

- seed 90702051: 4908/5000 = 98.16%, errors 0;
- seed 90702052: 4913/5000 = 98.26%, errors 0.

It clears the numerical 90% floor, but contains a `Hardmax` and a 13-input
floating `Einsum`; retained catalog evidence also flags task205 as a
private-zero task. Since neither seed is 100%, it cannot use the user's
private-zero guarantee exception. Reject.

## task328

The cost-554 SHA `4d0fc526...` is structurally truthful, UB-free, and known
267/267 in both ORT modes. The retained independent dual probe is threshold
correct on 16/16 cases, but four cases per mode contain raw positives below
0.25; the minimum is `7.316870026530253e-11`. More importantly, the graph has
a 58-input `Einsum`, so it fails the explicit no-giant-contraction gate even
if threshold accuracy remains above 90%. Reject.

## task343

The cost-172 SHA `6ada3c41...` is structurally clean and known 266/266. Two new
independent 5,000-case seeds give 4965/5000 (99.30%) and 4968/5000 (99.36%),
with identical decisions in both ORT modes and zero runtime errors.

It is nevertheless byte-identical to both quarantined
`task343_*_cost172_private0_decoded.onnx` artifacts. The two fresh streams also
contain 35 and 32 semantic failures, so private-set passage cannot be
guaranteed. Reject.

Machine-readable evidence is in `result.json`; the two new full fresh audits
are `task205_fresh_dual_two_seeds.json` and
`task343_fresh_dual_two_seeds.json`.
