
===== BAND-SPECIFIC GUIDANCE (cost 1000-3000 -> goal <=1000) =====
Context: many incumbents in this band are DOCUMENTED FLOORS from months of
shaving campaigns (e.g. 264@1613, 131@1516, 250@1277, 265@1088, 275@1267,
330@1726, 062@1088, 086@1282, 112@1386-era, 125@1515, 281@1437-era). Node-level
shaving of these is usually exhausted. A win here almost always requires a
STRUCTURALLY DIFFERENT formulation:
  - a different decomposition of the rule (per-object/coordinate engine vs
    whole-grid morphology, or vice versa),
  - a terminal fused Einsum (memory=0) folding several stages,
  - moving work from initializer tables to closed-form arithmetic,
  - chained diff-1 CenterCropPad cloaks to shrink crop pipelines,
  - int8/bool initializers, f32 (NOT f16) crops of the one-hot input.
If after reading the incumbent you judge it already optimal for its structure,
try ONE alternative structure from scratch before giving up.

COST MODEL (exact): cost = params + memory footprint by STATIC shape inference
(every materialized intermediate tensor counts: numel * dtype bytes). Whole-grid
[1,10,30,30] f32 intermediates cost 36000 each -- crop to ROI early.

!!! PRIVATE-SET WARNING (today's LB-verified failures in THIS band) !!!
Candidates at cost 2959(t009), 1831(t365), 1215(t086), 1628(t096), 910(t192)
passed ALL visible examples AND fresh k=30 yet scored ZERO on the private set
(-17..18 pts each). Derive the rule from the GENERATOR (task_<hash>.py); never
encode example-specific coordinates/colors/sizes; verify >=1000 fresh instances
covering every generator branch. A sound net at 1400 beats an unsound one at
1000. Conv/ConvTranspose bias length MUST equal output channels (ORT UB).
Banned: Loop/Scan/NonZero/Unique/Compress/Sequence/sparse_initializer/dynamic
shapes. TopK only if unavoidable.

===== BAND OVERRIDE of the MOVE-ON POLICY (this band only) =====
The generic move-on rule above ("stop at your FIRST adoption") is OVERRIDDEN:
a 1-5 unit shave is NOT worth your slot in this band. Promote small wins via
try_candidate whenever you find them (they are free), but DO NOT stop and DO
NOT print MOVED_ON unless (a) your adopted cost is <= 1500, or (b) you cut
>= 25% off the incumbent. Otherwise keep attacking with structurally different
formulations until the session times out. Banking a tiny shave and leaving is
the WORST outcome for this band.
