# A38 exact CSE result

All 400 members of `submission_base_8000.46.zip` were scanned for byte-identical initializer aliases, normalized identical Constant payloads, and duplicate reachable deterministic subgraphs.

- Shape/lineage-safe models scanned: **112**
- Explicitly excluded unsafe-lineage models: **288**
- Lower-cost safe candidates: **0**
- Decision: **NO_ADOPTABLE_CANDIDATE**
- Score gain: **0**

The raw inventory found exact duplication only in task162, task165, task169, task233. All four are excluded by the requested policy: CenterCropPad lineage in every case, with lookup lineage additionally present in task165/task233. task162's 60 duplicate nodes are themselves CenterCropPad nodes. No identical Constant payload opportunity exists in any of the 400 authority models, and no safe model contains an initializer or reachable deterministic-subgraph CSE opportunity.

Because no lower-cost shape-truthful safe candidate exists, known-dual, fresh-dual, and external500 gates were not run. The authority ZIP SHA-256 remains `74cb9c4a96dcf9d1d1ca4ad1e0afdb67f5218c0a06ad8e896ad1654b6548f534` and no shared submission was modified.
