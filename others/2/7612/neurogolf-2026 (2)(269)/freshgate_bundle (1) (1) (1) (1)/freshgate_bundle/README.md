# NeuroGolf Fresh-Gate (overfit detector) — portable bundle

Detects ONNX nets that pass the visible examples but are **overfit** (fail the
hidden/private set). It does this by generating fresh instances from each
task's OWN generator (`inputs/arc-gen-repo/tasks/task_<hash>.py`, the same code
the organizer uses for the hidden set) and checking the net on them.

## Requirements
    pip install numpy onnx onnxruntime    # onnxruntime 1.24.x, onnx 1.21.x

## Run
    ./run_freshgate.sh 158 path/to/task158.onnx 500
    ./run_freshgate.sh --batch 158=a.onnx,280=b.onnx 500

## Reading the verdict (per task, JSON)
- decision: ADOPT (all checks pass) / REJECT
- fresh_total / fresh_fails: instances generated / failed   <-- key signal
- fresh_ok: true only if fresh_fails == 0
- lib_gold / official_gold: visible-gold (train+test+arc-gen) via our lib AND
  the organizer's official neurogolf_utils
- margin_stable, margin_min: raw-output margin check

## Rule of thumb (empirical)
- fresh_fails == 0           -> safe to adopt
- 0 < fresh_fails <= 0.5% K  -> borderline; confirm on the public LB
- fresh_fails  > 5% of K     -> overfit; reject (it WILL score 0 on the hidden set)

## Layout (paths are required; do not flatten)
    scripts/verify_fix.py            gate
    scripts/lib/{__init__,scoring}.py
    inputs/neurogolf-2026/neurogolf_utils/neurogolf_utils.py   official scorer
    inputs/arc-gen-repo/tasks/{common,task_<hash>}.py          generators
    inputs/neurogolf-2026/task<NNN>.json                       visible examples
    docs/golf/task_hash_map.json                               task->hash map
