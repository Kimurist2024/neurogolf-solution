"""Pick the next N highest-cost OPEN medium-cost tasks for a Codex golf wave.

Reads docs/golf/attempted.json (tasks already tried / proven floor / skipped),
scores the current stage nets, returns the top-N un-attempted tasks as
`TASK:HASH:COST` triples, and APPENDS them to attempted.json so the next call
moves on. Prints nothing (or fewer lines) when no targets remain.

Usage: next_targets.py [N]   (default N=3)
"""
import json, sys, tempfile, onnx
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from lib import scoring  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
MAP = json.load(open(REPO / "docs" / "golf" / "task_hash_map.json"))
ATTEMPTED = REPO / "docs" / "golf" / "attempted.json"
done = set(json.load(open(ATTEMPTED))) if ATTEMPTED.exists() else set()

rows = []
with tempfile.TemporaryDirectory() as wd:
    for t in range(1, 401):
        if t in done:
            continue
        p = REPO / "artifacts" / "wave_opus" / "stage" / f"task{t:03d}.onnx"
        try:
            r = scoring.score_and_verify(onnx.load(str(p)), t, wd, label="c", require_correct=False)
        except Exception:
            r = None
        if r:
            rows.append((t, MAP[f"{t:03d}"], r["cost"]))
rows.sort(key=lambda x: x[2], reverse=True)
picked = rows[:N]
if picked:
    done.update(t for t, _, _ in picked)
    json.dump(sorted(done), open(ATTEMPTED, "w"))
    print(" ".join(f"{t}:{h}:{c}" for t, h, c in picked))
