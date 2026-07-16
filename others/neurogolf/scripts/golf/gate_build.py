"""Fresh-gate (k=100) a wave's handcrafted nets, build a submission zip, and
promote the adopted ones into the canonical stage.

For each task: adopt its artifacts/handcrafted net ONLY if it is cheaper than the
current stage net AND passes verify_fix (fresh-0-fail + dual gold + margin).
Writes artifacts/final_submit/submission.zip (stage + adopted) and promotes the
adopted nets into artifacts/wave_opus/stage + rebuilds wave_opus/submission.zip.

Prints `ADOPTED <tasks> PROJ <score>` on success, `NONE` if nothing adopted.
Exit 0 = something to submit, 5 = nothing adopted (caller should skip submit).

Usage: gate_build.py TASK [TASK ...]
"""
import sys, math, zipfile, shutil, tempfile, json
from pathlib import Path
import onnx
REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from lib import scoring  # noqa: E402
import verify_fix  # noqa: E402

def sc(c): return max(1.0, 25.0 - math.log(max(1.0, c)))

STAGE = REPO / "artifacts" / "wave_opus" / "stage"
HC = REPO / "artifacts" / "handcrafted"
OUT = REPO / "artifacts" / "final_submit" / "submission.zip"
CANON = REPO / "artifacts" / "wave_opus" / "submission.zip"
BEST = REPO / "artifacts" / "best_score.json"

tasks = [int(x) for x in sys.argv[1:]]
adopted, gain = [], 0.0
with tempfile.TemporaryDirectory() as wd:
    for t in tasks:
        hc = HC / f"task{t:03d}.onnx"
        st = STAGE / f"task{t:03d}.onnx"
        if not hc.exists():
            continue
        s = scoring.score_and_verify(onnx.load(str(st)), t, wd, label="s", require_correct=False)
        if not s:
            continue
        v = verify_fix.verify_one(t, hc, 100)
        if v.get("decision") == "ADOPT" and v.get("cost") is not None and v["cost"] < s["cost"]:
            adopted.append(t)
            gain += sc(v["cost"]) - sc(s["cost"])

if not adopted:
    print("NONE")
    sys.exit(5)

# promote adopted to stage, build both zips
for t in adopted:
    shutil.copy2(HC / f"task{t:03d}.onnx", STAGE / f"task{t:03d}.onnx")
files = sorted(STAGE.glob("task*.onnx"))
assert len(files) == 400, f"expected 400, got {len(files)}"
assert not [f for f in files if f.stat().st_size > 1_509_949]
OUT.parent.mkdir(parents=True, exist_ok=True)
for path in (OUT, CANON):
    if path.exists():
        path.unlink()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, arcname=f.name)

try:
    base = json.load(open(BEST)).get("score", 0)
except Exception:
    base = 0
proj = round(base + gain, 2)
print(f"ADOPTED {','.join(map(str, adopted))} PROJ {proj}")
